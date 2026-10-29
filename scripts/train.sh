#!/usr/bin/env bash
# Training launcher of the paper's runs: Countdown, GRPO without std normalization, verl 8718ca30.
#
#   MODEL=Qwen/Qwen3-0.6B METHOD=dlc SOURCE=stale SEED=0 bash scripts/train.sh [extra hydra overrides ...]
#
# Data first: python scripts/build_countdown_data.py   (-> data/countdown/{train,heldout}.parquet)
#
# METHOD (paper name -> internal setting)
#   on_policy      policy gradient with the on-policy BF16 sampler (needs SOURCE=none)                pg
#   pg             policy gradient (REINFORCE with group-mean baseline, no ratio)                    pg
#   ppo            PPO with batch reference: clip 0.2 on p_theta / p_theta_batch, 4 updates per batch ppo (MINI=16)
#   ppo_noclip     PPO without clip: the same 4 updates per batch, no clip, no ratio                 pg4 (MINI=16)
#   sc             score centering, verl's top-128 head                                              sc
#   is             truncated importance sampling, token ratio p/q capped at 2                         tis
#   is_sc          importance sampling with centering                                                tis_sc
#   is_resync      importance sampling with resynchronization when ESS of p/q < ESS_MIN               tis + budget
#   m2po           M2PO (second-moment trust region on the ratio to the sampler's weights)           m2po
#   vcpo           VCPO (sequence IS at 8, ESS learning-rate scale, optimal off-policy baseline)     vcpo
#   dlc            DLC: clip of the lag ratio against the sampler's weights, centering term C_t,
#                  resynchronization when the ESS of the lag ratio falls below ESS_MIN               split + budget
#   dlc_noresync   DLC without resynchronization                                                     split + stale64
#   dlc_async      DLC under continuous asynchrony, clip against each token's generating version     split_async
#   dlc_admission  dlc_async + ESS admission: a response with ESS < ESS_MIN is deferred, regenerated  split_async_ess
# SOURCE (mismatch source of the sampler)
#   none           on-policy BF16 vLLM sampler                                                       bf16
#   quantized      FP8 weights (128x128 blocks), dynamic FP8 activations, FP8 KV cache               fp8
#   stale          BF16 sampler synchronized every 64 steps (keeps theta_0 for 40 steps)            stale64
#   quantized_stale  the stale weights theta_0, quantized as in `quantized`                          fp8_stale64
#   async          continuous asynchrony: weights pushed after every step, partial rollout,
#                  maximum version gap ASYNC_GAP=4 (two GPUs: one trainer, one sampler)             async4
# is_resync and dlc take SOURCE=stale or quantized_stale; the sampler then starts at theta_0 and is resynchronized
# instead of synchronized every 64 steps (internal SOURCE budget / fp8_budget).
#
# Other variables: SEED (0; vLLM seed 1000 + SEED), STEPS (40), SAVE_FREQ (5; 1 for the on-policy reference runs that
# the replay and deviation measurements read), LR (1e-2; 5e-3 for the learning-rate check), N_GPUS (1; 8 for Qwen3-8B),
# ESS_MIN (0.5), KERNELS (fa3_liger | sdpa), OUT_DIR (outputs/), RUN_PREFIX (run name prefix, see README), EXP (run name),
# TRAINER_LOGGER ([console,file]), DRY=1 (print the command and exit).
#
# Setup -> verl key mapping (keys verified at 8718ca30)
#   64 prompts x 8 samples           data.train_batch_size=64, rollout.n=8
#   one optimizer step per batch     actor.ppo_mini_batch_size=64 (ppo, ppo_noclip: 16 -> 4 updates), ppo_epochs=1
#   prompt / response lengths        max_prompt_length 135 (longest Countdown prompt), max_response_length 377,
#                                    rollout.max_model_len 512, data.truncation=error
#   chat template, no thinking       +data.apply_chat_template_kwargs.enable_thinking=False
#   fixed data order                 data.shuffle=False, dataloader_num_workers=0
#   sampling T=1, no truncation      rollout.temperature=1, top_p=1, top_k=-1
#   SGD, no momentum/warmup/decay    optim.optimizer=SGD (torch.optim; only lr and weight_decay are passed: momentum 0),
#                                    lr_warmup_steps_ratio=0, weight_decay=0, constant schedule
#   no gradient clipping             optim.clip_grad=1e9 (verl always calls clip_grad_norm_; 1e9 makes it the identity)
#   BF16 compute, FP32 master        fsdp_config.model_dtype=fp32, fsdp_config.dtype=bfloat16
#   advantage = reward - group mean  algorithm.adv_estimator=grpo, norm_adv_by_std_in_grpo=False, gamma=1, no KL
#   rewards +-1                      tim/tasks/countdown_reward.py:compute_score
#   token-mean loss                  actor.loss_agg_mode=token-mean (global token count)
#   top-128 sampler head             rollout.topk_log_probs=128 (sc, is_sc, dlc*)
#   sampler                          vLLM, TRITON_ATTN attention, CUDA graphs, calculate_log_probs=True
#   trainer kernels (fa3_liger)      FSDP2, FlashAttention-3, packed sequences, Liger (Triton RoPE), gradient
#                                    checkpointing, dynamic micro-batches of 16384 tokens, verl's fused log-prob kernels
#                                    off (score centering and DLC need the logits)
# Known differences to the reference implementation of Countdown + score centering: verl caps every response at 377
# tokens instead of 512 - len(prompt), and drops the last partial batch of an epoch.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
METHOD=${METHOD:?set METHOD (on_policy, pg, ppo, ppo_noclip, sc, is, is_sc, is_resync, m2po, vcpo, dlc, dlc_noresync, dlc_async, dlc_admission)}
SOURCE=${SOURCE:-none}
SEED=${SEED:-0}
STEPS=${STEPS:-40}
SAVE_FREQ=${SAVE_FREQ:-5}
N_GPUS=${N_GPUS:-1}
MICRO_BS=${MICRO_BS:-16}
ESS_MIN=${ESS_MIN:-0.5}
KERNELS=${KERNELS:-fa3_liger}
PY=${PY:-python}

# ---- paper names -> internal method / source --------------------------------------------------------------------------
case "$SOURCE" in
  none) S=bf16 ;; quantized) S=fp8 ;; stale) S=stale64 ;; quantized_stale) S=fp8_stale64 ;; async) S=async4 ;;
  *) echo "SOURCE=$SOURCE: none, quantized, stale, quantized_stale or async" >&2; exit 2 ;;
esac
need_stale() { case "$SOURCE" in stale|quantized_stale) ;; *) echo "METHOD=$METHOD needs SOURCE=stale or quantized_stale (DLC without a policy lag reduces to score centering)" >&2; exit 2 ;; esac; }
case "$METHOD" in
  on_policy)     [ "$SOURCE" = none ] || { echo "METHOD=on_policy needs SOURCE=none" >&2; exit 2; }; M=pg ;;
  pg)            M=pg ;;
  ppo)           M=ppo ;;
  ppo_noclip)    M=pg4 ;;
  sc)            M=sc ;;
  is)            M=tis ;;
  is_sc)         M=tis_sc ;;
  is_resync)     need_stale; M=tis; S=${S/stale64/budget} ;;
  m2po)          M=m2po ;;
  vcpo)          M=vcpo ;;
  dlc)           need_stale; M=split; S=${S/stale64/budget} ;;
  dlc_noresync)  need_stale; M=split ;;
  dlc_async)     [ "$SOURCE" = async ] || { echo "METHOD=dlc_async needs SOURCE=async" >&2; exit 2; }; M=split_async ;;
  dlc_admission) [ "$SOURCE" = async ] || { echo "METHOD=dlc_admission needs SOURCE=async" >&2; exit 2; }; M=split_async_ess ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

# run name: <prefix>_<internal source>_<internal method>_seed<k>, the form the measurement scripts parse
if [ -z "${RUN_PREFIX:-}" ]; then
  case "$MODEL" in *0.6B*) RUN_PREFIX=s2 ;; *1.7B*) RUN_PREFIX=s5 ;; *8B*) RUN_PREFIX=s8b ;; *) RUN_PREFIX=run ;; esac
fi
EXP=${EXP:-${RUN_PREFIX}_${S}_${M}_seed${SEED}}
OUT_DIR=${OUT_DIR:-$REPO/outputs}
OUT=$OUT_DIR/$EXP
TRAIN_FILE=${TRAIN_FILE:-$REPO/data/countdown/train.parquet}
VAL_FILE=${VAL_FILE:-$REPO/data/countdown/heldout.parquet}
REWARD=$REPO/tim/tasks/countdown_reward.py
[ -f "$TRAIN_FILE" ] || { echo "no $TRAIN_FILE: run python scripts/build_countdown_data.py first" >&2; exit 2; }

# the model as a local directory (a Hugging Face id is downloaded once into the HF cache)
if [ -d "$MODEL" ]; then MODEL_PATH=$MODEL
else MODEL_PATH=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"); fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
MODULES=()      # hook modules of the method; verl imports all hook modules in every process (VERL_USE_EXTERNAL_MODULES)
SRC_MODULES=()  # hook modules of the source (imported before the method's, so their patches wrap first)

# ---- trainer kernels ----------------------------------------------------------------------------------------------------
case "$KERNELS" in
  fa3_liger)
    # FlashAttention-3 (H100, sm_90a): built from source, see envs/README.md; FA3_PATH adds its build dir to the path
    if [ -n "${FA3_PATH:-}" ]; then export PYTHONPATH="$FA3_PATH:$PYTHONPATH"; fi
    export LIGER_KERNEL_IMPL_ROPE=${LIGER_KERNEL_IMPL_ROPE:-nvidia-triton}
    KERNEL_ARGS=(+actor_rollout_ref.model.override_config.attn_implementation=${TRAINER_ATTN:-flash_attention_3}
                 actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.model.enable_gradient_checkpointing=true
                 actor_rollout_ref.model.use_liger=true actor_rollout_ref.actor.strategy=fsdp2
                 actor_rollout_ref.model.use_fused_kernels=false actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
                 actor_rollout_ref.actor.use_fused_kernels=false
                 actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKEN_BUDGET:-16384}
                 actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true
                 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOKEN_BUDGET:-16384}) ;;
  sdpa)
    # SDPA on padded micro-batches, FSDP1; no FA3 needed. DLC with resynchronization, the async source and the
    # centering term of DLC need fa3_liger (FSDP2 for the reference reload and the version snapshots, packed logits).
    KERNEL_ARGS=(actor_rollout_ref.model.use_fused_kernels=False actor_rollout_ref.actor.use_fused_kernels=False
                 +actor_rollout_ref.model.override_config.attn_implementation=sdpa
                 actor_rollout_ref.model.use_remove_padding=False) ;;
  *) echo "KERNELS=$KERNELS: fa3_liger or sdpa" >&2; exit 2 ;;
esac

# ---- method ------------------------------------------------------------------------------------------------------------
MINI=64; BYPASS=true; LOSS_MODE=bypass_mode; KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False); CLIP_ARGS=(); LOSS_AGG=token-mean
rollout_is_threshold=2.0
NO_CLIP=(actor_rollout_ref.actor.clip_ratio_low=1e9 actor_rollout_ref.actor.clip_ratio_high=1e9 actor_rollout_ref.actor.clip_ratio_c=1e9)
ASYNC_OLD=metrics
case "$M" in
  pg)     rollout_is=null;  score_centering=false; topk=0 ;;
  tis)    rollout_is=token; score_centering=false; topk=0 ;;
  sc)     rollout_is=null;  score_centering=true;  topk=128 ;;
  tis_sc) rollout_is=token; score_centering=true;  topk=128 ;;
  split)  # DLC: verl's decoupled mode, vanilla PPO clip 0.2 (dual clip 3) on p_theta / p_theta_old with p_theta_old from
          # the reference worker (use_kl_loss=True, kl_loss_coef=0); the centering term C_t added by tim.dlc.hooks; the
          # config keeps rollout_is=null and score_centering=false, the sampler's top-128 head comes from topk_log_probs
          rollout_is=null; score_centering=false; topk=128
          BYPASS=false; LOSS_MODE=vanilla
          KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.0)
          export TIM_SPLIT_OLD=ref TIM_SPLIT_MAX_STEP=$STEPS TIM_SPLIT_ENGINE=sc
          export TIM_SPLIT_SKIP_KL_REF=${TIM_SPLIT_SKIP_KL_REF:-0}
          MODULES+=(tim.dlc.hooks) ;;
  ppo)    # PPO with batch reference: decoupled mode, clip 0.2 on p_theta / p_theta_batch (the trainer before this batch's
          # updates), no IS, no centering; 4 mini-batches per batch so that the clip can act
          rollout_is=null; score_centering=false; topk=0
          BYPASS=false; LOSS_MODE=vanilla; MINI=16 ;;
  pg4)    # PPO without clip: policy gradient with the same 4 mini-batches per batch as ppo
          rollout_is=null; score_centering=false; topk=0; MINI=16 ;;
  m2po)   # M2PO: p_theta_old = the sampler's weights through tim.dlc.hooks (no discrepancy factor), PPO clip off, the
          # M2PO 0/1 mask written as rollout_is_weights by tim.baselines.hooks; SOURCE=async: p_theta_old per token
          case "$S" in stale*|fp8_stale*|async4) ;; *) echo "METHOD=m2po needs SOURCE=stale, quantized_stale or async" >&2; exit 2 ;; esac
          rollout_is=null; score_centering=false; topk=0
          BYPASS=false; LOSS_MODE=vanilla; CLIP_ARGS=("${NO_CLIP[@]}")
          if [ "$S" = async4 ]; then
            ASYNC_OLD=version
            export TIM_SPLIT_ENGINE=none TIM_BASELINE=m2po
            MODULES+=(tim.baselines.hooks)
          else
            KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.0)
            export TIM_SPLIT_OLD=ref TIM_SPLIT_MAX_STEP=$STEPS TIM_SPLIT_ENGINE=none TIM_BASELINE=m2po
            MODULES+=(tim.dlc.hooks tim.baselines.hooks)
          fi ;;
  vcpo)   # VCPO: decoupled (p_theta_old = theta_t), sequence IS at 8, PPO clip off, seq-mean-token-mean, ESS learning-rate
          # scale and the optimal off-policy baseline (TIM_VCPO_OPOB=1) by tim.baselines.hooks
          rollout_is=sequence; score_centering=false; topk=0; rollout_is_threshold=8.0
          BYPASS=false; LOSS_MODE=vanilla; CLIP_ARGS=("${NO_CLIP[@]}"); LOSS_AGG=seq-mean-token-mean
          export TIM_BASELINE=vcpo TIM_VCPO_OPOB=${TIM_VCPO_OPOB:-1}
          MODULES+=(tim.baselines.hooks) ;;
  split_async|split_async_ess)
          # DLC under continuous asynchrony: clip 0.2 on p_theta / p_theta_k(t) (the generating version of each token),
          # the centering term on the sampler's top-128 head with p_old = p_theta_k(t), no reference worker
          rollout_is=null; score_centering=false; topk=128
          BYPASS=false; LOSS_MODE=vanilla; ASYNC_OLD=version
          export TIM_SPLIT_OLD=async TIM_SPLIT_ENGINE=sc
          if [ "$M" = split_async_ess ]; then export TIM_ASYNC_ESS_MIN=$ESS_MIN; fi
          MODULES+=(tim.dlc.hooks) ;;
esac

# ---- source ------------------------------------------------------------------------------------------------------------
SRC=()
FP8=(actor_rollout_ref.rollout.quantization=fp8 +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8)
case "$S" in
  bf16) ;;
  fp8)  SRC=("${FP8[@]}") ;;
  stale64|fp8_stale64|budget|fp8_budget)
        # The sampler keeps its weights between synchronizations: free_cache_engine=False keeps the colocated vLLM
        # weights and KV pool resident (hence the smaller memory share); tim.dlc.resync skips the weight push.
        # quantized_stale: the step-0 synchronization loads theta_0 through vLLM's quantizing loader and the skipped
        # pushes never reach it, so the FP8 weights of theta_0 stay resident.
        SRC=(actor_rollout_ref.rollout.free_cache_engine=False
             actor_rollout_ref.rollout.gpu_memory_utilization=${STALE_GPU_MEM_UTIL:-0.3})
        case "$S" in fp8_*) SRC+=("${FP8[@]}") ;; esac
        case "$S" in
          *stale64) export TIM_SYNC_MODE=stale TIM_STALENESS=64 ;;
          *budget)  # resynchronize after a step whose chi2 = E[r^2] - 1 exceeds 1/ESS_MIN - 1 (ESS < ESS_MIN)
                    export TIM_SYNC_MODE=budget TIM_CHI2_BUDGET=$(awk -v e="$ESS_MIN" 'BEGIN { printf "%.10g", 1 / e - 1 }') ;;
        esac
        export TIM_STALE_SLEEP1=0
        SRC_MODULES+=(tim.dlc.resync) ;;
  async4)
        # verl's v1 separate-async trainer on two GPUs (one trainer GPU, one standalone rollout GPU, hybrid_engine=False),
        # weights pushed over NCCL after every step (parameter_sync_step=1), partial rollout (abort, push, resume), maximum
        # version gap ASYNC_GAP with the 'wait' strategy, ASYNC_WARMUP batches in flight, vLLM max_num_seqs
        # ASYNC_MAX_NUM_SEQS (64: the generation of a batch outlasts a training step, so responses span pushes). Every
        # token's generating version is recorded by tim.dlc.async_hooks.
        case "$M" in pg|tis|m2po|split_async|split_async_ess) ;; *) echo "SOURCE=async runs pg, is, m2po, dlc_async or dlc_admission" >&2; exit 2 ;; esac
        [ "$KERNELS" = fa3_liger ] || { echo "SOURCE=async needs KERNELS=fa3_liger (FSDP2 for the per-version snapshots)" >&2; exit 2; }
        [ "$N_GPUS" = 2 ] || { echo "SOURCE=async needs N_GPUS=2 (one trainer GPU + one rollout GPU), got $N_GPUS" >&2; exit 2; }
        N_GPUS=1   # trainer.n_gpus_per_node: the trainer's GPU; the second one is the standalone rollout's
        SRC=(trainer.v1.trainer_mode=separate_async
             trainer.v1.separate_async.parameter_sync_step=1
             trainer.v1.separate_async.num_warmup_batches=${ASYNC_WARMUP:-3}
             trainer.v1.sampler.max_off_policy_threshold=${ASYNC_GAP:-4}
             trainer.v1.sampler.max_off_policy_strategy=wait
             actor_rollout_ref.hybrid_engine=False
             actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=1
             actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
             actor_rollout_ref.rollout.max_num_seqs=${ASYNC_MAX_NUM_SEQS:-64})
        export TIM_ASYNC=1 TIM_ASYNC_OLD=$ASYNC_OLD TIM_ASYNC_KEEP=${ASYNC_KEEP:-5}
        SRC_MODULES+=(tim.dlc.async_hooks) ;;
esac
case "$S:$M" in
  budget:tis|budget:split|fp8_budget:tis|fp8_budget:split|stale64:*|fp8_stale64:*|bf16:*|fp8:*|async4:*) ;;
  *) echo "unsupported combination METHOD=$METHOD SOURCE=$SOURCE" >&2; exit 2 ;;
esac
case "$M" in split|split_async|split_async_ess) [ "$KERNELS" = fa3_liger ] || { echo "METHOD=$METHOD needs KERNELS=fa3_liger" >&2; exit 2; } ;; esac

# several runs on one node: one colocate ZMQ socket per run (tim/zmq_tag.py)
export TIM_RUN_TAG=$(printf %s "$EXP" | cksum | cut -d' ' -f1 | cut -c1-8)
MODULES=(tim.zmq_tag ${SRC_MODULES[@]+"${SRC_MODULES[@]}"} ${MODULES[@]+"${MODULES[@]}"})
VERL_USE_EXTERNAL_MODULES=$(IFS=,; echo "${MODULES[*]}")${VERL_USE_EXTERNAL_MODULES:+,$VERL_USE_EXTERNAL_MODULES}
export VERL_USE_EXTERNAL_MODULES

# ---- environment of the paper's runs (single node) ---------------------------------------------------------------------
export VLLM_USE_V1=1 VLLM_USE_DEEP_GEMM=0 VLLM_SKIP_DEEP_GEMM_WARMUP=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export PYTHONUNBUFFERED=1
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_$(printf %s "$EXP" | cksum | cut -d' ' -f1)}   # short: unix socket paths <= 107 bytes

mkdir -p "$OUT/ckpt" "$OUT/logs"
ATTEMPT=1   # verl's file logger truncates its file on open: one metrics file per attempt
while [ -e "$OUT/logs/metrics.$ATTEMPT.jsonl" ]; do ATTEMPT=$(( ATTEMPT + 1 )); done
export VERL_FILE_LOGGER_PATH=$OUT/logs/metrics.$ATTEMPT.jsonl

# the saved hf_model's config.json declares the initial model's full eos list ([<|im_end|>, <|endoftext|>] for Qwen3)
EOS_ARGS=()
if [ "${EOS_OVERRIDE:-1}" = 1 ]; then
  EOS=$(PYTHONPATH="$REPO/scripts:$PYTHONPATH" "$PY" -c "import json; from convert_ckpt import init_eos_ids; print(json.dumps(init_eos_ids('$MODEL_PATH')).replace(' ', ''))")
  EOS_ARGS=("++actor_rollout_ref.model.override_config.eos_token_id=$EOS")
fi

echo "train.sh: $EXP  (METHOD=$METHOD SOURCE=$SOURCE -> $M / $S, model $MODEL, $STEPS steps, checkpoints every $SAVE_FREQ in $OUT/ckpt, hooks ${VERL_USE_EXTERNAL_MODULES})"
DRYRUN=()
[ "${DRY:-0}" = 1 ] && DRYRUN=(printf '%s\n')
${DRYRUN[@]+"${DRYRUN[@]}"} "$PY" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" data.val_files="$VAL_FILE" \
  data.train_batch_size=64 data.max_prompt_length=135 data.max_response_length=377 \
  data.truncation=error data.shuffle=False data.dataloader_num_workers=0 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=$BYPASS algorithm.rollout_correction.loss_type=reinforce \
  algorithm.rollout_correction.rollout_is=$rollout_is \
  algorithm.rollout_correction.rollout_is_threshold=$rollout_is_threshold \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.score_centering=$score_centering \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  "${KERNEL_ARGS[@]}" \
  "${KL_ARGS[@]}" \
  actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
  +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=$BYPASS \
  +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce \
  +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=$rollout_is \
  +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=$rollout_is_threshold \
  +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=$score_centering \
  actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG \
  ${CLIP_ARGS[@]+"${CLIP_ARGS[@]}"} \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.optimizer=SGD \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-2} \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.clip_grad=1e9 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=${VLLM_ATTN_BACKEND:-TRITON_ATTN} \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_model_len=512 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.topk_log_probs=$topk \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BS \
  actor_rollout_ref.rollout.seed=$(( 1000 + SEED )) \
  ${SRC[@]+"${SRC[@]}"} \
  ${EOS_ARGS[@]+"${EOS_ARGS[@]}"} \
  reward.custom_reward_function.path="$REWARD" \
  reward.custom_reward_function.name=compute_score \
  trainer.logger="${TRAINER_LOGGER:-[console,file]}" \
  trainer.project_name=${PROJECT:-tim_countdown} trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node=$N_GPUS trainer.nnodes=1 \
  trainer.total_epochs=5 trainer.total_training_steps=$STEPS \
  trainer.save_freq=$SAVE_FREQ trainer.max_actor_ckpt_to_keep=null \
  trainer.default_local_dir="$OUT/ckpt" trainer.resume_mode=auto trainer.rollout_data_dir="$OUT/samples" \
  trainer.test_freq=-1 trainer.val_before_train=False trainer.balance_batch=False \
  "$@"
