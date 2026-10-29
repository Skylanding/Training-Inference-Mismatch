"""Pure-python tests of the stale-sampler schedule and of the resynchronization rule (tim/dlc/resync_config.py)."""
import pytest

from tim.dlc import resync_config as sc


def test_parse_source():
    assert not sc.parse_source("bf16").active and not sc.parse_source("fp8").active
    s = sc.parse_source("stale64")
    assert (s.mode, s.staleness, s.refresh_period) == ("stale", 64, 65)   # refresh period = staleness + 1
    assert sc.parse_source("fp8_stale64") == s
    b = sc.parse_source("budget", chi2_budget=2.5)
    assert (b.mode, b.chi2_budget, b.active) == ("budget", 2.5, True)
    assert sc.parse_source("fp8_budget").mode == "budget"
    with pytest.raises(ValueError):
        sc.parse_source("stale")


def test_env_roundtrip():
    for spec in (sc.parse_source("stale4"), sc.parse_source("stale64", sleep1=True), sc.parse_source("budget"),
                 sc.parse_source("budget", chi2_budget=3.0)):
        assert sc.SyncSpec.from_env(spec.to_env()) == spec
    assert sc.SyncSpec.from_env({}) == sc.SyncSpec()
    assert sc.SyncSpec.from_env({sc.ENV_MODE: "budget"}).chi2_budget == 1.0
    with pytest.raises(ValueError):
        sc.SyncSpec.from_env({sc.ENV_MODE: "stale", sc.ENV_STALENESS: "-1"})
    with pytest.raises(ValueError):
        sc.SyncSpec.from_env({sc.ENV_MODE: "nope"})


def test_should_sync_schedule_matches_reference_refresh():
    # reference loop: for step t = 0, 1, ...: if t % (staleness+1) == 0: sampler = trainer after t updates.
    # verl: update_weights(global_steps=t) is called with t updates done (0 = before training); the weights sample t+1.
    for staleness in (0, 4, 16, 64):
        P = staleness + 1
        synced = [t for t in range(0, 400) if sc.should_sync(t, P, first_call=(t == 0))]
        assert synced == [t for t in range(0, 400) if t % P == 0]
    assert all(sc.should_sync(t, 1, False) for t in range(10))          # staleness 0 syncs every step
    assert sc.should_sync(37, 65, first_call=True)                       # first call: fresh engine or resume
    assert sc.should_sync(None, 65, first_call=False)                    # a non-training wake-up
    assert not sc.should_sync(37, 65, first_call=False)


def test_sampler_age_and_sync_step():
    assert [sc.sampler_age(t, 5) for t in range(7)] == [0, 1, 2, 3, 4, 0, 1]
    for ck in range(25, 400, 25):
        m = sc.stale_sync_step(ck, 65)
        assert m % 65 == 0 and m <= ck < m + 65


def test_should_sync_budget():
    assert sc.should_sync_budget(True, None, 1.0)            # first call (before training): no metric yet
    assert sc.should_sync_budget(False, 1.5, 1.0)
    assert not sc.should_sync_budget(False, 1.0, 1.0)        # <= budget skips
    assert not sc.should_sync_budget(False, 0.2, 1.0)
    with pytest.raises(RuntimeError):
        sc.should_sync_budget(False, None, 1.0)


def test_ess_threshold_is_the_chi2_budget():
    # ESS = 1 / E[r^2] = 1 / (1 + chi2): ESS < 1/2  <=>  chi2 > 1
    assert sc.chi2_budget_from_ess(0.5) == 1.0
    assert sc.chi2_budget_from_ess(1.0) == 0.0
    assert abs(sc.chi2_budget_from_ess(0.25) - 3.0) < 1e-12
    with pytest.raises(ValueError):
        sc.chi2_budget_from_ess(0.0)


def test_chi2_token_and_keys():
    np = pytest.importorskip("numpy")
    assert sc.chi2_token(np.zeros(10)) == 0.0
    x = np.log(np.array([0.5, 1.5]))
    assert abs(sc.chi2_token(x) - (np.mean(np.array([0.25, 2.25])) - 1.0)) < 1e-12
    assert sc.chi2_token(np.array([100.0])) == pytest.approx(np.exp(40.0) - 1.0)   # log ratio clamped at 20
    assert sc.budget_chi2_key({}) == sc.CHI2_KEY_TIS
    assert sc.budget_chi2_key({sc.ENV_SPLIT_OLD: "ref"}) == sc.CHI2_KEY_SPLIT
    assert sc.split_ref_reload({sc.ENV_SPLIT_OLD: "ref", sc.ENV_MODE: "budget"})
    assert not sc.split_ref_reload({sc.ENV_SPLIT_OLD: "ref", sc.ENV_MODE: "stale"})
