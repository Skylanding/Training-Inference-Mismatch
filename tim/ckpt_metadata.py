"""Checkpoint facts read from a checkpoint's own files: its vocabulary (`token_ids`), in which the stop tokens of the
model family are resolved, and the chat template it renders prompts through (`chat_template_text`).
scripts/convert_ckpt.py copies missing tokenizer and template files from the initial model.
"""
from __future__ import annotations
import json
import os
import shutil

class GenerationPolicyError(Exception):
    """A refusal: what was asked for is not there, and guessing would be the defect itself."""

TOKEN_SOURCES: tuple[str, ...] = ('tokenizer_config.json', 'added_tokens.json', 'tokenizer.json', 'vocab.json')
METADATA_FILES: tuple[str, ...] = ('chat_template.jinja', 'chat_template.json', 'generation_config.json',
                                   'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'vocab.json',
                                   'merges.txt', 'added_tokens.json', 'special_tokens_map.json')

def _harvest(out: dict[str, int], token, ident) -> None:
    """Record one (string, id) pair, keeping the first source that declared it."""
    if not isinstance(token, str) or not token:
        return
    if isinstance(ident, bool) or not isinstance(ident, int) or ident < 0:
        return
    out.setdefault(token, ident)

def token_ids(artifact: str) -> dict[str, int]:
    """This artifact's own vocabulary lookup: `{token string: id}`, read from its own files, the first file that
    declares a token winning."""
    out: dict[str, int] = {}
    for name in TOKEN_SOURCES:
        path = os.path.join(artifact, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'rb') as fh:
                loaded = json.loads(fh.read().decode('utf-8'))
        except OSError as e:
            raise GenerationPolicyError(f'{path} is there and could not be read ({e})') from None
        except (UnicodeDecodeError, ValueError):
            continue
        if not isinstance(loaded, dict):
            continue
        if name == 'tokenizer_config.json':
            decoder = loaded.get('added_tokens_decoder')
            if isinstance(decoder, dict):
                for ident, entry in decoder.items():
                    content = entry.get('content') if isinstance(entry, dict) else None
                    try:
                        _harvest(out, content, int(ident))
                    except (TypeError, ValueError):
                        continue
            continue
        if name == 'tokenizer.json':
            added = loaded.get('added_tokens')
            if isinstance(added, list):
                for entry in added:
                    if isinstance(entry, dict):
                        _harvest(out, entry.get('content'), entry.get('id'))
            model = loaded.get('model')
            vocab = model.get('vocab') if isinstance(model, dict) else None
            if isinstance(vocab, dict):
                for token, ident in vocab.items():
                    _harvest(out, token, ident)
            continue
        for token, ident in loaded.items():
            _harvest(out, token, ident)
    return out

def resolve_tokens(vocab: dict[str, int], tokens) -> tuple[int, ...]:
    """The ids of `tokens` that this vocabulary actually carries, in order and de-duplicated."""
    out: list[int] = []
    for token in tokens:
        ident = vocab.get(token)
        if isinstance(ident, int) and (not isinstance(ident, bool)) and (ident not in out):
            out.append(ident)
    return tuple(out)

def chat_template_text(artifact: str) -> str:
    """The chat template this artifact would render prompts through, as text, or ""."""
    path = os.path.join(artifact, 'chat_template.jinja')
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return fh.read()
        except OSError as e:
            raise GenerationPolicyError(f'{path} is there and could not be read ({e})') from None
    for name in ('chat_template.json', 'tokenizer_config.json'):
        path = os.path.join(artifact, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                loaded = json.load(fh)
        except OSError as e:
            raise GenerationPolicyError(f'{path} is there and could not be read ({e})') from None
        except ValueError:
            continue
        template = loaded.get('chat_template') if isinstance(loaded, dict) else None
        if isinstance(template, str) and template:
            return template
        if template:
            return json.dumps(template, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return ''

def copy_missing_metadata(artifact: str, source: str) -> tuple[str, ...]:
    """Copy missing non-weight metadata; never rewrite native model/generation config."""
    if not os.path.isdir(source):
        raise GenerationPolicyError(f"the metadata parent {source} is not a directory")
    copied: list[str] = []
    for name in METADATA_FILES:
        target = os.path.join(artifact, name)
        origin = os.path.join(source, name)
        if os.path.exists(target) or not os.path.isfile(origin):
            continue
        shutil.copy2(origin, target)
        copied.append(name)
    return tuple(copied)
