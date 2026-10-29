"""Shared machinery of the verl hooks in this package.

verl imports every module named in VERL_USE_EXTERNAL_MODULES (verl/__init__.py) in every process, the driver and the
Ray workers alike. Each hook module of this package registers its patches with `when_imported`, so the target verl
module is patched right after it is first executed and importing verl does not pull in vLLM / FSDP / Ray modules
earlier than it would anyway. Every patch prints one line per process saying what was applied.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
import threading

TAG = "tim"
STATE = {}   # what each patch reported, shared by all hook modules of a process


def _say(msg: str) -> None:
    print(f"[{TAG} pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.pending = {}
        self._lock = threading.Lock()

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.pending:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            with self._lock:
                callbacks = self.pending.pop(fullname, [])
            original_exec = spec.loader.exec_module

            def exec_module(module, _orig=original_exec, _callbacks=callbacks):
                _orig(module)
                for cb in _callbacks:
                    cb(module)

            spec.loader.exec_module = exec_module
            return spec
        return None


_FINDER = _PostImportFinder()


def when_imported(modname: str, callback) -> None:
    """Run callback(module) right after `modname` is executed (immediately if it already is)."""
    if modname in sys.modules:
        callback(sys.modules[modname])
        return
    _FINDER.pending.setdefault(modname, []).append(callback)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def _guard(name: str, fn):
    """A patch whose failure leaves verl as installed and says so."""
    def run(module):
        try:
            STATE[name] = fn(module)
        except Exception as error:
            STATE[name] = f"not applied: {type(error).__name__}: {error}"
        _say(f"{name}: {STATE[name]}")
    return run


def _strict(name: str, fn):
    """A patch whose failure raises: without it the run would train a different method."""
    def run(module):
        STATE[name] = fn(module)
        if str(STATE[name]).startswith("not applied"):
            raise RuntimeError(f"tim {name}: {STATE[name]}")
        _say(f"{name}: {STATE[name]}")
    return run
