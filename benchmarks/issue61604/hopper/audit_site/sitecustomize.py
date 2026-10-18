"""Low-overhead import audit and opt-in manager counters for issue #61604."""

from __future__ import annotations

import atexit
import builtins
import importlib.util
import inspect
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path


if os.getenv("ISSUE61604_SOURCE_AUDIT"):
    spec = importlib.util.find_spec("vllm")
    print(
        "ISSUE61604_IMPORT_AUDIT "
        + json.dumps(
            {
                "pid": os.getpid(),
                "python": sys.executable,
                "vllm_origin": spec.origin if spec else None,
                "pythonpath": os.environ.get("PYTHONPATH", ""),
            },
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )


_diag_dir = os.getenv("ISSUE61604_DIAG_DIR")
_manager_stats: dict[int, dict] = {}
_original_import = builtins.__import__
_instrumented = False
_last_stats_write = 0.0


def _counter(self) -> dict | None:
    return _manager_stats.get(id(self))


def _install_instrumentation(module) -> None:
    global _instrumented
    if _instrumented or not _diag_dir:
        return
    manager_cls = getattr(module, "KVCacheManager", None)
    if manager_cls is None:
        return
    _instrumented = True
    original_init = manager_cls.__init__

    def init(self, *args, **kwargs):
        bound = inspect.signature(original_init).bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)
        coordinator = getattr(self, "coordinator", None)
        cache_spec = getattr(coordinator, "kv_cache_spec", None)
        record = {
            "pid": os.getpid(),
            "constructed_at_unix": time.time(),
            "coordinator_type": type(coordinator).__name__ if coordinator else None,
            "cache_spec_type": type(cache_spec).__name__ if cache_spec else None,
            "eagle_group_ids": list(getattr(coordinator, "eagle_group_ids", ())),
            "dcp_world_size": bound.arguments.get("dcp_world_size"),
            "pcp_world_size": bound.arguments.get("pcp_world_size"),
            "prefix_memo_enabled": getattr(self, "_memoize_prefix_hits", None),
            "stats": {
                "get_computed_blocks_calls": 0,
                "computed_tokens_returned": 0,
                "allocate_slots_calls": 0,
                "allocate_slots_failures": 0,
                "remember_failed_prefix_calls": 0,
                "memo_entries_created": 0,
                "memo_lookup_calls": 0,
                "memoized_tokens_returned": 0,
                "memo_invalidations": 0,
                "memo_entries_invalidated": 0,
            },
        }
        _manager_stats[id(self)] = record
        print("ISSUE61604_MANAGER_AUDIT " + json.dumps(record, separators=(",", ":")), file=sys.stderr, flush=True)
        _maybe_write_stats()

    manager_cls.__init__ = init

    if hasattr(manager_cls, "get_computed_blocks"):
        original = manager_cls.get_computed_blocks

        def get_computed_blocks(self, *args, _original=original, **kwargs):
            stats = _counter(self)
            if stats:
                stats["stats"]["get_computed_blocks_calls"] += 1
            result = _original(self, *args, **kwargs)
            if stats and isinstance(result, tuple) and len(result) > 1:
                stats["stats"]["computed_tokens_returned"] += int(result[1])
            _maybe_write_stats()
            return result

        manager_cls.get_computed_blocks = get_computed_blocks

    if hasattr(manager_cls, "allocate_slots"):
        original = manager_cls.allocate_slots

        def allocate_slots(self, *args, _original=original, **kwargs):
            stats = _counter(self)
            if stats:
                stats["stats"]["allocate_slots_calls"] += 1
            result = _original(self, *args, **kwargs)
            if stats and result is None:
                stats["stats"]["allocate_slots_failures"] += 1
            _maybe_write_stats()
            return result

        manager_cls.allocate_slots = allocate_slots

    if hasattr(manager_cls, "_remember_failed_prefix"):
        original = manager_cls._remember_failed_prefix

        def remember_failed_prefix(self, request, blocks, _original=original):
            stats = _counter(self)
            before = len(getattr(self, "_prefix_hits", ())) if stats else 0
            if stats:
                stats["stats"]["remember_failed_prefix_calls"] += 1
            result = _original(self, request, blocks)
            if stats:
                stats["stats"]["memo_entries_created"] += max(
                    0, len(getattr(self, "_prefix_hits", ())) - before
                )
            _maybe_write_stats()
            return result

        manager_cls._remember_failed_prefix = remember_failed_prefix

    if hasattr(manager_cls, "_find_memoized_prefix"):
        original = manager_cls._find_memoized_prefix

        def find_memoized_prefix(self, *args, _original=original, **kwargs):
            stats = _counter(self)
            if stats:
                stats["stats"]["memo_lookup_calls"] += 1
            result = _original(self, *args, **kwargs)
            if stats and isinstance(result, tuple) and len(result) > 1:
                stats["stats"]["memoized_tokens_returned"] += int(result[1])
            _maybe_write_stats()
            return result

        manager_cls._find_memoized_prefix = find_memoized_prefix

    if hasattr(manager_cls, "_invalidate_prefix_memos"):
        original = manager_cls._invalidate_prefix_memos

        def invalidate_prefix_memos(self, *args, _original=original, **kwargs):
            stats = _counter(self)
            before = len(getattr(self, "_prefix_hits", ())) if stats else 0
            if stats:
                stats["stats"]["memo_invalidations"] += 1
            result = _original(self, *args, **kwargs)
            if stats:
                stats["stats"]["memo_entries_invalidated"] += before
            _maybe_write_stats()
            return result

        manager_cls._invalidate_prefix_memos = invalidate_prefix_memos

    atexit.register(_write_stats)


def _write_stats() -> None:
    global _last_stats_write
    if not _diag_dir or not _manager_stats:
        return
    target_dir = Path(_diag_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"manager-{socket.gethostname()}-{os.getpid()}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(list(_manager_stats.values()), indent=2) + "\n")
    temporary.replace(destination)
    _last_stats_write = time.monotonic()


def _maybe_write_stats() -> None:
    if time.monotonic() - _last_stats_write >= 1.0:
        _write_stats()


def _write_stats_on_sigterm(signum, frame) -> None:
    _write_stats()
    raise SystemExit(128 + signum)


def _audit_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    if name == "vllm.v1.core.kv_cache_manager" or name.endswith(
        ".vllm.v1.core.kv_cache_manager"
    ):
        module = sys.modules.get("vllm.v1.core.kv_cache_manager")
        if module is not None:
            _install_instrumentation(module)
    return result


if _diag_dir:
    builtins.__import__ = _audit_import
    signal.signal(signal.SIGTERM, _write_stats_on_sigterm)
