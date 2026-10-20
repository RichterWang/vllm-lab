"""Isolated source overlay and split count/timing diagnostics for issue #61604.

Loaded only by diagnostic runs. Count mode has no per-block clocks; timing
mode measures outer calls only and never wraps get_cached_block.
"""

from __future__ import annotations

import atexit
import builtins
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


_source_root = os.environ.get("ISSUE61604_SOURCE_ROOT")
_binary_site = os.environ.get("ISSUE61604_BINARY_SITE")
_diag_dir = os.environ.get("ISSUE61604_DIAG_DIR")
_control_file = os.environ.get("ISSUE61604_DIAG_CONTROL_FILE")
_diag_mode = os.environ.get("ISSUE61604_DIAG_MODE", "count")
if _diag_mode not in ("count", "timing"):
    raise ValueError(f"invalid ISSUE61604_DIAG_MODE={_diag_mode!r}")
_stats: dict[int, dict] = {}
_managers_by_record_id: dict[int, object] = {}
_coordinator_stats: dict[int, dict] = {}
_scheduler_stats: dict[int, dict] = {}
_pool_stats: dict[int, dict] = {}
_pending_memo_blocks: dict[str, tuple[dict, list[tuple]]] = {}
_acquired_memo_blocks: dict[str, tuple[dict, list]] = {}
_thread = threading.local()
_patched_classes: set[tuple[str, str]] = set()
_import_logged: set[int] = set()
_process_cpu_start_ns = time.process_time_ns()
_recording = os.environ.get("ISSUE61604_DIAG_START_MANUAL") != "1"
_original_import = builtins.__import__


class _VllmOverlayFinder(importlib.abc.MetaPathFinder):
    """Load Python from the selected worktree and extensions from one wheel."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("vllm") or not _source_root or not _binary_site:
            return None
        source_package = Path(_source_root) / "vllm"
        binary_package = Path(_binary_site) / "vllm"
        if fullname == "vllm":
            spec = importlib.machinery.PathFinder.find_spec(fullname, [_source_root])
        else:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.submodule_search_locations is None:
            return spec
        relative_parts = fullname.split(".")[1:]
        source_dir = source_package.joinpath(*relative_parts)
        binary_dir = binary_package.joinpath(*relative_parts)
        locations = list(spec.submodule_search_locations)
        for candidate in (source_dir, binary_dir):
            if candidate.is_dir() and str(candidate) not in locations:
                locations.append(str(candidate))
        spec.submodule_search_locations[:] = locations
        return spec


if _source_root and _binary_site:
    sys.meta_path.insert(0, _VllmOverlayFinder())


def _now() -> tuple[int, int]:
    return time.thread_time_ns(), time.perf_counter_ns()


def _add_timing(record: dict, prefix: str, cpu_start: int, wall_start: int) -> None:
    record[f"{prefix}_cpu_ns"] = record.get(f"{prefix}_cpu_ns", 0) + (
        time.thread_time_ns() - cpu_start
    )
    record[f"{prefix}_wall_ns"] = record.get(f"{prefix}_wall_ns", 0) + (
        time.perf_counter_ns() - wall_start
    )


def _count_mode() -> bool:
    return _diag_mode == "count"


def _timing_mode() -> bool:
    return _diag_mode == "timing"


def _patch_method(cls, name, make_wrapper, module_name):
    key = (f"{cls.__module__}.{cls.__qualname__}", name)
    if key in _patched_classes or not hasattr(cls, name):
        return
    original = getattr(cls, name)
    setattr(cls, name, make_wrapper(original))
    _patched_classes.add(key)
    print(
        "ISSUE61604_DIAG_PATCH "
        + json.dumps({"class": key[0], "method": name, "module": module_name}),
        file=sys.stderr,
        flush=True,
    )


def _install_manager(module) -> None:
    cls = getattr(module, "KVCacheManager", None)
    if cls is None:
        return

    def wrap_init(original):
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            bound = inspect.signature(original).bind_partial(self, *args, **kwargs)
            coordinator = getattr(self, "coordinator", None)
            record = {
                "pid": os.getpid(),
                "python": sys.executable,
                "coordinator_type": type(coordinator).__name__ if coordinator else None,
                "cache_spec_type": type(getattr(coordinator, "kv_cache_spec", None)).__name__
                if coordinator
                else None,
                "eagle_group_ids": list(getattr(coordinator, "eagle_group_ids", ())),
                "dcp_world_size": bound.arguments.get("dcp_world_size"),
                "pcp_world_size": bound.arguments.get("pcp_world_size"),
                "prefix_memo_enabled": getattr(self, "_memoize_prefix_hits", None),
                "stats": {
                    "computed_lookup_calls": 0,
                    "computed_lookup_cpu_ns": 0,
                    "computed_lookup_wall_ns": 0,
                    "ordinary_lookup_calls": 0,
                    "ordinary_lookup_cpu_ns": 0,
                    "ordinary_lookup_wall_ns": 0,
                    "ordinary_block_map_probes": 0,
                    "ordinary_block_map_hits": 0,
                    "memo_lookup_calls": 0,
                    "memo_lookup_cpu_ns": 0,
                    "memo_lookup_wall_ns": 0,
                    "memo_fallback_cpu_ns": 0,
                    "memo_fallback_map_probes": 0,
                    "memo_fallback_map_hits": 0,
                    "memo_validated_blocks": 0,
                    "memo_build_calls": 0,
                    "memo_build_cpu_ns": 0,
                    "memo_build_wall_ns": 0,
                    "memo_entries_created": 0,
                    "memo_invalidation_checks": 0,
                    "memo_map_pointer_resyncs": 0,
                    "successful_prefix_cache_reset_calls": 0,
                    "memo_entries_removed_on_reset": 0,
                    "memo_entry_truncation_events": 0,
                    "memo_block_records_removed_by_validation": 0,
                    "memo_first_change_index_counts": {},
                    "memo_first_change_reason_counts": {},
                    "memo_entries_removed_on_allocation": 0,
                    "memo_entries_removed_on_free": 0,
                    "allocate_slots_calls": 0,
                    "allocate_slots_failures": 0,
                    "failed_attempts_by_request": {},
                    "repeated_failure_requests": 0,
                    "memo_entry_count_peak": 0,
                    "memo_block_records_peak": 0,
                    "memo_entries_live": 0,
                    "memo_block_records_live": 0,
                    "memo_reuse_calls": 0,
                    "memo_reused_blocks": 0,
                    "cache_mapping_deletions": 0,
                    "memo_integrity_checks": 0,
                    "memo_integrity_failures": 0,
                    "memo_reference_acquire_checks": 0,
                    "memo_reference_acquire_failures": 0,
                    "memo_reference_release_checks": 0,
                    "memo_reference_release_successes": 0,
                    "memo_build_success_existing": 0,
                    "memo_pending_cleared_without_acquire": 0,
                },
            }
            _stats[id(self)] = record
            _managers_by_record_id[id(record)] = self
            if coordinator is not None:
                _coordinator_stats[id(coordinator)] = record
                block_pool = getattr(coordinator, "block_pool", None)
                if block_pool is not None:
                    _pool_stats[id(block_pool)] = record
            print(
                "ISSUE61604_MANAGER_AUDIT " + json.dumps(record, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
            _write_stats()

        return init

    _patch_method(cls, "__init__", wrap_init, module.__name__)

    def wrap_computed(original):
        def get_computed_blocks(self, request, *args, **kwargs):
            record = _stats.get(id(self))
            if record is None or not _recording:
                return original(self, request, *args, **kwargs)
            counters = record["stats"]
            counters["computed_lookup_calls"] += 1
            if not _timing_mode():
                return original(self, request, *args, **kwargs)
            cpu_start, wall_start = _now()
            try:
                return original(self, request, *args, **kwargs)
            finally:
                _add_timing(counters, "computed_lookup", cpu_start, wall_start)

        return get_computed_blocks

    _patch_method(cls, "get_computed_blocks", wrap_computed, module.__name__)

    def wrap_allocate(original):
        def allocate_slots(self, request, *args, **kwargs):
            record = _stats.get(id(self))
            if record is None or not _recording:
                return original(self, request, *args, **kwargs)
            counters = record["stats"]
            counters["allocate_slots_calls"] += 1
            if _timing_mode():
                result = original(self, request, *args, **kwargs)
                counters["allocate_slots_failures"] += int(result is None)
                if result is None:
                    request_id = str(getattr(request, "request_id", "unknown"))
                    attempts = counters["failed_attempts_by_request"].get(request_id, 0) + 1
                    counters["failed_attempts_by_request"][request_id] = attempts
                return result
            memo_table = getattr(self, "_prefix_hits", None)
            had_memo = memo_table is not None and request in memo_table
            result = original(self, request, *args, **kwargs)
            if result is None:
                counters["allocate_slots_failures"] += 1
                request_id = str(getattr(request, "request_id", "unknown"))
                attempts = counters["failed_attempts_by_request"].get(request_id, 0) + 1
                counters["failed_attempts_by_request"][request_id] = attempts
            else:
                memo_table_after = getattr(self, "_prefix_hits", None)
                if had_memo and memo_table_after is not None and request not in memo_table_after:
                    counters["memo_entries_removed_on_allocation"] += 1
            return result

        return allocate_slots

    if _count_mode():
        _patch_method(cls, "allocate_slots", wrap_allocate, module.__name__)

    def wrap_remember(original):
        def remember(self, request, blocks):
            record = _stats.get(id(self))
            if record is None or not _recording:
                return original(self, request, blocks)
            counters = record["stats"]
            counters["memo_build_calls"] += 1
            if _timing_mode():
                cpu_start, wall_start = _now()
                try:
                    return original(self, request, blocks)
                finally:
                    _add_timing(counters, "memo_build", cpu_start, wall_start)
            before = request in self._prefix_hits
            result = original(self, request, blocks)
            after = request in self._prefix_hits
            counters["memo_entries_created"] += int(not before and after)
            counters["memo_build_success_existing"] += int(before and after)
            if _count_mode():
                _update_memo_peaks(self, counters)
            return result

        return remember

    _patch_method(cls, "_remember_failed_prefix", wrap_remember, module.__name__)

    def wrap_invalidate(original):
        def invalidate(self, *args, **kwargs):
            record = _stats.get(id(self))
            if record is None or not _recording:
                return original(self, *args, **kwargs)
            counters = record["stats"]
            counters["memo_invalidation_checks"] += 1
            if _timing_mode():
                return original(self, *args, **kwargs)
            old_map = self._prefix_hit_map
            before = len(self._prefix_hits)
            result = original(self, *args, **kwargs)
            if self._prefix_hit_map is not old_map:
                counters["memo_map_pointer_resyncs"] += 1
                counters["memo_entries_removed_on_reset"] += max(
                    0, before - len(self._prefix_hits)
                )
            if _count_mode():
                _update_memo_peaks(self, counters)
            return result

        return invalidate

    if _count_mode():
        _patch_method(cls, "_invalidate_prefix_memos", wrap_invalidate, module.__name__)

    def wrap_find_memo(original):
        def find_memoized_prefix(self, request, max_length, memo):
            record = _stats.get(id(self))
            if record is None or not _recording:
                return original(self, request, max_length, memo)
            counters = record["stats"]
            counters["memo_lookup_calls"] += 1
            existed = bool(memo)
            if existed:
                counters["memo_reuse_calls"] += 1
            if _timing_mode():
                cpu_start, wall_start = _now()
                try:
                    result = original(self, request, max_length, memo)
                finally:
                    _add_timing(counters, "memo_lookup", cpu_start, wall_start)
                return result
            before = tuple(memo)
            old_context = getattr(_thread, "memo_record", None)
            _thread.memo_record = record
            _thread.last_memo_fallback_hits = 0
            try:
                result = original(self, request, max_length, memo)
            finally:
                _thread.memo_record = old_context
            returned_blocks = len(result[0][0]) if result and result[0] else 0
            fallback_hits = getattr(_thread, "last_memo_fallback_hits", 0)
            counters["memo_validated_blocks"] += max(0, returned_blocks - fallback_hits)
            validated_memo_blocks = max(0, returned_blocks - fallback_hits)
            if validated_memo_blocks:
                selected = result[0][0][:validated_memo_blocks]
                memo_records = before[:validated_memo_blocks]
                integrity_failures = 0
                for index, ((expected_hash, expected_block, version), actual_block) in enumerate(
                    zip(memo_records, selected, strict=True)
                ):
                    expected_key = module.make_block_hash_with_group_id(expected_hash, 0)
                    valid = (
                        request.block_hashes[index] == expected_hash
                        and actual_block is expected_block
                        and actual_block.block_hash == expected_key
                        and actual_block._hash_lifecycle_version == version
                        and self._prefix_hit_map.contain(expected_key, actual_block.block_id)
                    )
                    counters["memo_integrity_checks"] += 1
                    integrity_failures += int(not valid)
                counters["memo_integrity_failures"] += integrity_failures
                if integrity_failures:
                    event = {
                        "request_id": request.request_id,
                        "validated_memo_blocks": validated_memo_blocks,
                        "returned_blocks": returned_blocks,
                        "fallback_hits": fallback_hits,
                    }
                    _append_event("integrity-failures.jsonl", event)
                    raise AssertionError(f"memo block identity/key/lifecycle check failed: {event}")
                _pending_memo_blocks[request.request_id] = (
                    record,
                    [(h, b, v) for h, b, v in memo_records],
                )
            after = tuple(memo)
            before_ids = {(item[0], item[1].block_id, item[2]) for item in before}
            after_ids = {(item[0], item[1].block_id, item[2]) for item in after}
            removed = len(before_ids - after_ids)
            if removed:
                counters["memo_entry_truncation_events"] += 1
                counters["memo_block_records_removed_by_validation"] += removed
                # The production validator truncates from its first changed
                # record. Record that index and the exact failed predicate so
                # a call counter is not mistaken for an invalidation count.
                first_changed = 0
                shared = min(len(before), len(after))
                while first_changed < shared and (
                    before[first_changed][0] == after[first_changed][0]
                    and before[first_changed][1].block_id
                    == after[first_changed][1].block_id
                    and before[first_changed][2] == after[first_changed][2]
                ):
                    first_changed += 1
                old_hash, old_block, old_version = before[first_changed]
                limit = min(
                    max_length // self.coordinator.block_size,
                    len(request.block_hashes),
                )
                if first_changed >= limit:
                    reason = "request_limit"
                elif request.block_hashes[first_changed] != old_hash:
                    reason = "request_hash_mismatch"
                elif old_block.block_hash != module.make_block_hash_with_group_id(
                    old_hash, 0
                ):
                    reason = "block_hash_mismatch"
                elif old_block._hash_lifecycle_version != old_version:
                    reason = "lifecycle_version_mismatch"
                else:
                    reason = "record_replaced_or_removed_without_predicate_mismatch"
                index_counts = counters["memo_first_change_index_counts"]
                reason_counts = counters["memo_first_change_reason_counts"]
                index_counts[str(first_changed)] = index_counts.get(str(first_changed), 0) + 1
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            counters["memo_reused_blocks"] += validated_memo_blocks if existed else 0
            _update_memo_peaks(self, counters)
            return result

        return find_memoized_prefix

    _patch_method(cls, "_find_memoized_prefix", wrap_find_memo, module.__name__)

    def wrap_free(original):
        def free(self, request, *args, **kwargs):
            record = _stats.get(id(self))
            if record is not None and _recording and _timing_mode():
                return original(self, request, *args, **kwargs)
            request_id = str(getattr(request, "request_id", ""))
            if record is not None and _recording and request_id in _pending_memo_blocks:
                record["stats"]["memo_pending_cleared_without_acquire"] += 1
                _pending_memo_blocks.pop(request_id, None)
            memo_table = getattr(self, "_prefix_hits", None)
            existed = (
                record is not None
                and _recording
                and memo_table is not None
                and request in memo_table
            )
            result = original(self, request, *args, **kwargs)
            memo_table_after = getattr(self, "_prefix_hits", None)
            if existed and memo_table_after is not None and request not in memo_table_after:
                record["stats"]["memo_entries_removed_on_free"] += 1
            return result

        return free

    if _count_mode():
        _patch_method(cls, "free", wrap_free, module.__name__)

    def wrap_reset(original):
        def reset(self, *args, **kwargs):
            record = _stats.get(id(self))
            if record is not None and _recording and _timing_mode():
                return original(self, *args, **kwargs)
            memo_table = getattr(self, "_prefix_hits", None)
            before = (
                len(memo_table)
                if record is not None and _recording and memo_table is not None
                else 0
            )
            result = original(self, *args, **kwargs)
            if result and record is not None and _recording:
                record["stats"]["successful_prefix_cache_reset_calls"] += 1
                memo_table_after = getattr(self, "_prefix_hits", None)
                record["stats"]["memo_entries_removed_on_reset"] += max(
                    0,
                    before - len(memo_table_after)
                    if memo_table_after is not None
                    else before,
                )
            return result

        return reset

    if _count_mode():
        _patch_method(cls, "reset_prefix_cache", wrap_reset, module.__name__)


def _update_memo_peaks(manager, counters):
    memos = getattr(manager, "_prefix_hits", {})
    counters["memo_entry_count_peak"] = max(
        counters["memo_entry_count_peak"], len(memos)
    )
    counters["memo_block_records_peak"] = max(
        counters["memo_block_records_peak"], sum(len(value) for value in memos.values())
    )


def _install_coordinator(module):
    for name, cls in vars(module).items():
        if (
            not isinstance(cls, type)
            or cls.__module__ != module.__name__
            or not hasattr(cls, "find_longest_cache_hit")
        ):
            continue
        # Only patch the class that defines this method; avoid wrapping abstract
        # base methods and inherited aliases more than once.
        if "find_longest_cache_hit" not in cls.__dict__:
            continue

        def make_wrapper(original):
            def find_longest_cache_hit(self, *args, **kwargs):
                record = _coordinator_stats.get(id(self))
                if record is None or not _recording:
                    return original(self, *args, **kwargs)
                counters = record["stats"]
                counters["ordinary_lookup_calls"] += 1
                old = getattr(_thread, "ordinary_record", None)
                _thread.ordinary_record = record
                if not _timing_mode():
                    try:
                        return original(self, *args, **kwargs)
                    finally:
                        _thread.ordinary_record = old
                cpu_start, wall_start = _now()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    _add_timing(counters, "ordinary_lookup", cpu_start, wall_start)
                    _thread.ordinary_record = old

            return find_longest_cache_hit

        _patch_method(cls, "find_longest_cache_hit", make_wrapper, module.__name__)

    for name, cls in vars(module).items():
        if (
            not isinstance(cls, type)
            or cls.__module__ != module.__name__
            or "allocate_new_computed_blocks" not in cls.__dict__
        ):
            continue
        original = cls.allocate_new_computed_blocks

        def make_allocate_wrapper(original):
            def allocate_new_computed_blocks(self, *args, **kwargs):
                bound = inspect.signature(original).bind_partial(self, *args, **kwargs)
                request_id = str(bound.arguments.get("request_id", ""))
                pending = _pending_memo_blocks.get(request_id)
                if pending is None or not _recording or not _count_mode():
                    return original(self, *args, **kwargs)
                record, memo_records = pending
                groups = bound.arguments.get("new_computed_blocks") or ()
                incoming = {block.block_id: block for group in groups for block in group}
                selected = [(h, b, v) for h, b, v in memo_records if incoming.get(b.block_id) is b]
                if len(selected) != len(memo_records):
                    event = {
                        "request_id": request_id,
                        "memo_blocks": len(memo_records),
                        "blocks_passed_to_allocation": len(selected),
                    }
                    record["stats"]["memo_reference_acquire_failures"] += 1
                    _append_event("reference-failures.jsonl", event)
                    raise AssertionError(f"memo blocks were not all passed to allocation: {event}")
                before_refs = {b.block_id: b.ref_cnt for _, b, _ in selected}
                result = original(self, *args, **kwargs)
                for _, block, _ in selected:
                    record["stats"]["memo_reference_acquire_checks"] += 1
                    acquired = block.ref_cnt == before_refs[block.block_id] + 1
                    record["stats"]["memo_reference_acquire_failures"] += int(not acquired)
                    if not acquired:
                        event = {
                            "request_id": request_id,
                            "block_id": block.block_id,
                            "before_ref_cnt": before_refs[block.block_id],
                            "after_ref_cnt": block.ref_cnt,
                        }
                        _append_event("reference-failures.jsonl", event)
                        raise AssertionError(f"memo block refcount was not acquired exactly once: {event}")
                _acquired_memo_blocks[request_id] = (record, selected)
                _pending_memo_blocks.pop(request_id, None)
                return result

            return allocate_new_computed_blocks

        _patch_method(cls, "allocate_new_computed_blocks", make_allocate_wrapper, module.__name__)

    for name, cls in vars(module).items():
        if (
            not isinstance(cls, type)
            or cls.__module__ != module.__name__
            or "free" not in cls.__dict__
        ):
            continue
        original = cls.free

        def make_free_wrapper(original):
            def free(self, request_id, *args, **kwargs):
                tracked = _acquired_memo_blocks.get(str(request_id))
                before = {
                    block.block_id: block.ref_cnt for _, block, _ in tracked[1]
                } if tracked and _recording and _count_mode() else {}
                result = original(self, request_id, *args, **kwargs)
                if tracked and before:
                    record, selected = tracked
                    released = 0
                    for _, block, _ in selected:
                        released += int(block.ref_cnt == before[block.block_id] - 1)
                    record["stats"]["memo_reference_release_checks"] += len(selected)
                    record["stats"]["memo_reference_release_successes"] += released
                    if released != len(selected):
                        event = {
                            "request_id": str(request_id),
                            "selected_blocks": len(selected),
                            "release_successes": released,
                            "ref_counts_after_free": {
                                str(block.block_id): block.ref_cnt for _, block, _ in selected
                            },
                        }
                        _append_event("reference-failures.jsonl", event)
                        raise AssertionError(f"memo block references were not released exactly once: {event}")
                    _acquired_memo_blocks.pop(str(request_id), None)
                return result

            return free

        _patch_method(cls, "free", make_free_wrapper, module.__name__)


def _install_block_pool(module):
    cls = getattr(module, "BlockPool", None)
    if cls is None:
        return

    def wrap_lookup(original):
        def get_cached_block(self, *args, **kwargs):
            record = getattr(_thread, "memo_record", None) or getattr(
                _thread, "ordinary_record", None
            )
            if record is None or not _recording:
                return original(self, *args, **kwargs)
            if not _count_mode():
                return original(self, *args, **kwargs)
            result = original(self, *args, **kwargs)
            if getattr(_thread, "memo_record", None) is record:
                counters = record["stats"]
                counters["memo_fallback_map_probes"] += 1
                _thread.last_memo_fallback_hits = getattr(
                    _thread, "last_memo_fallback_hits", 0
                ) + int(result is not None)
                counters["memo_fallback_map_hits"] += int(result is not None)
            elif getattr(_thread, "ordinary_record", None) is record:
                counters = record["stats"]
                counters["ordinary_block_map_probes"] += 1
                counters["ordinary_block_map_hits"] += int(result is not None)
            return result

        return get_cached_block

    if _count_mode():
        _patch_method(cls, "get_cached_block", wrap_lookup, module.__name__)

    def wrap_remove(original):
        def remove_cached_hashes(self, block, *args, **kwargs):
            record = _pool_stats.get(id(self))
            result = original(self, block, *args, **kwargs)
            if record is not None and _recording and _count_mode():
                record["stats"]["cache_mapping_deletions"] += len(result or ())
            return result

        return remove_cached_hashes

    if _count_mode():
        _patch_method(cls, "_remove_cached_block_hashes", wrap_remove, module.__name__)


def _install_scheduler(module):
    cls = getattr(module, "Scheduler", None)
    if cls is None:
        return
    schedulers: dict[int, dict] = {}

    def wrap_init(original):
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            record = {
                "pid": os.getpid(),
                "schedule_calls": 0,
                "schedule_cpu_ns": 0,
                "schedule_wall_ns": 0,
                "schedule_wall_samples_ns": [],
                "schedule_cpu_samples_ns": [],
                "queue_samples": [],
                "waiting_steps_by_request": {},
                "preempted_request_events": 0,
                "preempted_request_ids": [],
                "scheduled_token_steps": 0,
            }
            schedulers[id(self)] = record
            _scheduler_stats[id(self)] = record
            _write_record(f"scheduler-{os.getpid()}-{id(self)}.json", record)

        return init

    _patch_method(cls, "__init__", wrap_init, module.__name__)

    def wrap_schedule(original):
        def schedule(self, *args, **kwargs):
            record = schedulers.get(id(self))
            if record is None or not _recording:
                return original(self, *args, **kwargs)
            record["schedule_calls"] += 1
            if not _timing_mode():
                return original(self, *args, **kwargs)
            cpu_start, wall_start = _now()
            try:
                return original(self, *args, **kwargs)
            finally:
                record["schedule_cpu_ns"] += time.thread_time_ns() - cpu_start
                record["schedule_wall_ns"] += time.perf_counter_ns() - wall_start

        return schedule

    _patch_method(cls, "schedule", wrap_schedule, module.__name__)

    # This is an event counter, not a timer. Keep it in count mode only so a
    # churn run can prove that the scheduler naturally preempted requests.
    if _count_mode():
        def wrap_preempt(original):
            def preempt(self, request, *args, **kwargs):
                record = schedulers.get(id(self))
                if record is not None and _recording:
                    record["preempted_request_events"] += 1
                    record["preempted_request_ids"].append(
                        str(getattr(request, "request_id", "unknown"))
                    )
                return original(self, request, *args, **kwargs)

            return preempt

        _patch_method(cls, "_preempt_request", wrap_preempt, module.__name__)


def _install(module_name):
    module = sys.modules.get(module_name)
    if module is None:
        return
    if module_name.endswith("kv_cache_manager"):
        _install_manager(module)
    elif module_name.endswith("kv_cache_coordinator"):
        _install_coordinator(module)
    elif module_name.endswith("block_pool"):
        _install_block_pool(module)
    elif module_name.endswith("sched.scheduler"):
        _install_scheduler(module)


def _audit_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    if name.startswith("vllm."):
        for candidate in (
            "vllm.v1.core.kv_cache_manager",
            "vllm.v1.core.kv_cache_coordinator",
            "vllm.v1.core.block_pool",
            "vllm.v1.core.sched.scheduler",
        ):
            _install(candidate)
        if os.getpid() not in _import_logged:
            _import_logged.add(os.getpid())
            print(
                "ISSUE61604_IMPORT_AUDIT "
                + json.dumps(
                    {
                        "pid": os.getpid(),
                        "python": sys.executable,
                        "vllm_origin": getattr(sys.modules.get("vllm"), "__file__", None),
                        "vllm_path": list(getattr(sys.modules.get("vllm"), "__path__", ())),
                        "source_root": _source_root,
                        "binary_site": _binary_site,
                    },
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
    return result


if _diag_dir:
    builtins.__import__ = _audit_import
    signal.signal(signal.SIGTERM, lambda signum, frame: (_write_stats(), sys.exit(128 + signum)))

    def _watch_diagnostic_control() -> None:
        """Enable counters after warmup and checkpoint them while EngineCore lives.

        vLLM may replace Python signal handlers during startup and its normal
        API shutdown can force-kill EngineCore before atexit runs. A shared
        marker file makes activation independent of signal-handler ordering;
        periodic atomic snapshots preserve counters before forced shutdown.
        """
        global _recording, _process_cpu_start_ns
        activated = False
        checkpoint_interval = 30.0 if _timing_mode() else 5.0
        next_checkpoint = 0.0
        while True:
            if not activated and _control_file and Path(_control_file).exists():
                _process_cpu_start_ns = time.process_time_ns()
                _recording = True
                activated = True
                next_checkpoint = time.monotonic()
                print(
                    "ISSUE61604_DIAG_RECORDING_STARTED "
                    + json.dumps({"pid": os.getpid(), "control_file": _control_file}),
                    file=sys.stderr,
                    flush=True,
                )
            if activated:
                now = time.monotonic()
                if now >= next_checkpoint:
                    try:
                        _write_stats()
                    except (OSError, RuntimeError, TypeError, ValueError) as error:
                        print(
                            f"ISSUE61604_DIAG_CHECKPOINT_ERROR pid={os.getpid()} {error!r}",
                            file=sys.stderr,
                            flush=True,
                        )
                    next_checkpoint = now + checkpoint_interval
                if _control_file and Path(_control_file + ".stop").exists():
                    _write_stats()
                    return
            time.sleep(0.5)

    threading.Thread(
        target=_watch_diagnostic_control,
        name="issue61604-diagnostic-checkpoint",
        daemon=True,
    ).start()


def _write_record(name: str, value: dict) -> None:
    if not _diag_dir:
        return
    target = Path(_diag_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _append_event(name: str, value: dict) -> None:
    if not _diag_dir:
        return
    target = Path(_diag_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / name).open("a") as stream:
        stream.write(json.dumps(value, separators=(",", ":")) + "\n")


def _write_stats() -> None:
    if not _diag_dir:
        return
    for record in list(_stats.values()):
        failures = list(record["stats"]["failed_attempts_by_request"].values())
        record["stats"]["failed_attempt_request_count"] = len(failures)
        record["stats"]["repeated_failure_requests"] = sum(count > 1 for count in failures)
        manager = _managers_by_record_id.get(id(record))
        memos = getattr(manager, "_prefix_hits", None)
        if memos is not None and _count_mode():
            record["stats"]["memo_entries_live"] = len(memos)
            record["stats"]["memo_block_records_live"] = sum(
                len(value) for value in memos.values()
            )
    _write_record(
        f"manager-{os.getpid()}.json",
        {
            "managers": list(_stats.values()),
            "process_cpu_elapsed_ns": time.process_time_ns() - _process_cpu_start_ns,
            "scheduler": "separate per scheduler json",
            "pending_memo_request_ids": sorted(_pending_memo_blocks),
            "acquired_memo_request_ids": sorted(_acquired_memo_blocks),
        },
    )
    for scheduler_id, record in _scheduler_stats.items():
        _write_record(f"scheduler-{os.getpid()}-{scheduler_id}.json", record)


atexit.register(_write_stats)
