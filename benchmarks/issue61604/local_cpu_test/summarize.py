"""Summarize paired local CPU reproduction runs."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

pairs = []
for path in sorted(args.input.glob('base-*.json')):
    round_id = path.stem.removeprefix('base-')
    base = json.loads(path.read_text())
    patch = json.loads((args.input / f'pr-{round_id}.json').read_text())
    if base['trace_sha256'] != patch['trace_sha256']:
        raise SystemExit(f'scheduling trace differs in paired round {round_id}')
    if not base['admitted_after_release'] or not patch['admitted_after_release']:
        raise SystemExit(f'waiter failed to admit after capacity release: {round_id}')
    pairs.append({
        'round': int(round_id),
        'base_schedule_ms': base['schedule_ns'] / 1e6,
        'pr_schedule_ms': patch['schedule_ns'] / 1e6,
        'base_lookup_ms': base['lookup_ns'] / 1e6,
        'pr_lookup_ms': patch['lookup_ns'] / 1e6,
        'base_probes': base['probes'],
        'pr_probes': patch['probes'],
        'trace_sha256': base['trace_sha256'],
    })
if not pairs:
    raise SystemExit('no paired runs found')
summary = {
    'pairs': pairs,
    'base_schedule_median_ms': statistics.median(p['base_schedule_ms'] for p in pairs),
    'pr_schedule_median_ms': statistics.median(p['pr_schedule_ms'] for p in pairs),
    'base_lookup_median_ms': statistics.median(p['base_lookup_ms'] for p in pairs),
    'pr_lookup_median_ms': statistics.median(p['pr_lookup_ms'] for p in pairs),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
