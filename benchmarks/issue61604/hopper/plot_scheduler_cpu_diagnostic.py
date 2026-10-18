#!/usr/bin/env python3
"""Generate the issue #61604 single-pair instrumented CPU timing chart."""

from pathlib import Path


OUT = Path(__file__).with_name("scheduler_cpu_diagnostic.svg")

# Corrected outer-timer measurements from the same seed-6161805 trace.
MEASUREMENTS = [
    ("Scheduler.schedule()", 14.927, 11.121, "−25.5%"),
    ("get_computed_blocks()", 11.663, 6.426, "−44.9%"),
]


def main() -> None:
    width, height = 1000, 520
    left, right = 270, 920
    scale = (right - left) / 16
    rows = [170, 310]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.label{font-size:19px;font-weight:600}.value{font-size:16px}.note{font-size:14px;fill:#555}.title{font-size:23px;font-weight:700}</style>',
        '<text class="title" x="500" y="42" text-anchor="middle">Issue #61604: instrumented CPU timing</text>',
    ]

    # Shared horizontal scale, with a subtle axis and grid.
    for seconds in range(0, 17, 2):
        x = left + seconds * scale
        parts.append(f'<line x1="{x:.1f}" y1="95" x2="{x:.1f}" y2="390" stroke="#e5e7eb"/>')
        parts.append(f'<text class="note" x="{x:.1f}" y="414" text-anchor="middle">{seconds}s</text>')

    colors = {"Base": "#64748b", "Patch": "#2563eb"}
    for (name, base, patch, change), y in zip(MEASUREMENTS, rows, strict=True):
        parts.append(f'<text class="label" x="{left - 18}" y="{y - 5}" text-anchor="end">{name}</text>')
        for label, value, offset in (("Base", base, 12), ("Patch", patch, 54)):
            bar_y = y + offset
            bar_width = value * scale
            parts.append(f'<text class="note" x="{left - 10}" y="{bar_y + 18}" text-anchor="end">{label}</text>')
            parts.append(f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" height="26" rx="3" fill="{colors[label]}"/>')
            parts.append(f'<text class="value" x="{left + bar_width + 9:.1f}" y="{bar_y + 19}">{value:.3f} s</text>')
        parts.append(f'<text x="{right}" y="{y + 48}" text-anchor="end" font-size="16" font-weight="700" fill="#15803d">CPU time {change}</text>')

    parts.extend([
        '<text class="note" x="500" y="462" text-anchor="middle">One matched diagnostic pair · same seed-6161805 trace · 200 requests per arm</text>',
        '<text class="note" x="500" y="486" text-anchor="middle">Outer CPU timers were instrumented and nested; this is not an end-to-end throughput result.</text>',
        '</svg>',
    ])
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
