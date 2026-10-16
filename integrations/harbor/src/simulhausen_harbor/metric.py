# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Harbor ``uv-script`` metric: simulhausen's pass rate over the rewards of a job.

Harbor's default ``mean`` counts a trial without a reward as 0. simulhausen's verifier writes no ``reward.json``
for WARN (the harness broke, or criteria came back inconclusive with none failed), and simulhausen keeps WARN out
of the pass rate: ``pass_rate = PASS / (PASS + FAIL)``. This script reproduces that in a job config::

    metrics:
      - type: uv-script
        kwargs:
          script_path: path/to/metric.py

Harbor runs ``uv run metric.py -i rewards.jsonl -o metric.json``: one JSON object (a trial's rewards) or ``null``
per line. Output keys:

- ``pass_rate`` (0-1; 0 when nothing was judged), ``n_pass``, ``n_fail``, ``n_warn``. A row whose ``reward`` is 1
  is a PASS, any other number a FAIL; ``null``, a row without a numeric ``reward``, or anything that is not an
  object counts as WARN.
- the mean of every other numeric key over the rows that carry it, under the key's own name (``judge_score``,
  ``criteria_pass_rate``, ``criterion_1``...), or ``mean_<key>`` if the name clashes with one of the keys above.
  Missing keys are skipped, not counted as 0: ``--rewards detailed`` omits inconclusive criteria on purpose.

Standard library only (PEP 723 header above), so it also runs as ``python metric.py`` or
``python -m simulhausen_harbor.metric -i rewards.jsonl -o metric.json``; without ``-o`` it prints the result.
"""

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REWARD_KEY = "reward"
SUMMARY_KEYS = ("pass_rate", "n_pass", "n_fail", "n_warn")


def _is_number(value: Any) -> bool:
    # bool is an int subclass; a finite check keeps NaN/inf out of the output JSON.
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def compute(rewards: Iterable[Any]) -> dict[str, float | int]:
    """Aggregate per-trial reward dicts (``None`` for a trial without a reward) into the metric."""
    n_pass = n_fail = n_warn = 0
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rewards:
        if not isinstance(row, dict):
            n_warn += 1
            continue
        reward = row.get(REWARD_KEY)
        if not _is_number(reward):
            n_warn += 1
        elif reward == 1:
            n_pass += 1
        else:
            n_fail += 1
        for key, value in row.items():
            if key != REWARD_KEY and _is_number(value):
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
    judged = n_pass + n_fail
    result: dict[str, float | int] = {
        "pass_rate": n_pass / judged if judged else 0.0,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_warn": n_warn,
    }
    for key in sorted(sums):
        result[f"mean_{key}" if key in SUMMARY_KEYS else key] = sums[key] / counts[key]
    return result


def read_rewards(path: Path) -> list[Any]:
    """Parse Harbor's ``rewards.jsonl``; blank lines are ignored."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simulhausen pass rate for a Harbor job (uv-script metric).")
    parser.add_argument("-i", "--input-path", type=Path, required=True, help="rewards.jsonl written by Harbor.")
    parser.add_argument("-o", "--output-path", type=Path, help="Where to write metric.json (default: stdout).")
    args = parser.parse_args(argv)
    payload = json.dumps(compute(read_rewards(args.input_path)), allow_nan=False)
    if args.output_path is None:
        sys.stdout.write(payload + "\n")
    else:
        args.output_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
