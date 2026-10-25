"""``metareason`` command line: benchmarks and step verification."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from metareason import benchmarks as B
from metareason.symbolic.verifier import StepVerifier

log = logging.getLogger("metareason")


def _bench(args: argparse.Namespace) -> int:
    cfg = B.BenchConfig.from_toml(args.config) if args.config else B.BenchConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.quick:
        cfg = cfg.quick()
    which = ["sinusoid", "fewshot", "ewc"] if args.which == "all" else [args.which]
    for w in which:
        log.info("running %s", w)
        rows = getattr(B, f"bench_{w}")(cfg)
        print(f"\n### {w} (seed={cfg.seed})\n")
        print(json.dumps(rows, indent=2) if args.json else B.format_table(rows))
    return 0


def _verify(args: argparse.Namespace) -> int:
    with open(args.file, encoding="utf-8") as fh:
        spec = json.load(fh)
    res = StepVerifier(spec.get("clauses", [])).verify(spec["steps"])
    for r in res.steps:
        print(f"step {r.index + 1}: {r.status}")
    if res.diagnosis:
        print("DIAGNOSIS:", res.diagnosis.explain())
        return 1
    print("all checked steps consistent")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the CLI."""
    p = argparse.ArgumentParser(prog="metareason")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="run reproducible benchmarks")
    b.add_argument("which", choices=["sinusoid", "fewshot", "ewc", "all"])
    b.add_argument("--config")
    b.add_argument("--seed", type=int)
    b.add_argument("--quick", action="store_true", help="tiny CI-sized run")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=_bench)
    v = sub.add_parser("verify", help="verify a JSON reasoning trace {clauses, steps}")
    v.add_argument("file")
    v.set_defaults(func=_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
