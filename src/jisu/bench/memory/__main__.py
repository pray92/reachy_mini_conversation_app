"""Memory retrieval benchmark CLI.

python -m jisu.bench.memory                         # json, gbrain, gbrain-native; k=5; repeat=3; saves the report
python -m jisu.bench.memory --subjects json gbrain --k 3 --repeat 5
python -m jisu.bench.memory --dataset my.toml --no-save --detail
python -m jisu.bench.memory --subjects gbrain-native gbrain-embed-native   # embeddings (needs GEMINI_API_KEY)
python -m jisu.bench.memory --subjects gbrain gbrain-http                  # CLI vs resident HTTP server
python -m jisu.bench.memory --list                  # saved reports
python -m jisu.bench.memory --show <file>           # print a saved report again, with its run log
"""

from __future__ import annotations
import io
import sys
import argparse
from typing import TextIO, cast
from pathlib import Path
from collections.abc import Sequence

from jisu.bench.memory.table import render_report, render_report_list
from jisu.bench.memory.report import (
    ReportError,
    load_report,
    save_report,
    list_reports,
    load_report_log,
    save_report_log,
)
from jisu.bench.memory.runner import BenchConfig, run_memory_bench
from jisu.dashboard.constants import DEFAULT_GBRAIN_EMBEDDING_DIMS, DEFAULT_GBRAIN_EMBEDDING_MODEL
from jisu.bench.memory.dataset import DatasetError
from jisu.bench.memory.progress import ProgressPrinter, remove_log_handler, install_log_handler
from jisu.bench.memory.constants import (
    DEFAULT_K,
    SUBJECT_NAMES,
    DEFAULT_REPEAT,
    DEFAULT_DATASET,
    DEFAULT_SUBJECTS,
    default_results_dir,
)


EXIT_OK = 0
EXIT_NOTHING_RAN = 2


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jisu.bench.memory",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset TOML")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECT_NAMES, default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="results considered per query")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="search passes per query (latency)")
    parser.add_argument(
        "--embedding-model", default=DEFAULT_GBRAIN_EMBEDDING_MODEL, help="model for gbrain-embed* subjects"
    )
    parser.add_argument(
        "--embedding-dims", type=int, default=DEFAULT_GBRAIN_EMBEDDING_DIMS, help="vector size for gbrain-embed*"
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="where gbrain-embed* looks for GEMINI_API_KEY"
    )
    parser.add_argument("--out", type=Path, default=None, help=f"report folder (default {default_results_dir()})")
    parser.add_argument("--no-save", action="store_true", help="do not write the report file")
    parser.add_argument("--detail", action="store_true", help="list every failed query")
    parser.add_argument(
        "--quiet", action="store_true", help="no progress log on screen (still saved); --show: tables only"
    )
    parser.add_argument("--brief", action="store_true", help="progress log shows only failed queries")
    parser.add_argument("--list", action="store_true", help="list saved reports and exit")
    parser.add_argument("--show", type=Path, metavar="FILE", help="print a saved report and exit")
    return parser.parse_args(argv)


class _Tee(io.TextIOBase):
    """Write to several text streams at once (screen + saved log)."""

    def __init__(self, *streams: TextIO) -> None:
        super().__init__()
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.list:
        print(render_report_list(list_reports(args.out)))
        return EXIT_OK
    if args.show:
        try:
            report = load_report(args.show)
        except ReportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NOTHING_RAN
        log = None if args.quiet else load_report_log(args.show)
        if log:
            print(log.rstrip("\n"))
            print()
        print(render_report(report, detail=args.detail))
        return EXIT_OK

    try:
        config = BenchConfig(
            dataset=args.dataset,
            subjects=tuple(args.subjects),
            k=args.k,
            repeat=args.repeat,
            embedding_model=args.embedding_model,
            embedding_dims=args.embedding_dims,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOTHING_RAN

    # The progress log always goes to a buffer (saved next to the report); the screen gets it unless --quiet.
    captured = io.StringIO()
    stream: TextIO = captured if args.quiet else cast(TextIO, _Tee(sys.stdout, captured))
    handler = install_log_handler(stream)
    printer = ProgressPrinter(k=config.k, repeat=config.repeat, brief=args.brief, stream=stream)
    try:
        report = run_memory_bench(config, on_event=printer, dotenv_path=args.env_file)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOTHING_RAN
    finally:
        remove_log_handler(handler)

    print()
    print(render_report(report, detail=args.detail))
    if not report.subjects:
        return EXIT_NOTHING_RAN
    if not args.no_save:
        path = save_report(report, args.out)
        log_path = save_report_log(path, captured.getvalue())
        print(f"\nsaved {path}\n      {log_path} (run log; shown by --show)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
