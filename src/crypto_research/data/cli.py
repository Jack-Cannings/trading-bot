"""Explicit public historical-data commands; no network on import."""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from crypto_research.contracts import Market, Period
from crypto_research.data.models import Manifest, Request
from crypto_research.data.provider import BinanceProvider, ProviderError
from crypto_research.data.service import NoDataError, acquire
from crypto_research.data.storage import IntegrityError, Store


def date(value: str) -> datetime:
    try:
        if len(value) == 10:
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timestamp requires an explicit timezone")
        return timestamp.astimezone(UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid UTC date/timestamp: {value}: {exc}") from exc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Public historical OHLCV, UTC [start,end) candle opens"
    )
    root.add_argument(
        "--root", type=Path, default=Path("data"), help="local immutable dataset directory"
    )
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="explicitly contact Binance public endpoints")
    download.add_argument("--exchange", choices=["binance"], default="binance")
    download.add_argument(
        "--market", nargs="+", required=True, choices=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    download.add_argument("--timeframe", nargs="+", choices=["1d", "4h"], required=True)
    start = download.add_mutually_exclusive_group(required=True)
    start.add_argument("--start", type=date, help="inclusive; date-only means UTC midnight")
    start.add_argument(
        "--longest", action="store_true", help="probe from Unix epoch; report pre-listing gaps"
    )
    download.add_argument(
        "--end",
        type=date,
        required=True,
        help="exclusive aligned end; date-only means UTC midnight",
    )
    download.add_argument("--batch-size", type=int, default=1000)
    download.add_argument("--retries", type=int, default=3)
    update = commands.add_parser(
        "update", help="resume/update an immutable version with recent overlap"
    )
    update.add_argument("version")
    update.add_argument("--end", type=date, help="new exclusive end; defaults to original request")
    update.add_argument(
        "--overlap", type=int, default=2, help="number of recent candles checked for revisions"
    )
    for name in ("validate", "coverage", "quality"):
        command = commands.add_parser(name, help=f"show stored {name}; never contacts exchange")
        command.add_argument("version")
    commands.add_parser("list", help="list committed immutable versions")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = Store(args.root)
    try:
        if args.command == "list":
            print(json.dumps(store.versions()))
            return 0
        if args.command in {"validate", "quality", "coverage"}:
            manifest, _, report = store.read(args.version)
            print(
                manifest.canonical_json() if args.command == "coverage" else report.canonical_json()
            )
            return 0 if args.command != "validate" or report.status == "clean" else 2
        provider = BinanceProvider()

        def progress(manifest: Manifest) -> None:
            # Manifest summaries are durable restart points, emitted before the next page.
            print(f"checkpoint: {manifest.version}", file=sys.stderr)

        if args.command == "update":
            parent = store.manifest(args.version)
            request = Request.model_validate(
                parent.request.model_dump()
                | {
                    "period": Period(
                        start=parent.request.period.start, end=args.end or parent.request.period.end
                    )
                }
            )
            result = acquire(
                provider,
                store,
                request,
                parent_version=args.version,
                overlap=args.overlap,
                checkpoint=progress,
            )
            print(result.canonical_json())
            return 0
        for pair in args.market:
            for timeframe in args.timeframe:
                request = Request(
                    market=Market(exchange=args.exchange, pair=pair, timeframe=timeframe),
                    period=Period(
                        start=datetime(1970, 1, 1, tzinfo=UTC) if args.longest else args.start,
                        end=args.end,
                    ),
                    batch_size=args.batch_size,
                    retries=args.retries,
                )
                result = acquire(provider, store, request, checkpoint=progress)
                print(result.canonical_json())
        return 0
    except NoDataError as exc:
        print(exc.report.canonical_json(), file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, ProviderError, ValidationError, IntegrityError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: resume with update <last checkpoint version>", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
