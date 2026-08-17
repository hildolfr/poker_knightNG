"""Package entrypoint for running the bounded service runtime."""

from __future__ import annotations

import argparse
import asyncio
import signal

from .runtime import ServiceRuntime, add_runtime_arguments


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="poker-knight-ng-service",
        description="Run the bounded Poker Knight NG local service.",
    )
    add_runtime_arguments(parser)
    return parser.parse_args(args=argv)


async def _run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    runtime = ServiceRuntime(
        max_sessions=args.max_sessions,
        graceful_drain_seconds=args.graceful_drain_seconds,
    )
    stop = asyncio.Event()

    try:
        for signo in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(signo, stop.set)
    except TypeError:
        # Platforms without POSIX signal delivery keep no signal handler support here;
        # process shutdown remains operator-driven.
        pass

    await runtime.serve(stop)


def main(argv: list[str] | None = None) -> int:
    try:
        asyncio.run(_run(argv))
        return 0
    except KeyboardInterrupt:
        return 130


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover - module invocation shim
    raise SystemExit(main(argv))


if __name__ == "__main__":
    _main()
