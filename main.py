"""
Farmers EA Topline Multi-Agent Analytics System
Entry point for data generation, pipeline execution, and API server.
"""

import argparse
import logging
import os
import sys

from colorama import Fore, Style, init as colorama_init


def setup_logging():
    """Configure INFO-level logging with colored, timestamped output."""
    colorama_init(autoreset=True)
    fmt = (
        f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} "
        f"{Fore.GREEN}%(levelname)-8s{Style.RESET_ALL} "
        f"%(name)s - %(message)s"
    )
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def data_files_exist() -> bool:
    """Return True if at least one CSV exists in the data/ directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.isdir(data_dir):
        return False
    return any(f.endswith((".parquet", ".csv")) for f in os.listdir(data_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Farmers EA Topline Multi-Agent Analytics System",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "pipeline", "generate-data"],
        default="server",
        help="Run mode (default: server)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip data generation if data files already exist",
    )
    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args()

    # ---- generate-data mode ----
    if args.mode == "generate-data":
        logger.info("Generating synthetic data ...")
        from generate_data import main as generate_main

        generate_main()
        logger.info("Data generation complete.")
        return

    # ---- pipeline mode ----
    if args.mode == "pipeline":
        logger.info("Running full analytics pipeline ...")
        from graph.workflow import run_pipeline

        final_state = run_pipeline()

        report = final_state.get("report", "")
        errors = final_state.get("errors", [])

        if report:
            print("\n" + "=" * 60)
            print("PIPELINE REPORT")
            print("=" * 60)
            print(report)
            print("=" * 60)

        error_count = len(errors)
        if error_count:
            logger.warning("Pipeline finished with %d error(s):", error_count)
            for err in errors:
                logger.warning("  - %s", err)
        else:
            logger.info("Pipeline finished successfully with 0 errors.")
        return

    # ---- server mode (default) ----
    if not args.no_data and not data_files_exist():
        logger.info("No data files found in data/ -- generating synthetic data first ...")
        from generate_data import main as generate_main

        generate_main()
        logger.info("Data generation complete.")

    import uvicorn

    banner = (
        f"\n{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}\n"
        f"  Farmers EA Topline Multi-Agent Analytics System\n"
        f"  Server starting at {Fore.CYAN}http://{args.host}:{args.port}{Style.RESET_ALL}\n"
        f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}\n"
    )
    print(banner)

    uvicorn.run("api.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
