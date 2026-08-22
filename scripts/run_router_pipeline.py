from __future__ import annotations

import argparse
from pathlib import Path

from router.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline router evaluation pipeline.")
    parser.add_argument("--export-dir", type=Path, default=Path("export"), help="Directory of chunked JSONL exports")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write CSVs and PNGs")
    args = parser.parse_args()

    artifacts = run_pipeline(args.export_dir, args.output_dir)
    print(f"trajectories={len(artifacts.trajectory_frame)}")
    print(f"summary_rows={len(artifacts.summary_frame)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
