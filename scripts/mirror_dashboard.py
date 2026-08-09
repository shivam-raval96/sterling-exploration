from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from sterling_exploration.dashboard import render_dashboard


def pull(modal: Path, run_id: str, local_dir: Path) -> None:
    remote_root = f"/experiments/advbench_jailbreak/runs/{run_id}"
    for name in ("progress.json", "dashboard_history.jsonl"):
        subprocess.run(
            [
                str(modal),
                "volume",
                "get",
                "--force",
                "sterling-outputs",
                f"{remote_root}/{name}",
                str(local_dir / name),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    history = [
        json.loads(line)
        for line in (local_dir / "dashboard_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    (local_dir / "dashboard.html").write_text(render_dashboard(history))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--modal", type=Path, default=Path(".venv/bin/modal"))
    args = parser.parse_args()
    local_dir = Path("experiments/advbench_jailbreak/runs") / args.run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            pull(args.modal, args.run_id, local_dir)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            print(f"dashboard mirror retry: {error}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
