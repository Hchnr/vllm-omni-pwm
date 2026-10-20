# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run independent optimization profiles serially and preserve failures."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="JSON list of {id, options} profiles")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--wait-pid", type=int, help="Wait for an earlier measurement before any model loading")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    while args.wait_pid:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(2)
    events = []
    for profile in json.loads(args.suite.read_text()):
        name = profile["id"]
        if not name or Path(name).name != name:
            raise ValueError("Profile ids must be plain filenames")
        options = args.output_dir / f"{name}.options.json"
        options.write_text(json.dumps(profile["options"], indent=2) + "\n")
        command = [
            sys.executable,
            "tools/pwm_wam/bench_opt.py",
            "--options",
            str(options),
            "--output",
            str(args.output_dir / f"{name}.json"),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--input",
            *args.input,
        ]
        if profile.get("profile"):
            command.append("--profile")
        event = dict(id=name, command=command, start=time.time())
        print(f"Starting {name}", flush=True)
        with (args.output_dir / f"{name}.log").open("w") as log:
            event["exit_code"] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        event["end"] = time.time()
        events.append(event)
        (args.output_dir / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        print(f"Finished {name}, exit={event['exit_code']}", flush=True)


if __name__ == "__main__":
    main()
