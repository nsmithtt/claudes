#!/usr/bin/env python3
"""Launch parallel Claude instances for a skill, one per stdin line.

Each line from stdin is a separate skill invocation. Spaces on a line
become arguments passed to the skill.

Subcommands:
    skill   Run parallel Claude skill invocations (default behavior)
    merge   Iteratively rebase and merge each worker branch into the current branch
    clean   Remove all worker branches and worktrees

Example:
    echo -e "model-a\\nmodel-b\\nmodel-c" | python claudes.py skill --workers 3 --worktree my-branch port-huggingface-model
    python claudes.py merge my-branch
    python claudes.py clean my-branch
"""

import argparse
import datetime
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CLAUDES_DIR = Path.cwd() / ".claudes"
LOG_DIR = CLAUDES_DIR / "logs"


class TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def run_invocation(
    skill: str, skill_args: str, worktree_branch: str | None, log_file: Path
):
    """Run claude on a single skill invocation."""
    prompt = f"/{skill} {skill_args}".strip()
    cmd = [
        "claude",
        "-p",
        prompt,
        "--allowedTools",
        "Edit,Write,Read,Glob,Grep,Bash,Skill,Agent",
    ]
    if worktree_branch is not None:
        cmd.extend(["--worktree", worktree_branch])

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    reason = ""
    if result.returncode != 0:
        try:
            last_line = log_file.read_text().strip().rsplit("\n", 1)[-1]
            reason = last_line[:200]
        except OSError:
            reason = "could not read log"

    return skill_args, result.returncode, reason


def run_worker(
    worker_index: int,
    invocations: list,
    skill: str,
    worktree_base: str | None,
    result_queue: multiprocessing.Queue,
):
    """Run all assigned invocations sequentially in this worker."""
    worktree_branch = (
        f"{worktree_base}-{worker_index}" if worktree_base is not None else None
    )
    for skill_args in invocations:
        safe_name = skill_args.replace("/", "_").replace(" ", "_")[:80]
        log_file = LOG_DIR / f"{safe_name}.log"
        _, rc, reason = run_invocation(skill, skill_args, worktree_branch, log_file)
        result_queue.put((worker_index, skill_args, rc, reason))


def get_worker_branches(worktree_base: str) -> list[str]:
    """Find all worker branches matching the base pattern."""
    result = subprocess.run(
        ["git", "branch", "--list", f"{worktree_base}-*"],
        capture_output=True,
        text=True,
    )
    branches = []
    for line in result.stdout.strip().splitlines():
        branch = line.strip().lstrip("* ")
        if branch:
            branches.append(branch)
    # Sort numerically by worker index suffix
    branches.sort(
        key=lambda b: int(b.rsplit("-", 1)[-1]) if b.rsplit("-", 1)[-1].isdigit() else 0
    )
    return branches


def cmd_skill(args):
    """Run parallel Claude skill invocations."""
    worktree_base = None
    if args.worktree is not None:
        if args.worktree:
            worktree_base = args.worktree
        else:
            worktree_base = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()

    lines = [line.strip() for line in sys.stdin if line.strip()]

    if args.start > 0:
        lines = lines[args.start :]
    if args.limit > 0:
        lines = lines[: args.limit]

    if not lines:
        print("No input lines to process.", file=sys.stderr)
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = LOG_DIR / f"claudes_{timestamp}.log"
    log_fh = open(main_log, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_fh)
    sys.stderr = TeeStream(sys.__stderr__, log_fh)
    print(f"Logging to {main_log}")

    print(f"Processing {len(lines)} invocations with up to {args.workers} workers")
    print(f"Skill: /{args.skill_name}")

    num_workers = min(args.workers, len(lines))
    chunks = [[] for _ in range(num_workers)]
    for i, line in enumerate(lines):
        chunks[i % num_workers].append(line)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                run_worker, idx, chunk, args.skill_name, worktree_base, result_queue
            )
            for idx, chunk in enumerate(chunks)
        ]

        print("Workers started...")

        completed = 0
        while completed < len(lines):
            worker_index, skill_args, rc, reason = result_queue.get()
            completed += 1
            status = "OK" if rc == 0 else f"FAILED (rc={rc}): {reason}"
            print(
                f"[{completed}/{len(lines)}] worker-{worker_index} {skill_args}: {status}"
            )

        for future in futures:
            future.result()

    print("All done.")
    log_fh.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def resolve_worktree_base(args):
    """Return worktree_base, defaulting to current branch if not specified."""
    if args.worktree_base:
        return args.worktree_base
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def cmd_merge(args):
    """Iteratively rebase and merge each worker branch into the current branch."""
    worktree_base = resolve_worktree_base(args)
    branches = get_worker_branches(worktree_base)
    if not branches:
        print(f"No worker branches found matching '{worktree_base}-*'", file=sys.stderr)
        sys.exit(1)

    current_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    print(f"Merging {len(branches)} worker branches into {current_branch}")

    for branch in branches:
        subprocess.run(["git", "checkout", branch], capture_output=True)
        print(f"\n--- Rebasing {branch} onto {current_branch} (strategy: theirs) ---")
        result = subprocess.run(
            [
                "git",
                "rebase",
                "--keep-empty",
                "--allow-empty-message",
                "-X",
                "theirs",
                current_branch,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Rebase failed for {branch}:\n{result.stderr}", file=sys.stderr)
            subprocess.run(["git", "rebase", "--abort"])
            sys.exit(1)

        print(f"--- Merging {branch} into {current_branch} ---")
        subprocess.run(["git", "checkout", current_branch], capture_output=True)
        result = subprocess.run(
            ["git", "merge", branch],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Merge failed for {branch}:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        print(f"Merged {branch}")

    print(f"\nAll {len(branches)} branches merged into {current_branch}.")


def cmd_clean(args):
    """Remove all worker branches and worktrees."""
    worktree_base = resolve_worktree_base(args)
    branches = get_worker_branches(worktree_base)
    if not branches:
        print(f"No worker branches found matching '{worktree_base}-*'")
        return

    # Get list of worktrees
    worktree_result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
    )
    worktree_paths = {}
    current_path = None
    for line in worktree_result.stdout.splitlines():
        if line.startswith("worktree "):
            current_path = line[len("worktree ") :]
        elif line.startswith("branch refs/heads/"):
            branch = line[len("branch refs/heads/") :]
            if current_path:
                worktree_paths[branch] = current_path
            current_path = None

    print(f"Cleaning {len(branches)} worker branches for '{worktree_base}'")

    for branch in branches:
        # Remove worktree if it exists
        if branch in worktree_paths:
            print(f"Removing worktree for {branch} at {worktree_paths[branch]}")
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_paths[branch]],
                capture_output=True,
                text=True,
            )

        # Delete the branch
        print(f"Deleting branch {branch}")
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True,
            text=True,
        )

    # Prune worktree metadata
    subprocess.run(["git", "worktree", "prune"], capture_output=True)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel Claude instances for a skill"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # skill subcommand
    skill_parser = subparsers.add_parser(
        "skill", help="Run parallel Claude skill invocations"
    )
    skill_parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N input lines (default: 0)",
    )
    skill_parser.add_argument(
        "--limit", type=int, default=0, help="Max invocations to process (0 = all)"
    )
    skill_parser.add_argument(
        "--workers", type=int, default=25, help="Max concurrent Claude instances"
    )
    skill_parser.add_argument(
        "--worktree",
        nargs="?",
        const="",
        default=None,
        help="Run in a git worktree. Optional branch name (default: current branch). Worker id is appended.",
    )
    skill_parser.add_argument(
        "skill_name",
        type=str,
        help="Claude skill to parallelize (e.g. port-huggingface-model)",
    )

    # merge subcommand
    merge_parser = subparsers.add_parser(
        "merge",
        help="Iteratively rebase and merge each worker branch into the current branch",
    )
    merge_parser.add_argument(
        "worktree_base",
        nargs="?",
        default="",
        help="Worktree base name (default: current branch). Branches named <base>-0, <base>-1, ... will be merged.",
    )

    # clean subcommand
    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove all worker branches and worktrees",
    )
    clean_parser.add_argument(
        "worktree_base",
        nargs="?",
        default="",
        help="Worktree base name (default: current branch). Branches named <base>-0, <base>-1, ... will be removed.",
    )

    args = parser.parse_args()

    if args.command == "skill":
        cmd_skill(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "clean":
        cmd_clean(args)


if __name__ == "__main__":
    main()
