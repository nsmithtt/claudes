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
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_DELAY = 30.0
RATE_LIMIT_MAX_DELAY = 300.0

CLAUDES_DIR = Path.cwd() / ".claudes"
LOG_DIR = CLAUDES_DIR / "logs"


def list_worktrees() -> dict[str, Path]:
    """Return {branch_name: worktree_path} for every worktree in this repo."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    worktrees: dict[str, Path] = {}
    current_path: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            current_path = line[len("worktree ") :]
        elif line.startswith("branch refs/heads/"):
            branch = line[len("branch refs/heads/") :]
            if current_path:
                worktrees[branch] = Path(current_path)
            current_path = None
    return worktrees


def ensure_worktree(branch: str) -> Path:
    """Create or reuse a worktree for `branch` at the conventional path.

    Raises if the branch is checked out at a different path, or if the target
    path is occupied by a different branch.
    """
    path = Path.cwd() / "worktrees" / branch.replace("/", "+")
    existing = list_worktrees()

    if branch in existing:
        if existing[branch].resolve() == path.resolve():
            return path
        raise RuntimeError(
            f"Branch '{branch}' is already checked out at {existing[branch]}, "
            f"expected {path}"
        )
    for b, p in existing.items():
        if p.resolve() == path.resolve():
            raise RuntimeError(
                f"Worktree path {path} is occupied by branch '{b}', not '{branch}'"
            )

    branch_exists = (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        ).returncode
        == 0
    )

    # Drop stale worktree registrations so a half-cleaned prior run doesn't
    # collide with `git worktree add` below.
    subprocess.run(["git", "worktree", "prune"], check=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    add_cmd = ["git", "worktree", "add"]
    if branch_exists:
        add_cmd += [str(path), branch]
    else:
        add_cmd += ["-b", branch, str(path)]
    subprocess.run(add_cmd, check=True)

    # `git worktree add` doesn't materialize submodule worktree gitdirs, so
    # the `.git` pointer files inside each submodule end up dangling. Running
    # `submodule update` inside the new worktree creates those gitdirs.
    subprocess.run(
        ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    return path


def build_bwrap_command(
    workspace_path: Path,
    extra_ro_binds: list[str],
    extra_rw_binds: list[str],
) -> list[str]:
    """Bubblewrap prefix that restricts Claude to `workspace_path` plus the
    minimum host paths needed for Claude itself and git to function.

    `extra_ro_binds` / `extra_rw_binds` are host paths the caller wants
    exposed beyond the defaults. They are applied last so they can override
    the base binds on overlapping paths.
    """
    home = Path.home()
    # The workspace's `.git` is a pointer file to the real gitdir. Resolve it
    # so we bind the actual directory: `<repo>/.git` for a main-repo worktree,
    # `<superproject>/.git/modules/<sub>` for a submodule worktree.
    git_dir = Path(
        subprocess.check_output(
            ["git", "-C", str(workspace_path), "rev-parse", "--git-common-dir"],
            text=True,
        ).strip()
    ).resolve()
    main_git = Path.cwd()

    cmd = [
        "bwrap",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/etc", "/etc",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/sbin", "/sbin",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # Bind all of home which is kind of unsafe, but claude barfs on ~/.claude.json if this is read-only.
        "--bind", str(home), str(home),
    ]

    for path in extra_ro_binds:
        cmd += ["--ro-bind", path, path]
    for path in extra_rw_binds:
        cmd += ["--bind", path, path]

    cmd += [
        "--ro-bind", str(main_git), str(main_git),
        "--bind", str(git_dir), str(git_dir),
        "--bind", str(workspace_path), str(workspace_path),
        "--chdir", str(workspace_path),
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-uts",
        "--die-with-parent",
    ]

    # Pass the ssh-agent socket through if one is available on the host.
    ssh_auth_sock = os.environ.get("SSH_AUTH_SOCK")
    if ssh_auth_sock and Path(ssh_auth_sock).exists():
        cmd += [
            "--ro-bind-try", ssh_auth_sock, ssh_auth_sock,
            "--setenv", "SSH_AUTH_SOCK", ssh_auth_sock,
        ]
    cmd.append("--")
    return cmd


class TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, stream, file_handle):
        self._stream = stream
        self._file = file_handle

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


def build_claude_command(
    skill: str,
    skill_args: str,
    model: str | None,
    headless=True,
):
    """Build the claude CLI command for a single skill invocation."""
    prompt = f"/{skill} {skill_args}".strip()
    cmd = [
        "claude",
        prompt,
        "--allowedTools",
        "Edit,Write,Read,Glob,Grep,Bash,Skill,Agent",
    ]
    if headless:
        cmd.append("-p")
    if model is not None:
        cmd.extend(["--model", model])
    return cmd


def build_sandboxed_command(
    skill: str,
    skill_args: str,
    worktree_path: Path | None,
    model: str | None,
    extra_ro_binds: list[str],
    extra_rw_binds: list[str],
    sandbox: bool = False,
    headless: bool = True,
) -> list[str]:
    """Claude argv, optionally wrapped in bwrap when `sandbox` is True."""
    cmd = build_claude_command(skill, skill_args, model, headless)
    if not sandbox:
        return cmd

    workspace = worktree_path if worktree_path is not None else Path.cwd()

    # The `claude` launcher on PATH is often a symlink (sometimes chained
    # through a $HOME symlink like ~/.local -> /proj_sw/...) into a versioned
    # install tree. Bind the fully-resolved real binary and invoke it directly
    # so the sandbox doesn't have to reproduce the host's symlink topology.
    extra_ro = list(extra_ro_binds)
    claude_on_path = shutil.which("claude")
    if claude_on_path:
        claude_real = str(Path(claude_on_path).resolve())
        cmd[0] = claude_real
        extra_ro.append(claude_real)

    return build_bwrap_command(workspace, extra_ro, extra_rw_binds) + cmd


def _kill_process_group(pgid: int, log) -> None:
    """SIGTERM the whole group, then SIGKILL any stragglers after 5s.

    Without this, `claude`'s children (Bash tools, MCP servers, sub-agents)
    survive as orphans when we time out or get interrupted.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
        log.write("\n--- process group did not exit on SIGTERM, sent SIGKILL ---\n")
        log.flush()
    except ProcessLookupError:
        pass


def _run_in_process_group(
    cmd: list[str],
    log,
    cwd: str | None,
    timeout: float | None,
    active_pgids=None,
) -> tuple[int, bool]:
    """Run `cmd` in its own process group, killing the whole group on timeout.

    `log` is where stdout+stderr go, and where kill-diagnostics are written.
    Pass None to inherit the parent's stdout/stderr (useful for debug mode).
    Returns (returncode, timed_out). On timeout, returncode is 124.

    When `active_pgids` is a shared list, the child's pgid is registered while
    it runs so the parent's SIGINT handler can fan out kills even if this
    worker is stuck elsewhere when the interrupt arrives.
    """
    popen_kwargs = {"cwd": cwd, "start_new_session": True}
    if log is not None:
        popen_kwargs["stdout"] = log
        popen_kwargs["stderr"] = subprocess.STDOUT
    proc = subprocess.Popen(cmd, **popen_kwargs)
    if active_pgids is not None:
        try:
            active_pgids.append(proc.pid)
        except Exception:
            pass
    diag = log if log is not None else sys.stderr
    try:
        rc = proc.wait(timeout=timeout)
        return rc, False
    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid, diag)
        proc.wait()
        return 124, True
    except BaseException:
        # KeyboardInterrupt / worker teardown: don't leak the tree.
        _kill_process_group(proc.pid, diag)
        proc.wait()
        raise
    finally:
        if active_pgids is not None:
            try:
                active_pgids.remove(proc.pid)
            except (ValueError, Exception):
                pass


def _log_tail(log_file: Path, n_chars: int = 4000) -> str:
    try:
        return log_file.read_text()[-n_chars:]
    except OSError:
        return ""


def _is_rate_limited(log_tail: str) -> bool:
    # Matches `API Error: Request rejected (429) · ... rate limit ...` from the
    # claude CLI. We require both markers so a stray "429" in model output
    # doesn't trigger spurious retries.
    return "(429)" in log_tail and "rate limit" in log_tail.lower()


def run_generator(cmd: str, inflight: list[str]) -> list[str]:
    """Invoke the generator shell command, piping the current in-flight
    skill_args on stdin. Return non-empty output lines."""
    stdin_payload = "\n".join(inflight)
    if stdin_payload:
        stdin_payload += "\n"
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write("[claudes] generator command timed out\n")
        return []
    if proc.returncode != 0:
        sys.stderr.write(
            f"[claudes] generator exit {proc.returncode}: "
            f"{proc.stderr.strip()[:500]}\n"
        )
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def run_invocation(
    skill: str,
    skill_args: str,
    worktree_path: Path | None,
    log_file: Path,
    model: str | None,
    extra_ro_binds: list[str],
    extra_rw_binds: list[str],
    sandbox: bool,
    timeout: float | None,
    active_pgids=None,
):
    """Run claude on a single skill invocation, retrying on 429 rate limits."""
    cmd = build_sandboxed_command(
        skill, skill_args, worktree_path, model, extra_ro_binds, extra_rw_binds,
        sandbox=sandbox,
    )

    attempt = 0
    while True:
        mode = "w" if attempt == 0 else "a"
        with open(log_file, mode) as log:
            if attempt > 0:
                log.write(f"\n--- retry attempt {attempt} after 429 ---\n")
                log.flush()
            rc, timed_out = _run_in_process_group(
                cmd,
                log,
                cwd=str(worktree_path) if worktree_path is not None else None,
                timeout=timeout,
                active_pgids=active_pgids,
            )
            if timed_out:
                log.write(f"\n--- timed out after {timeout}s ---\n")
                log.flush()
                return skill_args, 124, f"timed out after {timeout}s"

        if rc == 0:
            return skill_args, 0, ""

        tail = _log_tail(log_file)
        if _is_rate_limited(tail) and attempt < RATE_LIMIT_MAX_RETRIES:
            # Exponential backoff with jitter. Jitter spreads simultaneously-
            # throttled workers so they don't re-collide on the same minute
            # window when the limit resets.
            delay = min(
                RATE_LIMIT_BASE_DELAY * (2 ** attempt), RATE_LIMIT_MAX_DELAY
            ) + random.uniform(0, 15)
            time.sleep(delay)
            attempt += 1
            continue

        last_line = tail.strip().rsplit("\n", 1)[-1] if tail.strip() else "could not read log"
        return skill_args, rc, last_line[:200]


def run_worker(
    worker_index: int,
    work_queue,
    skill: str,
    worktree_base: str | None,
    result_queue: multiprocessing.Queue,
    model: str | None,
    extra_ro_binds: list[str],
    extra_rw_binds: list[str],
    sandbox: bool,
    timeout: float | None,
    active_pgids=None,
):
    """Pull skill_args off the shared work queue until a None sentinel.

    A shared queue (vs pre-sharded per-worker chunks) means a worker that
    finishes a fast task immediately picks up the next one instead of idling
    while a slow peer drains its own chunk.
    """
    # Reset SIGINT to Python's default handler: the parent installs a custom
    # handler that would get inherited via fork, but workers should just raise
    # KeyboardInterrupt so the _run_in_process_group cleanup fires.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    worktree_path: Path | None = None
    if worktree_base is not None:
        worktree_path = ensure_worktree(f"{worktree_base}-{worker_index}")

    while True:
        skill_args = work_queue.get()
        if skill_args is None:
            break
        safe_name = skill_args.replace("/", "_").replace(" ", "_")[:80]
        log_file = LOG_DIR / f"{safe_name}.log"
        _, rc, reason = run_invocation(
            skill, skill_args, worktree_path, log_file, model,
            extra_ro_binds, extra_rw_binds, sandbox, timeout,
            active_pgids=active_pgids,
        )
        result_queue.put((worker_index, skill_args, rc, reason))


def run_worker_generator(
    worker_index: int,
    skill: str,
    worktree_base: str | None,
    result_queue: multiprocessing.Queue,
    model: str | None,
    extra_ro_binds: list[str],
    extra_rw_binds: list[str],
    sandbox: bool,
    timeout: float | None,
    generator_cmd: str,
    inflight,
    inflight_lock,
    active_pgids=None,
):
    """Pull the next skill_args from a generator command whenever we need
    work. Exits when the generator returns empty.

    The generator is serialized across workers behind `inflight_lock` so its
    view of what's in flight is always consistent. The lock is released
    before running the claude invocation so other workers can pick up work
    concurrently.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)

    worktree_path: Path | None = None
    if worktree_base is not None:
        worktree_path = ensure_worktree(f"{worktree_base}-{worker_index}")

    while True:
        with inflight_lock:
            lines = run_generator(generator_cmd, list(inflight))
            if not lines:
                break
            skill_args = lines[0]
            inflight.append(skill_args)

        try:
            safe_name = skill_args.replace("/", "_").replace(" ", "_")[:80]
            log_file = LOG_DIR / f"{safe_name}.log"
            _, rc, reason = run_invocation(
                skill, skill_args, worktree_path, log_file, model,
                extra_ro_binds, extra_rw_binds, sandbox, timeout,
                active_pgids=active_pgids,
            )
            result_queue.put((worker_index, skill_args, rc, reason))
        finally:
            with inflight_lock:
                try:
                    inflight.remove(skill_args)
                except ValueError:
                    pass

    # Sentinel so the main loop knows this worker has drained the generator.
    result_queue.put((worker_index, None, None, None))


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


def _install_sigint_handler(active_pgids):
    """SIGINT handler that fans SIGTERM to every tracked claude pgid. First
    ^C raises KeyboardInterrupt for a clean stop; second ^C force-exits.
    Returns a callable that restores the previous handler."""
    sigint_count = [0]

    def handle_sigint(signum, frame):
        # Workers also receive this SIGINT (same foreground process group) and
        # clean up their own claude child via the BaseException branch in
        # _run_in_process_group. This handler is a safety net: it fans out
        # SIGTERM to every claude pgid we know about, covering the case where
        # a worker was between invocations (outside the try/except window).
        sigint_count[0] += 1
        if sigint_count[0] >= 2:
            sys.__stderr__.write("\n[claudes] second SIGINT, force exiting\n")
            os._exit(130)
        sys.__stderr__.write(
            f"\n[claudes] SIGINT received, terminating {len(active_pgids)} "
            f"claude process group(s)...\n"
        )
        for pgid in list(active_pgids):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        raise KeyboardInterrupt

    original = signal.signal(signal.SIGINT, handle_sigint)
    return lambda: signal.signal(signal.SIGINT, original)


def _open_main_log():
    """Create the per-run log file and tee stdout/stderr into it.
    Returns the file handle so callers can close it at teardown."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    main_log = LOG_DIR / f"claudes_{timestamp}.log"
    log_fh = open(main_log, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_fh)
    sys.stderr = TeeStream(sys.__stderr__, log_fh)
    print(f"=== claudes run started at {now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()} ===")
    print(f"Logging to {main_log}")
    return log_fh


def _close_main_log(log_fh):
    log_fh.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def cmd_skill_generator(args, worktree_base):
    """Generator-driven mode: workers pull fresh work from `args.generator`
    on demand instead of pre-sharding a stdin list into per-worker chunks."""
    log_fh = _open_main_log()
    print(f"Generator: {args.generator}")
    print(f"Up to {args.workers} workers")
    print(f"Skill: /{args.skill_name}")

    num_workers = max(1, args.workers)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    active_pgids = manager.list()
    inflight = manager.list()
    inflight_lock = manager.Lock()

    restore_sigint = _install_sigint_handler(active_pgids)
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_worker_generator,
                    idx,
                    args.skill_name,
                    worktree_base,
                    result_queue,
                    args.model,
                    args.ro_bind,
                    args.rw_bind,
                    args.sandbox,
                    args.timeout,
                    args.generator,
                    inflight,
                    inflight_lock,
                    active_pgids,
                )
                for idx in range(num_workers)
            ]

            print("Workers started...")

            workers_done = 0
            while workers_done < num_workers:
                worker_index, skill_args, rc, reason = result_queue.get()
                if skill_args is None:
                    workers_done += 1
                    continue
                completed += 1
                status = "OK" if rc == 0 else f"FAILED (rc={rc}): {reason}"
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] [{completed}] worker-{worker_index} {skill_args}: {status}"
                )

            for future in futures:
                future.result()
    finally:
        restore_sigint()

    print(f"All done. {completed} invocation(s) completed.")
    _close_main_log(log_fh)


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

    if args.generator:
        if args.debug:
            print("--debug and --generator cannot be used together", file=sys.stderr)
            sys.exit(2)
        cmd_skill_generator(args, worktree_base)
        return

    lines = [line.strip() for line in sys.stdin if line.strip()]

    if args.start > 0:
        lines = lines[args.start :]
    if args.limit > 0:
        lines = lines[: args.limit]

    if not lines:
        print("No input lines to process.", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        skill_args = lines[0]
        worktree_path: Path | None = None
        if worktree_base is not None:
            worktree_path = ensure_worktree(f"{worktree_base}-debug")
        cmd = build_sandboxed_command(
            args.skill_name, skill_args, worktree_path, args.model,
            args.ro_bind, args.rw_bind, sandbox=args.sandbox, headless=False,
        )
        print("Debug mode: running 1 invocation inline")
        print(f"Skill: /{args.skill_name} {skill_args}")
        print(f"Command: {' '.join(cmd)}")
        rc, timed_out = _run_in_process_group(
            cmd,
            log=None,
            cwd=str(worktree_path) if worktree_path is not None else None,
            timeout=args.timeout,
        )
        if timed_out:
            print(f"Timed out after {args.timeout}s")
            sys.exit(124)
        sys.exit(rc)

    log_fh = _open_main_log()

    print(f"Processing {len(lines)} invocations with up to {args.workers} workers")
    print(f"Skill: /{args.skill_name}")

    num_workers = min(args.workers, len(lines))

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    active_pgids = manager.list()
    work_queue = manager.Queue()
    for line in lines:
        work_queue.put(line)
    # One sentinel per worker so blocked get() calls all wake cleanly once
    # the real work is drained.
    for _ in range(num_workers):
        work_queue.put(None)

    restore_sigint = _install_sigint_handler(active_pgids)
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_worker,
                    idx,
                    work_queue,
                    args.skill_name,
                    worktree_base,
                    result_queue,
                    args.model,
                    args.ro_bind,
                    args.rw_bind,
                    args.sandbox,
                    args.timeout,
                    active_pgids,
                )
                for idx in range(num_workers)
            ]

            print("Workers started...")

            completed = 0
            while completed < len(lines):
                worker_index, skill_args, rc, reason = result_queue.get()
                completed += 1
                status = "OK" if rc == 0 else f"FAILED (rc={rc}): {reason}"
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{ts}] [{completed}/{len(lines)}] worker-{worker_index} {skill_args}: {status}"
                )

            for future in futures:
                future.result()
    finally:
        restore_sigint()

    print("All done.")
    _close_main_log(log_fh)


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

    worktree_paths = list_worktrees()

    print(f"Cleaning {len(branches)} worker branches for '{worktree_base}'")

    for branch in branches:
        if branch in worktree_paths:
            print(f"Removing worktree for {branch} at {worktree_paths[branch]}")
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_paths[branch])],
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

    # Prune worktree metadata in the main repo and every submodule — the
    # parent `git worktree remove` above doesn't recurse, so submodule
    # worktree gitdirs under .git/modules/*/worktrees/<branch> are orphaned
    # until pruned.
    subprocess.run(["git", "worktree", "prune"], capture_output=True)
    subprocess.run(
        ["git", "submodule", "foreach", "--recursive", "git worktree prune"],
        capture_output=True,
    )
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
        "--model",
        type=str,
        default=None,
        help="Claude model to use (e.g. sonnet, opus, claude-sonnet-4-6)",
    )
    skill_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: take 1 input, run claude directly in this process (no worker pool)",
    )
    skill_parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run claude inside a bwrap sandbox restricted to the worktree (or cwd) plus essential host paths",
    )
    skill_parser.add_argument(
        "--ro-bind",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional host path to expose read-only inside the sandbox (may repeat, requires --sandbox)",
    )
    skill_parser.add_argument(
        "--rw-bind",
        action="append",
        default=[],
        metavar="PATH",
        help="Additional host path to expose writable inside the sandbox (may repeat, requires --sandbox)",
    )
    skill_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Kill each claude invocation if it doesn't finish in N seconds",
    )
    skill_parser.add_argument(
        "--generator",
        type=str,
        default=None,
        metavar="CMD",
        help=(
            "Shell command that emits the next skill_args on stdout. "
            "Invoked whenever a worker needs work; the current in-flight "
            "skill_args are piped on stdin. Empty output = no more work. "
            "When set, stdin is not read and --start/--limit are ignored."
        ),
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
