# claudes

Run many Claude Code instances in parallel, each working in its own git worktree.

Feed a list of tasks via stdin, and `claudes` fans them out across worker processes that each invoke a Claude skill. After the work is done, merge all branches back together and clean up.

## Installation

```bash
pip install .
```

This installs the `claudes` command globally.

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) installed and authenticated
- Git repository as the working directory

## Usage

### 1. Run parallel skill invocations

Pipe one task per line into `claudes skill`. Each line becomes the argument string passed to the named Claude skill.

```bash
echo -e "model-a\nmodel-b\nmodel-c" | claudes skill --workers 3 --worktree my-branch port-huggingface-model
```

| Flag | Description |
|------|-------------|
| `--workers N` | Max concurrent Claude instances (default: 25) |
| `--worktree [BRANCH]` | Run each worker in a git worktree. Branches are named `<BRANCH>-0`, `<BRANCH>-1`, etc. Defaults to the current branch name if no value is given. |
| `--start N` | Skip the first N input lines |
| `--limit N` | Process at most N invocations (0 = all) |

### 2. Merge worker branches

After all workers finish, rebase and merge every worker branch into the current branch:

```bash
claudes merge my-branch
```

Branches are merged in order (`my-branch-0`, `my-branch-1`, ...) using the `theirs` merge strategy. If no branch base is given, the current branch name is used.

### 3. Clean up

Remove all worker branches and their worktrees:

```bash
claudes clean my-branch
```

## Full example

```bash
# Generate a list of tasks (one per line)
cat tasks.txt | claudes skill --workers 5 --worktree feature port-huggingface-model

# Merge all worker branches into the current branch
claudes merge feature

# Delete worker branches and worktrees
claudes clean feature
```

## Logs

All output is logged to `.claudes/logs/` in the working directory. Each invocation gets its own log file, plus a timestamped main log capturing overall progress.
