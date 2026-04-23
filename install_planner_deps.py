from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/Faildes/Chattiori-Model-Merger.git"
REPO_BRANCH = "notebook"
DEFAULT_DEST = Path("tools") / "chattiori_model_merger"

PIP_PACKAGES = [
    "requests",
    "filelock",
    "fake_useragent",
    "huggingface_hub",
    "pillow",
    "papermill",
    "jupyter",
    "nbconvert",
    "nbformat",
    "ipython",
    "ipykernel",
]


class InstallerError(RuntimeError):
    pass


def run(cmd: list[str], *, check: bool = True, cwd: str | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    pretty = " ".join(str(x) for x in cmd)
    print(f"$ {pretty}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture,
    )
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout
        raise InstallerError(f"Command failed with exit code {proc.returncode}: {pretty}\n{detail}".rstrip())
    return proc


def pip_install(packages: list[str]):
    if not packages:
        return
    run([sys.executable, "-m", "pip", "install", *packages])


def _git_ok(dest: Path) -> bool:
    return dest.exists() and (dest / ".git").exists()


def get_local_commit(dest: Path) -> str | None:
    if not _git_ok(dest):
        return None
    proc = run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=False, capture=True)
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def get_remote_commit(repo_url: str = REPO_URL, branch: str = REPO_BRANCH) -> str | None:
    proc = run(["git", "ls-remote", repo_url, f"refs/heads/{branch}"], check=False, capture=True)
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    return line[0].split()[0].strip() or None


def repo_has_update(dest: Path, repo_url: str = REPO_URL, branch: str = REPO_BRANCH) -> tuple[bool | None, str | None, str | None]:
    local_commit = get_local_commit(dest)
    remote_commit = get_remote_commit(repo_url, branch)
    if local_commit is None or remote_commit is None:
        return None, local_commit, remote_commit
    return local_commit != remote_commit, local_commit, remote_commit


def clone_repo(dest: Path, repo_url: str = REPO_URL, branch: str = REPO_BRANCH):
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--branch", branch, "--single-branch", repo_url, str(dest)])


def update_repo(dest: Path, repo_url: str = REPO_URL, branch: str = REPO_BRANCH, *, force: bool = False) -> dict[str, Any]:
    if not _git_ok(dest):
        raise InstallerError(f"Destination is not a git repo: {dest}")

    needs_update, local_commit, remote_commit = repo_has_update(dest, repo_url, branch)
    updated = False

    if force or needs_update is True:
        run(["git", "-C", str(dest), "fetch", "origin", branch])
        run(["git", "-C", str(dest), "checkout", branch])
        run(["git", "-C", str(dest), "pull", "--ff-only", "origin", branch])
        updated = True
    elif needs_update is False:
        print(f"Repository is already up to date at {local_commit}.")
    else:
        print("Could not determine remote update state. Skipping git pull.")

    return {
        "updated": updated,
        "local_commit": get_local_commit(dest),
        "remote_commit": get_remote_commit(repo_url, branch),
        "needs_update": needs_update,
    }


def ensure_repo(dest: Path, *, update_if_needed: bool = True, force_update: bool = False) -> dict[str, Any]:
    dest = Path(dest).expanduser().resolve()
    if dest.exists() and not _git_ok(dest):
        raise InstallerError(f"Destination already exists and is not a git repo: {dest}")

    result: dict[str, Any]
    if not dest.exists():
        clone_repo(dest)
        result = {
            "cloned": True,
            "updated": False,
            "local_commit": get_local_commit(dest),
            "remote_commit": get_remote_commit(),
            "needs_update": False,
        }
    else:
        if update_if_needed or force_update:
            update_result = update_repo(dest, force=force_update)
        else:
            update_result = {
                "updated": False,
                "local_commit": get_local_commit(dest),
                "remote_commit": get_remote_commit(),
                "needs_update": None,
            }
        result = {
            "cloned": False,
            **update_result,
        }

    req = dest / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=False)
    result["path"] = str(dest)
    return result


def install_or_update(
    dest: str | Path = DEFAULT_DEST,
    *,
    skip_pip: bool = False,
    skip_repo: bool = False,
    update_if_needed: bool = True,
    force_update: bool = False,
) -> dict[str, Any]:
    dest = Path(dest).expanduser().resolve()
    print(f"Python: {sys.executable}")
    print(f"Repo destination: {dest}")

    if not skip_pip:
        pip_install(PIP_PACKAGES)

    repo_result: dict[str, Any] = {"path": str(dest), "skipped": True}
    if not skip_repo:
        repo_result = ensure_repo(dest, update_if_needed=update_if_needed, force_update=force_update)

    print("\nDone.")
    print(f"Chattiori Model Merger path: {dest}")
    print("If you use the planner locally, keep this folder at tools/chattiori_model_merger or adjust your paths accordingly.")
    return repo_result


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Install Model Planner dependencies and Chattiori Model Merger.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Clone destination for Chattiori-Model-Merger")
    parser.add_argument("--skip-pip", action="store_true", help="Skip pip installs")
    parser.add_argument("--skip-repo", action="store_true", help="Skip cloning/updating the merger repo")
    parser.add_argument("--check-update", action="store_true", help="Only check whether the repo has updates")
    parser.add_argument("--force-update", action="store_true", help="Force git fetch/pull even if local and remote commits match")
    args = parser.parse_args(argv)

    dest = Path(args.dest).expanduser().resolve()

    if args.check_update:
        needs_update, local_commit, remote_commit = repo_has_update(dest)
        print(f"Python: {sys.executable}")
        print(f"Repo destination: {dest}")
        print(f"Local commit:  {local_commit}")
        print(f"Remote commit: {remote_commit}")
        if needs_update is True:
            print("Update available.")
            return 2
        if needs_update is False:
            print("Already up to date.")
            return 0
        print("Could not determine update state.")
        return 1

    install_or_update(
        dest=dest,
        skip_pip=args.skip_pip,
        skip_repo=args.skip_repo,
        update_if_needed=True,
        force_update=args.force_update,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
