#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coding-agent-traces @ git+https://github.com/Infatoshi/traces@3e3b2cb05809f7e5d533ad53f422b476c4442bbf",
# ]
# ///
"""Export coding-agent usage and traces from one folder (e.g. the folder you used for a course).

Run this script from the folder that contains all your work for the course and upload the resulting zip file::

    uv run export_agent_data.py

Every agent session that was started inside this folder, or any of its subfolders, is included. Each session is tagged
with the folder it was started in (relative to the root folder), so start your coding agents from the folder of the task
you are working on. Use --exclude to leave out subfolders.

What happens:

1. Conversations of all coding agents on this machine are extracted into a temporary folder.
2. Only conversations that ran inside the root folder are kept (everything else is dropped). The temporary folder is
   deleted at the end.
3. The kept conversations are redacted (your username, home path, secrets, e-mails, ...), verified and exported.
4. Token usage from `ccusage` is filtered down to the kept conversations.
5. Everything is zipped. No username, git author, e-mail or hostname is written to the zip. Use --keep-dir to also
   write the unzipped files somewhere so you can inspect exactly what you are about to upload.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import posixpath
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "2.0.0"
SCHEMA_VERSION = 2
CCUSAGE_VERSION = "20.0.31"
CCUSAGE_TIMEOUT = 300  # seconds
INDEX_KEY = "course_idx"  # field injected into raw conversations to track them through redaction

USAGE_FIELDS = ("inputTokens", "outputTokens", "cacheCreationTokens", "cacheReadTokens", "totalTokens", "totalCost")
WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
MIN_ID_LENGTH = 8  # shorter session ids are never matched by substring


class ExportError(Exception):
    """Expected failure that is reported to the student without a traceback."""


def info(message: str = "") -> None:
    """Print a progress message to stderr."""
    print(message, file=sys.stderr, flush=True)


def warn(message: str) -> None:
    """Print a loud warning to stderr."""
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Paths and classification
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathKey:
    """A path split into components, with a flag telling if comparisons should ignore case."""

    parts: tuple[str, ...]
    case_insensitive: bool

    def folded(self, case_insensitive: bool) -> tuple[str, ...]:
        """Return the components, lower-cased if needed."""
        return tuple(p.lower() for p in self.parts) if case_insensitive else self.parts


def make_key(path: str) -> PathKey:
    """Normalize a (possibly Windows-style) path string to components, resolving symlinks where possible."""
    text = path.strip()
    windows_style = bool(WINDOWS_PATH_RE.match(text)) or text.startswith("\\\\")
    # foreign (Windows-style on a non-Windows host) paths are handled purely lexically
    text = text.replace("\\", "/") if windows_style and os.name != "nt" else os.path.realpath(text)
    text = posixpath.normpath(text.replace("\\", "/"))
    if text.startswith("//?/"):
        text = text[4:]
    parts = tuple(p for p in text.split("/") if p and p != ".")
    return PathKey(parts, windows_style or os.name == "nt")


def is_inside(child: PathKey, parent: PathKey) -> bool:
    """Return True if child is parent or located below it."""
    ci = child.case_insensitive or parent.case_insensitive
    n = len(parent.parts)
    return child.folded(ci)[:n] == parent.folded(ci)


@dataclass
class Layout:
    """The root folder and the subfolders that are excluded."""

    root: str
    exclude: list[str]
    root_key: PathKey = field(init=False)
    exclude_keys: list[PathKey] = field(init=False)

    def __post_init__(self) -> None:
        """Pre-compute normalized paths."""
        self.root_key = make_key(self.root)
        self.exclude_keys = [make_key(f"{self.root}/{name}") for name in self.exclude]


def classify(path: str | None, layout: Layout) -> tuple[str | None, str]:
    """Classify a conversation by its working directory.

    Returns (folder, reason), where folder is the working directory relative to the root ("." for the root itself).
    The folder is None if the conversation must be dropped, in which case reason is one of "unresolved",
    "outside_root", "excluded".
    """
    if not path:
        return None, "unresolved"
    key = make_key(path)
    if not is_inside(key, layout.root_key):
        return None, "outside_root"
    if any(is_inside(key, excluded) for excluded in layout.exclude_keys):
        return None, "excluded"
    below = key.parts[len(layout.root_key.parts) :]
    return "/".join(below) or ".", "kept"


def top_folder(folder: str) -> str:
    """The top-level subfolder of the root that a folder belongs to."""
    return folder.split("/", 1)[0]


def workspace_folder(conv: dict[str, Any], installations: list[Path]) -> str | None:
    """Resolve the folder of a Cursor/Windsurf workspace id via workspaceStorage/<id>/workspace.json."""
    workspace_id = conv.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id:
        return None
    candidates = [Path(conv["installation"])] if conv.get("installation") else []
    for install in [*candidates, *installations]:
        file = install / "User" / "workspaceStorage" / workspace_id / "workspace.json"
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        uri = data.get("folder")
        if isinstance(uri, str):
            return uri_to_path(uri)
        uri = data.get("workspace")  # multi-root: path of the .code-workspace file
        if isinstance(uri, str) and (path := uri_to_path(uri)):
            return posixpath.dirname(path.replace("\\", "/"))
    return None


def uri_to_path(uri: str) -> str | None:
    """Convert a local file:// URI to a path string (None for remote URIs)."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    path = urllib.parse.unquote(parsed.path)
    if re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return path


def conversation_path(conv: dict[str, Any], installations: list[Path]) -> str | None:
    """Return the working directory of a conversation, if it can be determined."""
    path = conv.get("project_path")
    if isinstance(path, str) and path.strip():
        return path
    return workspace_folder(conv, installations)


# ----------------------------------------------------------------------------------------------------------------------
# Root folder
# ----------------------------------------------------------------------------------------------------------------------


def find_root(root_arg: str | None) -> Path:
    """Return the root folder (--root or the current directory), refusing folders that would include unrelated work."""
    root = Path(root_arg).expanduser().resolve() if root_arg else Path.cwd().resolve()
    if not root.is_dir():
        msg = f"{root} is not a directory."
        raise ExportError(msg)
    home = Path.home().resolve()
    if home.is_relative_to(root):
        msg = (
            f"{root} is your home folder (or above it), so all your agent sessions would be included.\n"
            "Please cd into the folder that contains your work for the course, or pass --root PATH."
        )
        raise ExportError(msg)
    return root


# ----------------------------------------------------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------------------------------------------------


def iter_jsonl(path: Path):
    """Yield the JSON objects of a JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def find_installations() -> list[Path]:
    """Locate Cursor/Windsurf installations (used to resolve workspace ids to folders)."""
    from traces.extract.cursor import CursorExtractor
    from traces.extract.windsurf import WindsurfExtractor

    found: list[Path] = []
    for extractor in (CursorExtractor, WindsurfExtractor):
        try:
            found.extend(extractor(Path.home()).find_installations())
        except OSError:
            continue
    return found


def timestamp_to_ms(value: Any) -> Any:
    """Turn ISO timestamps into epoch milliseconds so that redaction rules cannot mangle them (e.g. the year)."""
    if not isinstance(value, str):
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return int(parsed.replace(tzinfo=parsed.tzinfo or UTC).timestamp() * 1000)


@dataclass
class Kept:
    """A kept conversation (raw identity is only held in memory)."""

    folder: str
    session_id: str | None
    parent_session_id: str | None


def filter_raw(raw_dir: Path, out_dir: Path, layout: Layout) -> tuple[list[Kept], Counter[str]]:
    """Classify raw conversations and write only the kept ones (tagged with an index) to out_dir."""
    installations = find_installations()
    kept: list[Kept] = []
    dropped: Counter[str] = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(raw_dir.glob("*.jsonl")):
        with (out_dir / src.name).open("w", encoding="utf-8") as out:
            for conv in iter_jsonl(src):
                folder, reason = classify(conversation_path(conv, installations), layout)
                if folder is None:
                    dropped[reason] += 1
                    continue
                conv[INDEX_KEY] = len(kept)
                for key in ("created_at", "updated_at"):
                    conv[key] = timestamp_to_ms(conv.get(key))
                kept.append(Kept(folder, conv.get("session_id"), conv.get("parent_session_id")))
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
    return kept, dropped


def redact_and_verify(filtered: Path, work: Path, root: Path) -> tuple[Path, dict[str, Any]]:
    """Redact the filtered conversations and drop rows that still fail the leftover scan."""
    from traces.config import RedactorConfig
    from traces.redact.pipeline import run_redact
    from traces.verify.scanner import clean_tree, scan

    home = Path.home()
    terms = [] if root.is_relative_to(home) else [str(root)]
    config = RedactorConfig(user_name=getpass.getuser(), home_dir=str(home), private_terms=terms)
    redacted = work / "redacted"
    report = run_redact(filtered, redacted, config)
    verify: dict[str, Any] = {"leftovers_before_clean": 0, "rows_dropped": 0, "ok": True}
    scanned = scan(redacted)
    final = redacted
    if not scanned.ok:
        verify["leftovers_before_clean"] = len(scanned.leftovers)
        warn(f"{len(scanned.leftovers)} leftover(s) found after redaction; those rows are dropped.")
        final = work / "redacted_clean"
        cleaned = clean_tree(redacted, final)
        verify["rows_dropped"] = cleaned["rows_dropped"]
        verify["dropped_by_category"] = cleaned["dropped_by_category"]
        verify["ok"] = scan(final).ok
    return final, {"counts": dict(report["counts"]), "verify": verify}


def export_traces(in_dir: Path, out_dir: Path, metadata_key: str) -> dict[str, int]:
    """Export all four variants and return the number of rows per file."""
    from traces.config import ExportConfig
    from traces.export.execution import export_assistant_steps, export_execution
    from traces.export.messages import export_messages
    from traces.export.sharegpt import export_sharegpt

    config = ExportConfig(variant="all", metadata_key=metadata_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "messages": export_messages(in_dir, out_dir / "messages.jsonl", config)["rows"],
        "sharegpt": export_sharegpt(in_dir, out_dir / "sharegpt.jsonl", config)["pairs"],
        "execution": export_execution(in_dir, out_dir / "execution.jsonl", config)["rows"],
        "assistant_steps": export_assistant_steps(in_dir, out_dir / "assistant_steps.jsonl", config)["rows"],
    }


def verify_export(export_dir: Path, work: Path) -> dict[str, Any]:
    """Scan the exported files; replace them by a cleaned copy if anything is found."""
    from traces.verify.scanner import clean_tree, scan

    report = scan(export_dir)
    result: dict[str, Any] = {"leftovers": len(report.leftovers), "rows_dropped": 0, "ok": True}
    if report.ok:
        return result
    warn(f"{len(report.leftovers)} leftover(s) found in the export; those rows are dropped.")
    cleaned_dir = work / "export_clean"
    cleaned = clean_tree(export_dir, cleaned_dir)
    result["rows_dropped"] = cleaned["rows_dropped"]
    shutil.rmtree(export_dir)
    shutil.copytree(cleaned_dir, export_dir)
    result["ok"] = scan(export_dir).ok
    return result


def count_exported_rows(export_dir: Path) -> dict[str, Counter[str]]:
    """Count exported rows per trace_id and file type (so sessions can be joined to the export)."""
    counts: dict[str, Counter[str]] = {}
    for file in export_dir.rglob("*.jsonl"):
        for row in iter_jsonl(file):
            trace_id = row["metadata"]["trace_id"]
            counts.setdefault(trace_id, Counter())[file.stem] += 1
    return counts


# ----------------------------------------------------------------------------------------------------------------------
# ccusage
# ----------------------------------------------------------------------------------------------------------------------


def ccusage_commands() -> list[list[str]]:
    """Possible ways of invoking ccusage, in order of preference."""
    pinned = f"ccusage@{CCUSAGE_VERSION}"
    commands = []
    if shutil.which("ccusage"):
        commands.append(["ccusage"])
    if shutil.which("bunx"):
        commands.append(["bunx", pinned])
    if shutil.which("npx"):
        commands.append(["npx", "-y", pinned])
    return commands


def run_ccusage() -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    """Run ccusage and return (session rows, version, error message)."""
    commands = ccusage_commands()
    if not commands:
        return None, None, "neither ccusage, bunx nor npx was found on PATH"
    error = ""
    for command in commands:
        info(f"Running {' '.join(command)} session --json --by-agent (this can take a minute)...")
        try:
            proc = subprocess.run(
                [*command, "session", "--json", "--by-agent"],
                capture_output=True,
                text=True,
                timeout=CCUSAGE_TIMEOUT,
                check=False,
                encoding="utf-8",
            )
            if proc.returncode != 0:
                error = f"{' '.join(command)} exited with code {proc.returncode}: {proc.stderr.strip()[-300:]}"
                continue
            data = json.loads(proc.stdout[proc.stdout.index("{") :])
            return list(data["session"]), ccusage_version(command), None
        except subprocess.TimeoutExpired:
            error = f"{' '.join(command)} timed out after {CCUSAGE_TIMEOUT}s"
        except (OSError, ValueError, KeyError) as exc:
            error = f"{' '.join(command)} failed: {exc!r}"
    return None, None, error


def ccusage_version(command: list[str]) -> str:
    """Return the version of the ccusage that was used."""
    if command[0] != "ccusage":
        return CCUSAGE_VERSION
    try:
        proc = subprocess.run([*command, "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def id_matches(session_id: str | None, period: str) -> bool:
    """Exact match, or session id contained in a (path-like) period."""
    if not session_id:
        return False
    return session_id == period or (len(session_id) >= MIN_ID_LENGTH and session_id in period)


def match_usage(
    rows: list[dict[str, Any]], kept: list[Kept], trace_ids: dict[int, str]
) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    """Keep only ccusage rows belonging to kept conversations; replace the period by trace ids.

    A row is attributed to the conversations whose own session id matches; if there are none, to subagent
    conversations whose parent session id matches. Returns the output rows and conversation -> row indices.
    """
    out: list[dict[str, Any]] = []
    per_conv: dict[int, list[int]] = {}
    for row in rows:
        period = str(row.get("period", ""))
        matches = [i for i in trace_ids if id_matches(kept[i].session_id, period)]
        if not matches:
            matches = [i for i in trace_ids if id_matches(kept[i].parent_session_id, period)]
        if not matches:
            continue
        first = kept[matches[0]]
        last = (row.get("metadata") or {}).get("lastActivity")
        out.append(
            {
                "trace_ids": [trace_ids[i] for i in matches],
                "folder": first.folder,
                "agent": row.get("agent"),
                "last_activity": str(last)[:10] if last else None,
                "models": row.get("modelsUsed", []),
                "model_breakdowns": row.get("modelBreakdowns", []),
                **{k: row.get(k, 0) for k in USAGE_FIELDS},
            }
        )
        for i in matches:
            per_conv.setdefault(i, []).append(len(out) - 1)
    return out, per_conv


def sum_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Sum the usage fields over rows."""
    return {k: sum(r.get(k, 0) for r in rows) for k in USAGE_FIELDS}


# ----------------------------------------------------------------------------------------------------------------------
# Git state and manifest
# ----------------------------------------------------------------------------------------------------------------------


def git(path: Path, *args: str) -> str | None:
    """Run a git command in path and return stdout (None on failure)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, timeout=60, check=False, encoding="utf-8"
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_state(root: Path, layout: Layout) -> dict[str, Any]:
    """Commit, branch, dirty flag and commit count of the root and every top-level git repo (no author or remote)."""
    excluded = set(layout.exclude)
    state: dict[str, Any] = {}
    for path in [root, *sorted(root.iterdir())]:
        if path != root and (not path.is_dir() or path.name.startswith(".") or path.name in excluded):
            continue
        if not (path / ".git").exists():
            continue
        count = git(path, "rev-list", "--count", "HEAD")
        status = git(path, "status", "--porcelain")
        state["." if path == root else path.name] = {
            "commit": git(path, "rev-parse", "HEAD"),
            "branch": git(path, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "commit_count": int(count) if count and count.isdigit() else None,
        }
    return state


def package_version(name: str) -> str:
    """Installed version of a Python package."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sessions(
    final: Path, kept: list[Kept], metadata_key: str, usage_rows: list[dict[str, Any]], per_conv: dict[int, list[int]]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Build one sessions.jsonl row per surviving conversation."""
    from traces.export.metadata import build_metadata

    sessions: list[dict[str, Any]] = []
    index_to_trace: dict[int, str] = {}
    metas: dict[int, dict[str, Any]] = {}
    parents: dict[int, str | None] = {}
    for src in sorted(final.glob("*.jsonl")):
        for conv in iter_jsonl(src):
            idx = conv.get(INDEX_KEY)
            if not isinstance(idx, int) or not 0 <= idx < len(kept):
                continue
            metas[idx] = build_metadata(conv, metadata_key=metadata_key)
            index_to_trace[idx] = metas[idx]["trace_id"]
            parents[idx] = conv.get("parent_session_id")
    by_session = {kept[i].session_id: t for i, t in index_to_trace.items() if kept[i].session_id}
    for idx, meta in sorted(metas.items()):
        item = kept[idx]
        row: dict[str, Any] = {
            "trace_id": meta["trace_id"],
            "source": meta["source"],
            "folder": item.folder,
            "session": {k: v for k, v in meta["session"].items() if k in ("created_at", "updated_at")},
            "models": meta["models"],
            "counts": meta["counts"],
        }
        if item.parent_session_id:
            row["parent_trace_id"] = by_session.get(item.parent_session_id)
        used = [usage_rows[i] for i in per_conv.get(idx, [])]
        if used:
            row["usage"] = {**sum_usage(used), "shared": any(len(u["trace_ids"]) > 1 for u in used)}
        sessions.append(row)
    return sessions, index_to_trace


def folder_summary(sessions: list[dict[str, Any]], usage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per top-level folder counts for the manifest."""
    summary: dict[str, Any] = {}
    for top in sorted({top_folder(s["folder"]) for s in sessions}):
        subset = [s for s in sessions if top_folder(s["folder"]) == top]
        urows = [u for u in usage_rows if top_folder(u["folder"]) == top]
        summary[top] = {
            "sessions": len(subset),
            "agents": sorted({s["source"] for s in subset}),
            "subfolders": dict(Counter(s["folder"] for s in subset)),
            "usage": {"matched_rows": len(urows), **sum_usage(urows)},
        }
    return summary


def make_zip(source: Path, target: Path) -> None:
    """Zip a directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(source.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(source).as_posix())


def print_summary(summary: dict[str, Any], zip_path: Path) -> None:
    """Print the per-folder summary table."""
    print("\nSummary of what was exported")
    width = max([len("folder"), *(len(name) for name in summary)]) + 2
    print(f"{'folder':<{width}}{'sessions':>9}{'tokens':>16}   agents")
    for name, item in summary.items():
        tokens = int(item["usage"]["totalTokens"])
        print(f"{name:<{width}}{item['sessions']:>9}{tokens:>16,}   {', '.join(item['agents']) or '-'}")
    print(f"\nWrote {zip_path}")
    print("Please upload this file to DTU Learn.")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Run the full export."""
    from traces.extract.discovery import run_extract

    root = find_root(args.root)
    exclude = [name.strip("/\\") for name in args.exclude]
    layout = Layout(str(root), exclude)
    print(f"Root folder: {root}", flush=True)
    print(f"  excluded subfolders: {', '.join(exclude) or '(none)'}", flush=True)
    output = Path(args.output).expanduser().resolve() if args.output else root / "agent-data.zip"
    keep = Path(args.keep_dir).expanduser().resolve() if args.keep_dir else None
    if keep is not None and keep.exists() and (not keep.is_dir() or any(keep.iterdir())):
        msg = f"--keep-dir {keep} already exists and is not an empty folder; please choose a new folder."
        raise ExportError(msg)
    metadata_key = secrets.token_hex(32)

    with tempfile.TemporaryDirectory(prefix="agent-export-") as tmp:
        work = Path(tmp)
        info("Extracting agent conversations (temporary, deleted afterwards)...")
        counts = run_extract(None, Path.home(), work / "raw")
        info(f"  found {sum(counts.values())} conversations on this machine; keeping those from {root.name}/")
        kept, dropped = filter_raw(work / "raw" / "raw_extracted", work / "filtered", layout)
        info("Redacting...")
        final, redaction = redact_and_verify(work / "filtered", work, root)
        info("Exporting traces...")
        out = work / "package"
        exported = export_traces(final, out / "traces", metadata_key)
        export_check = verify_export(out / "traces", work)
        rows_per_trace = count_exported_rows(out / "traces")

        usage_rows, ccusage_ver, usage_error = run_ccusage()
        if usage_rows is None:
            warn(f"ccusage unavailable, token usage will be missing: {usage_error}")
        sessions, index_to_trace = build_sessions(final, kept, metadata_key, [], {})
        matched: list[dict[str, Any]] = []
        if usage_rows is not None:
            matched, per_conv = match_usage(usage_rows, kept, index_to_trace)
            sessions, index_to_trace = build_sessions(final, kept, metadata_key, matched, per_conv)
        for session in sessions:
            session["exported_rows"] = dict(rows_per_trace.get(session["trace_id"], {}))
        known = {s["trace_id"] for s in sessions}
        orphans = sorted(set(rows_per_trace) - known)
        if orphans:
            warn(f"{len(orphans)} exported trace ids are missing from sessions.jsonl")

        write_jsonl(out / "sessions.jsonl", sessions)
        write_jsonl(out / "usage.jsonl", matched)
        summary = folder_summary(sessions, matched)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now(UTC).date().isoformat(),
            "versions": {
                "coding_agent_traces": package_version("coding-agent-traces"),
                "ccusage": ccusage_ver,
                "python": platform.python_version(),
                "os": platform.system(),
            },
            "excluded_folders": exclude,
            "folders": summary,
            "exported_rows": exported,
            "dropped": {
                "outside_root": dropped["outside_root"],
                "excluded": dropped["excluded"],
                "unattributed": dropped["unresolved"],
                "redaction_leftovers": redaction["verify"]["rows_dropped"],
            },
            "usage": {
                "available": usage_rows is not None,
                "error": usage_error,
                "matched_rows": len(matched),
                "unmatched_rows_discarded": (len(usage_rows) - len(matched)) if usage_rows is not None else None,
            },
            "redaction": {"counts": redaction["counts"]},
            "verify": {"redacted": redaction["verify"], "export": export_check, "orphan_trace_ids": len(orphans)},
            "git": git_state(root, layout),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        make_zip(out, output)
        if keep is not None:
            shutil.copytree(out, keep, dirs_exist_ok=True)
            print(f"Unzipped copy for inspection: {keep}")

    print_summary(summary, output)
    print(
        f"Not included: {dropped['outside_root']} outside {root.name}/, {dropped['excluded']} in excluded folders, "
        f"{dropped['unresolved']} without a known folder."
    )
    if not sessions:
        warn(
            f"No sessions were found in {root}. You probably ran the script from the wrong folder (use --root) or "
            "started your coding agents outside this folder."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export coding-agent usage and traces of all sessions started inside one folder."
    )
    parser.add_argument("--root", help="folder to export sessions from (default: the current directory)")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBFOLDER",
        help="subfolder of the root (relative path) whose sessions are left out; can be given multiple times",
    )
    parser.add_argument("--output", help="zip file to write (default: <root>/agent-data.zip)")
    parser.add_argument("--keep-dir", help="also write the unzipped export to this folder for inspection")
    return parser.parse_args(argv)


def main() -> int:
    """Entry point."""
    args = parse_args()
    try:
        run(args)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; temporary files were deleted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
