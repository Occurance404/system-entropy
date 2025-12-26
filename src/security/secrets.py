from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    line_no: Optional[int]


class SecretScanner:
    """
    Lightweight secret detection/redaction for agent-controlled workspaces.

    Goal: prevent accidental hard-coding and logging of common credential formats
    without pulling heavy dependencies (detect-secrets/trufflehog) in V1.
    """

    _PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
        ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
        ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
        ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
        ("github_token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
        ("aws_access_key_id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
        ("slack_token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
        ("stripe_live_key", re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b")),
    )

    @classmethod
    def policy(cls) -> str:
        """
        Secret policy for write operations.

        Values:
          - "block": reject writes that introduce likely secrets (default)
          - "warn": allow writes, but still redact tool outputs
          - "off": no scanning/redaction (not recommended)
        """
        return os.getenv("SECRETS_POLICY", "block").strip().lower() or "block"

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return text
        redacted = text
        for kind, pattern in cls._PATTERNS:
            redacted = pattern.sub(f"<REDACTED:{kind}>", redacted)
        return redacted

    @classmethod
    def scan_text(cls, text: str) -> List[SecretFinding]:
        findings: List[SecretFinding] = []
        if not text:
            return findings
        for line_no, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in cls._PATTERNS:
                if pattern.search(line):
                    findings.append(SecretFinding(kind=kind, line_no=line_no))
        return findings

    @classmethod
    def sha256(cls, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @classmethod
    def scan_added_lines_from_unified_diff(cls, diff_lines: Iterable[str]) -> List[SecretFinding]:
        """
        Extracts added lines with their *new file* line numbers from a unified diff,
        then scans those added lines for secrets.
        """
        findings: List[SecretFinding] = []
        new_line_no: Optional[int] = None

        hunk_re = re.compile(r"\+(\d+)(?:,(\d+))?")

        for line in diff_lines:
            if line.startswith("@@"):
                match = hunk_re.search(line)
                new_line_no = int(match.group(1)) if match else None
                continue

            if new_line_no is None:
                continue

            if line.startswith("+") and not line.startswith("+++"):
                text = line[1:]
                for kind, pattern in cls._PATTERNS:
                    if pattern.search(text):
                        findings.append(SecretFinding(kind=kind, line_no=new_line_no))
                new_line_no += 1
                continue

            if line.startswith("-") and not line.startswith("---"):
                continue

            if line.startswith(" "):
                new_line_no += 1
                continue

        return findings

    @classmethod
    def format_block_message(cls, path: str, findings: Sequence[SecretFinding], content: str) -> str:
        kinds = sorted({f.kind for f in findings})
        lines = sorted({f.line_no for f in findings if f.line_no is not None})
        digest = cls.sha256(content)[:12]
        where = f"lines {', '.join(map(str, lines))}" if lines else "unknown lines"
        return (
            "Security Error: refusing to write likely secret material.\n"
            f"- path: {path}\n"
            f"- kinds: {', '.join(kinds)}\n"
            f"- location: {where}\n"
            f"- content_sha256: {digest}\n"
            "Fix: use environment variables (e.g., `os.getenv(...)`) or mock values; never hardcode real keys."
        )
