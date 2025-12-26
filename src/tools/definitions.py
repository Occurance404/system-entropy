import ast
import difflib
import os
import re
from typing import Dict, Any, List, Tuple, Optional
from src.tools.base import ToolProtocol
from src.interfaces import SandboxConnectorProtocol
from src.security.secrets import SecretScanner

class ReadFileTool(ToolProtocol):
    name = "read_file"
    description = "Reads a file from the filesystem."

    _DEFAULT_MAX_CHARS = 16_000
    _DEFAULT_MAX_LINES = 400
    _OUTLINE_THRESHOLD_LINES = 200
    _OUTLINE_MAX_IMPORTS = 60
    _OUTLINE_MAX_SYMBOLS = 200

    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        path = args.get('path', '')
        mode = (args.get("mode") or "auto").strip().lower()
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        with_line_numbers = bool(args.get("with_line_numbers", False))

        raw = connector.read_file(path)
        if raw.startswith("Error:") or raw.startswith("Error reading file:"):
            return raw, None

        lines = raw.splitlines()
        line_count = len(lines)

        if isinstance(start_line, int) or isinstance(end_line, int):
            start = 1 if not isinstance(start_line, int) else max(1, start_line)
            end = line_count if not isinstance(end_line, int) else max(1, end_line)
            if end < start:
                start, end = end, start

            snippet_lines = lines[start - 1 : end]
            if with_line_numbers:
                snippet_lines = [f"{i:6d} | {line}" for i, line in enumerate(snippet_lines, start=start)]
            snippet = "\n".join(snippet_lines)
            return SecretScanner.redact(self._truncate(snippet)), None

        if mode == "outline" or (mode == "auto" and line_count > self._OUTLINE_THRESHOLD_LINES):
            outline = self._outline(path, raw)
            return SecretScanner.redact(self._truncate(outline)), None

        if mode == "full" or mode == "auto":
            return SecretScanner.redact(self._truncate(raw)), None

        return f"Error: unknown mode '{mode}'. Use one of: auto, full, outline.", None

    def _truncate(self, text: str) -> str:
        if not text:
            return text

        lines = text.splitlines()
        if len(lines) > self._DEFAULT_MAX_LINES:
            head = "\n".join(lines[: int(self._DEFAULT_MAX_LINES * 0.6)])
            tail = "\n".join(lines[-int(self._DEFAULT_MAX_LINES * 0.4) :])
            text = (
                f"{head}\n\n... <truncated {len(lines) - self._DEFAULT_MAX_LINES} lines> ...\n\n{tail}"
            )

        if len(text) > self._DEFAULT_MAX_CHARS:
            head = text[: int(self._DEFAULT_MAX_CHARS * 0.6)]
            tail = text[-int(self._DEFAULT_MAX_CHARS * 0.4) :]
            text = f"{head}\n\n... <truncated {len(text) - self._DEFAULT_MAX_CHARS} chars> ...\n\n{tail}"
        return text

    def _outline(self, path: str, raw: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            return self._outline_python(path, raw)
        return self._outline_generic(path, raw)

    def _outline_python(self, path: str, raw: str) -> str:
        lines = raw.splitlines()
        header = [f"# Outline for {path} ({len(lines)} lines)"]
        header.append("# Tip: call read_file with start_line/end_line for exact bodies.")

        try:
            tree = ast.parse(raw)
        except Exception:
            header.append("# NOTE: Python parse failed; falling back to generic outline.")
            return "\n".join(header + [""] + [self._outline_generic(path, raw)])

        import_line_nos: List[int] = []
        symbols: List[Tuple[int, int, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_line_nos.append(getattr(node, "lineno", 0) or 0)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", 0) or 0
                end = getattr(node, "end_lineno", start) or start
                if 1 <= start <= len(lines):
                    signature = lines[start - 1].rstrip()
                else:
                    signature = f"<unknown signature for {getattr(node, 'name', '<anon>')}>"
                symbols.append((start, end, signature))

        import_line_nos = sorted({n for n in import_line_nos if n > 0})
        symbols.sort(key=lambda x: x[0])

        body: List[str] = []
        if import_line_nos:
            body.append("## Imports")
            shown = import_line_nos[: self._OUTLINE_MAX_IMPORTS]
            for n in shown:
                if 1 <= n <= len(lines):
                    body.append(f"{n:6d} | {lines[n - 1].rstrip()}")
            if len(import_line_nos) > len(shown):
                body.append(f"... <{len(import_line_nos) - len(shown)} more imports omitted> ...")

        if symbols:
            body.append("## Symbols")
            shown_syms = symbols[: self._OUTLINE_MAX_SYMBOLS]
            for start, end, sig in shown_syms:
                body.append(f"{start:6d}-{end:<6d} | {sig}")
            if len(symbols) > len(shown_syms):
                body.append(f"... <{len(symbols) - len(shown_syms)} more symbols omitted> ...")

        if not body:
            body.append("# (No imports or top-level defs found.)")

        return "\n".join(header + [""] + body)

    def _outline_generic(self, path: str, raw: str) -> str:
        lines = raw.splitlines()
        out: List[str] = [
            f"# Outline for {path} ({len(lines)} lines)",
            "# Tip: call read_file with start_line/end_line for exact content.",
            "",
        ]

        import_re = re.compile(
            r"^\s*(from\s+\S+\s+import\s+|import\s+\S+|#include\s+<|#include\s+\"|using\s+\S+|require\(|import\s+.*\s+from\s+)"
        )
        symbol_re = re.compile(r"^\s*(class|def|function|func|struct|interface|enum)\b")

        imports: List[Tuple[int, str]] = []
        symbols: List[Tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            if import_re.search(line):
                imports.append((idx, line.rstrip()))
            if symbol_re.search(line):
                symbols.append((idx, line.rstrip()))

        if imports:
            out.append("## Imports")
            for idx, line in imports[: self._OUTLINE_MAX_IMPORTS]:
                out.append(f"{idx:6d} | {line}")
            if len(imports) > self._OUTLINE_MAX_IMPORTS:
                out.append(f"... <{len(imports) - self._OUTLINE_MAX_IMPORTS} more imports omitted> ...")
            out.append("")

        if symbols:
            out.append("## Symbols")
            for idx, line in symbols[: self._OUTLINE_MAX_SYMBOLS]:
                out.append(f"{idx:6d} | {line}")
            if len(symbols) > self._OUTLINE_MAX_SYMBOLS:
                out.append(f"... <{len(symbols) - self._OUTLINE_MAX_SYMBOLS} more symbols omitted> ...")
            out.append("")

        if not imports and not symbols:
            preview = lines[: min(80, len(lines))]
            out.append("## Preview")
            out.extend(f"{idx:6d} | {line}" for idx, line in enumerate(preview, start=1))
            if len(lines) > len(preview):
                out.append(f"... <{len(lines) - len(preview)} more lines omitted> ...")

        return "\n".join(out).rstrip()

class WriteFileTool(ToolProtocol):
    name = "write_file"
    description = "Writes content to a file."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        path = args.get('path', '')
        content = args.get('content', '')

        policy = SecretScanner.policy()
        if policy != "off":
            old = connector.read_file(path)
            if old.startswith("Error:") or old.startswith("Error reading file:"):
                old = ""

            diff = difflib.unified_diff(
                old.splitlines(),
                content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
            findings = SecretScanner.scan_added_lines_from_unified_diff(diff)
            if findings and policy == "block":
                return SecretScanner.format_block_message(path, findings, content), self._measure_cbf(content) if path.endswith('.py') else None

        success = connector.write_file(path, content)
        result = "File written successfully." if success else "Failed to write file."
        
        # Calculate CBF if python file
        cbf = None
        if path.endswith('.py'):
            cbf = self._measure_cbf(content)
            
        if policy == "warn":
            # Best-effort scanning on full content; redaction happens on tool outputs elsewhere.
            full_findings = SecretScanner.scan_text(content)
            if full_findings:
                kinds = ", ".join(sorted({f.kind for f in full_findings}))
                result = f"{result} (WARNING: possible secrets detected: {kinds})"

        return result, cbf

    def _measure_cbf(self, code: str) -> int:
        try:
            tree = ast.parse(code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
                    complexity += 1
            return complexity
        except:
            return -1

class ExecutePythonTool(ToolProtocol):
    name = "execute_python"
    description = "Executes a python script."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        script_path = args.get('script_path', '')
        exit_code, output = connector.execute_command(f"python3 {script_path}")
        output = SecretScanner.redact(output)
        output = self._truncate_output(output)
        return f"Exit Code: {exit_code}\nOutput:\n{output}", None

    def _truncate_output(self, text: str, max_chars: int = 12_000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: int(max_chars * 0.3)]
        tail = text[-int(max_chars * 0.7) :]
        return f"{head}\n\n... <truncated {len(text) - max_chars} chars> ...\n\n{tail}"

class RunShellTool(ToolProtocol):
    name = "run_shell"
    description = "Executes a shell command."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        command = args.get('command', '')
        exit_code, output = connector.execute_command(command)
        output = SecretScanner.redact(output)
        output = self._truncate_output(output)
        return f"Exit Code: {exit_code}\nOutput:\n{output}", None

    def _truncate_output(self, text: str, max_chars: int = 12_000) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: int(max_chars * 0.3)]
        tail = text[-int(max_chars * 0.7) :]
        return f"{head}\n\n... <truncated {len(text) - max_chars} chars> ...\n\n{tail}"

class SearchWebTool(ToolProtocol):
    name = "search_web"
    description = "Searches the web."
    
    def execute(self, args: Dict[str, Any], connector: SandboxConnectorProtocol) -> Tuple[str, Optional[int]]:
        return "Search is currently disabled in the sandbox environment.", None
