"""Shared local startup for the Windows and macOS double-click launchers."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)

# Check the current requirements rather than trusting an old installation marker.
# packaging is a Streamlit dependency; if missing, setup installs requirements.
DEPENDENCY_CHECK = """
import importlib
from importlib.metadata import version
from pathlib import Path
import sys
from packaging.requirements import Requirement
for line in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    line = line.split(' #', 1)[0].strip()
    if not line or line.startswith('#'):
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = version(requirement.name)
    if not requirement.specifier.contains(installed):
        raise RuntimeError(f'{requirement}: installed version is {installed}')
for name in ('streamlit', 'pandas', 'numpy', 'scipy', 'sklearn', 'plotly'):
    importlib.import_module(name)
"""


def run_logged(command: list[str], root: Path, log) -> bool:
    log.flush()
    return subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                          check=False).returncode == 0


def usable_environment(python: Path, root: Path, log) -> bool:
    if not python.is_file():
        return False
    code = ("import sys; raise SystemExit(not (sys.version_info >= (3, 11) "
            "and sys.prefix != sys.base_prefix))")
    try:
        return run_logged([str(python), "-c", code], root, log)
    except OSError as exc:
        print(f"Environment could not be run: {exc}", file=log)
        return False


def dependencies_ready(python: Path, requirements: Path, root: Path, log) -> bool:
    return run_logged([str(python), "-c", DEPENDENCY_CHECK, str(requirements)], root, log)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Start CIRRUS++ in its local virtual environment.")
    parser.add_argument("--check", action="store_true", help="Check setup without installing or starting the app.")
    parser.add_argument("--headless", action="store_true", help="Start without automatically opening a browser.")
    parser.add_argument("--port", type=int, default=8501, help="Local Streamlit port (default: 8501).")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if sys.version_info < MIN_PYTHON:
        print("CIRRUS++ requires Python 3.11 or newer.", file=sys.stderr)
        return 1
    root = PROJECT_ROOT
    project_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    log_directory = Path(tempfile.gettempdir()) / f"cirruspp-{project_id}"
    log_path = log_directory / "cirruspp_startup.log"
    try:
        # A cloud-synced project folder can block while closing a written log.
        # Keep startup diagnostics on the local temporary filesystem instead.
        log_directory.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            print(f"CIRRUS++ startup {datetime.now().astimezone().isoformat()}", file=log)
            requirements = root / "requirements.txt"
            if not (root / "app.py").is_file() or not requirements.is_file():
                raise RuntimeError("app.py or requirements.txt is missing. Keep the launcher in the project folder.")
            python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not usable_environment(python, root, log):
                if args.check:
                    raise RuntimeError("No usable .venv was found. Start without --check to create one.")
                print("[1/3] Creating or repairing the local Python environment…", flush=True)
                bootstrap = getattr(sys, "_base_executable", sys.executable)
                if not run_logged([bootstrap, "-m", "venv", str(root / ".venv")], root, log):
                    raise RuntimeError("Could not create .venv. Check the startup log for the Python error.")
                if not usable_environment(python, root, log):
                    raise RuntimeError("The local environment is not usable. Move the old .venv aside and try again.")
            else:
                print("[1/3] Local Python environment ready.", flush=True)
            if not dependencies_ready(python, requirements, root, log):
                if args.check:
                    raise RuntimeError("Dependencies are missing or do not match requirements.txt. Start normally to install them.")
                print("[2/3] Installing dependencies; the first start can take a few minutes…", flush=True)
                if not run_logged([str(python), "-m", "ensurepip", "--upgrade"], root, log):
                    raise RuntimeError("Could not prepare pip in the local environment.")
                if not run_logged([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                                   "-r", str(requirements)], root, log):
                    raise RuntimeError("Dependency installation failed. Check your internet connection and the startup log.")
                if not dependencies_ready(python, requirements, root, log):
                    raise RuntimeError("Dependencies still cannot be loaded. See the startup log.")
            else:
                print("[2/3] Dependencies match requirements.txt.", flush=True)
            if args.check:
                print("Setup check passed. No installation or app startup was performed.", flush=True)
                return 0
            command = [str(python), "-m", "streamlit", "run", str(root / "app.py"),
                       "--server.address", "127.0.0.1", "--server.port", str(args.port),
                       "--server.headless", "true" if args.headless else "false"]
            print(f"[3/3] Starting CIRRUS++ at http://localhost:{args.port}", flush=True)
            print("Keep this terminal open. Press Ctrl+C to stop CIRRUS++.", flush=True)
            print(f"Startup log: {log_path}", flush=True)
            print(f"Starting Streamlit on port {args.port}", file=log)
        return subprocess.call(command, cwd=root)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"\nCIRRUS++ could not start: {exc}\nStartup log: {log_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
