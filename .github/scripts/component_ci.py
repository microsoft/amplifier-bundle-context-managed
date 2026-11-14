"""Run component tests, not the separate Core/Foundation media pairing scripts."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ["-c", "modules/context-managed/pyproject.toml"]
CORRECTIONS = (
    "test_summary_request_release",
    "test_preliminary_request_release",
)
# Separate processes avoid the two modules' tests.conftest namespace collision.
SUITES = {
    "corrections": [
        *CONFIG,
        *(f"modules/context-managed/tests/{name}.py" for name in CORRECTIONS),
    ],
    "managed": [*CONFIG, "modules/context-managed/tests"],
    "transcript": [
        "-c", "modules/tool-transcript/pyproject.toml",
        "modules/tool-transcript/tests",
    ],
    "bundle": ["--import-mode=importlib", "--asyncio-mode=strict", "tests"],
}


def junit_result(path, *, corrections=False):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    counts = {
        "tests": len(cases),
        **{key: sum(len(case.findall(tag)) for case in cases)
           for key, tag in (("failures", "failure"), ("errors", "error"),
                            ("skipped", "skipped"))},
    }
    if not cases:
        raise ValueError("No test cases collected")
    if counts["failures"] or counts["errors"]:
        raise ValueError(f"Unsuccessful JUnit: {counts}")
    if corrections:
        if counts["skipped"]:
            raise ValueError("Correction tests must not skip or xfail")
        for name in CORRECTIONS:
            if not any(case.get("classname", "").split(".")[-1] == name
                       for case in cases):
                raise ValueError(f"Missing correction module: {name}")
    return counts


def main():
    os.chdir(ROOT)
    evidence = ROOT / "component-evidence"
    # Fail rather than combine fresh results with stale reports.
    evidence.mkdir(exist_ok=True)
    if any(evidence.glob("*.xml")) or (evidence / "results.json").exists():
        raise RuntimeError("Use a fresh component-evidence directory")
    metadata = {
        "scope": "component only; no Foundation/F1 pairing or installed media graph",
        "python": sys.version,
        "platform": platform.platform(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "github": {key: os.environ.get(key) for key in (
            "GITHUB_EVENT_NAME", "GITHUB_SHA", "GITHUB_REF",
            "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")},
        "distributions": {
            dist.metadata["Name"]: {
                "version": dist.version,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "null"),
            }
            for dist in importlib.metadata.distributions()
        },
    }
    (evidence / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if importlib.metadata.version("amplifier-core") != "2.0.2":
        raise RuntimeError("This CI profile requires amplifier-core==2.0.2")
    results = {}
    for name, selectors in SUITES.items():
        report = evidence / f"{name}.xml"
        command = [sys.executable, "-m", "pytest", *selectors, "-ra",
                   f"--junitxml={report}"]
        print(f"\n=== {name}: {command!r} ===", flush=True)
        with (evidence / f"{name}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
        result = {"command": command, "returncode": returncode}
        try:
            result["junit"] = junit_result(report, corrections=name == "corrections")
        except (OSError, ET.ParseError, ValueError) as error:
            result["report_error"] = str(error)
        results[name] = result
    (evidence / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    # Corrections are repeated within managed: do not sum them as unique cases.
    summary = json.dumps(results, indent=2)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write("## Component tests only\n\n"
                         "Corrections are a subset of managed, not extra unique cases. "
                         "No scripts/test_* F1 pairing or installed media qualification.\n\n"
                         f"```json\n{summary}\n```\n")
    hashes = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
              for path in sorted(evidence.iterdir()) if path.is_file()]
    (evidence / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    return int(any(item["returncode"] or "report_error" in item
                   for item in results.values()))


if __name__ == "__main__":
    sys.exit(main())