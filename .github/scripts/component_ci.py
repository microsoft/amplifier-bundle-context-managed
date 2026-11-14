"""Run component tests, not the separate Core/Foundation media pairing scripts."""

import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = Path(__file__).with_name("correction-nodeids.txt")
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


class OutcomeRecorder:
    """Public pytest hooks retain collection and outcomes, including non-strict XPASS."""

    def __init__(self, path):
        self.path = path
        self.collected = []
        self.reports = []
        self.collection_errors = []

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        self.reports.append({
            "nodeid": report.nodeid, "when": report.when,
            "outcome": report.outcome, "wasxfail": hasattr(report, "wasxfail"),
        })

    def pytest_sessionfinish(self, session, exitstatus):
        self.path.write_text(json.dumps({
            "collected": self.collected, "reports": self.reports,
            "collection_errors": self.collection_errors, "exitstatus": int(exitstatus),
        }, indent=2) + "\n")


def junit_identity(nodeid):
    # Match pytest's path/class/function encoding, preserving parameter text.
    address, bracket, parameters = nodeid.partition("[")
    parts = address.split("::")
    classname = ".".join([parts[0].removesuffix(".py").replace("/", "."), *parts[1:-1]])
    return classname, parts[-1] + bracket + parameters


def outcome_result(path, *, corrections=False):
    data = json.loads(path.read_text())
    collected, reports = data["collected"], data["reports"]
    if data["exitstatus"] or data["collection_errors"]:
        raise ValueError("Unsuccessful pytest session/collection")
    if not collected or len(set(collected)) != len(collected):
        raise ValueError("Empty or duplicate collected node IDs")
    if corrections:
        expected = INVENTORY.read_text().splitlines()
        if len(set(expected)) != len(expected) or Counter(collected) != Counter(expected):
            raise ValueError("Correction collected node IDs differ from reviewed inventory")
    grouped = {nodeid: {} for nodeid in collected}
    for report in reports:
        nodeid, phase = report["nodeid"], report["when"]
        if nodeid not in grouped or phase not in {"setup", "call", "teardown"}:
            raise ValueError("Extra/renamed outcome node ID or unknown phase")
        if phase in grouped[nodeid]:
            raise ValueError("Duplicate outcome node ID/phase")
        outcome, wasxfail = report["outcome"], report["wasxfail"]
        if outcome not in {"passed", "failed", "skipped"} or type(wasxfail) is not bool:
            raise ValueError("Invalid typed outcome")
        if outcome == "failed" or (outcome == "passed" and wasxfail):
            raise ValueError("Failed or XPASS outcome (including strict=False)")
        if corrections and (outcome != "passed" or wasxfail):
            raise ValueError("Correction tests must not skip or xfail")
        grouped[nodeid][phase] = report
    for phases in grouped.values():
        expected_phases = {"setup", "teardown"}
        if phases.get("setup", {}).get("outcome") == "passed":
            expected_phases.add("call")
        if set(phases) != expected_phases:
            raise ValueError("Missing call/setup/teardown outcome")
    return data


def junit_result(path, outcomes, *, corrections=False):
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
    expected = Counter(junit_identity(nodeid) for nodeid in outcomes["collected"])
    actual = Counter((case.get("classname"), case.get("name")) for case in cases)
    if expected != actual or any(count != 1 for count in actual.values()):
        raise ValueError("Missing/extra/duplicate/renamed JUnit test identities")
    skipped = sum(any(report["nodeid"] == nodeid and report["outcome"] == "skipped"
                      for report in outcomes["reports"])
                  for nodeid in outcomes["collected"])
    if counts["skipped"] != skipped:
        raise ValueError("JUnit skips disagree with typed outcomes")
    if corrections and counts["skipped"]:
        raise ValueError("Correction tests must not skip or xfail")
    return counts


def main():
    os.chdir(ROOT)
    evidence = ROOT / "component-evidence"
    # Fail rather than combine fresh results with stale reports.
    evidence.mkdir(exist_ok=True)
    if (any(evidence.glob("*.xml")) or any(evidence.glob("*-outcomes.json"))
            or (evidence / "results.json").exists()):
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
        outcomes_path = evidence / f"{name}-outcomes.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--pytest",
                   str(outcomes_path), *selectors, "-ra",
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
            outcomes = outcome_result(outcomes_path, corrections=name == "corrections")
            result["junit"] = junit_result(
                report, outcomes, corrections=name == "corrections")
        except (OSError, ET.ParseError, ValueError, KeyError, TypeError) as error:
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
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        import pytest

        sys.exit(pytest.main(sys.argv[3:], plugins=[OutcomeRecorder(Path(sys.argv[2]))]))
    else:
        sys.exit(main())