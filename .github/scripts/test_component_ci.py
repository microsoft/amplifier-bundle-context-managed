"""Fail-closed CI harness controls, not component or media-graph qualification."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import component_ci as ci


def typed_outcomes(nodeids):
    return {
        "collected": list(nodeids), "exitstatus": 0, "collection_errors": [],
        "reports": [
            {"nodeid": nodeid, "when": phase, "outcome": "passed", "wasxfail": False}
            for nodeid in nodeids for phase in ("setup", "call", "teardown")
        ],
    }


def junit_body(nodeids):
    root = ET.Element("testsuites")
    for nodeid in nodeids:
        classname, name = ci.junit_identity(nodeid)
        ET.SubElement(root, "testcase", classname=classname, name=name)
    return root


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.ids = ci.INVENTORY.read_text().splitlines()
        self.data = typed_outcomes(self.ids)

    def check_outcomes(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.json"
            path.write_text(json.dumps(data))
            return ci.outcome_result(path, **kwargs)

    def check_junit(self, body, data=None, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xml"
            ET.ElementTree(body).write(path)
            return ci.junit_result(path, data or self.data, **kwargs)

    def test_exact_inventory_and_junit_pass(self):
        self.assertEqual(len(self.ids), 56)
        self.assertEqual(sum("test_summary_request_release.py::" in n for n in self.ids), 33)
        self.assertEqual(sum("test_preliminary_request_release.py::" in n for n in self.ids), 23)
        self.check_outcomes(self.data, corrections=True)
        self.assertEqual(
            self.check_junit(junit_body(self.ids), corrections=True),
            {"tests": 56, "failures": 0, "errors": 0, "skipped": 0},
        )

    def test_two_cases_one_per_module_is_not_sufficient(self):
        ids = [self.ids[0], next(n for n in self.ids if "test_preliminary" in n)]
        with self.assertRaisesRegex(ValueError, "reviewed inventory"):
            self.check_outcomes(typed_outcomes(ids), corrections=True)
        with self.assertRaisesRegex(ValueError, "JUnit test identities"):
            self.check_junit(junit_body(ids), corrections=True)

    def test_collected_missing_extra_duplicate_and_renamed_fail(self):
        variants = [
            self.ids[:-1], self.ids + ["tests/test_extra.py::test_extra"],
            self.ids[:-1] + [self.ids[0]], self.ids[:-1] + [self.ids[-1] + "_renamed"],
        ]
        for ids in variants:
            with self.subTest(ids=ids[-1]), self.assertRaises(ValueError):
                self.check_outcomes(typed_outcomes(ids), corrections=True)

    def test_junit_missing_extra_duplicate_renamed_and_unrelated_fail(self):
        variants = [
            self.ids[:-1], self.ids + ["tests/test_extra.py::test_extra"],
            self.ids[:-1] + [self.ids[0]], self.ids[:-1] + [self.ids[-1] + "_renamed"],
            ["tests/test_summary_request_release.py::test_unrelated"] * 33
            + ["tests/test_preliminary_request_release.py::test_unrelated"] * 23,
        ]
        for ids in variants:
            with self.subTest(ids=ids[-1]), self.assertRaisesRegex(ValueError, "JUnit test identities"):
                self.check_junit(junit_body(ids), corrections=True)

    def test_missing_extra_duplicate_and_renamed_outcomes_fail(self):
        variants = []
        for mode in ("missing", "extra", "duplicate", "renamed"):
            data = copy.deepcopy(self.data)
            if mode == "missing":
                data["reports"].pop(1)
            elif mode == "extra":
                data["reports"].extend(typed_outcomes(["test_extra.py::test_extra"])["reports"])
            elif mode == "duplicate":
                data["reports"].append(data["reports"][0])
            else:
                data["reports"][1]["nodeid"] += "_renamed"
            variants.append((mode, data))
        for mode, data in variants:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.check_outcomes(data, corrections=True)

    def test_skip_xfail_xpass_and_failure_are_not_correction_passes(self):
        for outcome, wasxfail in [
            ("skipped", False), ("skipped", True), ("passed", True), ("failed", False),
        ]:
            data = copy.deepcopy(self.data)
            data["reports"][1].update(outcome=outcome, wasxfail=wasxfail)
            with self.subTest(outcome=outcome, xfail=wasxfail), self.assertRaises(ValueError):
                self.check_outcomes(data, corrections=True)

    def test_collection_and_nonzero_exit_fail(self):
        for key, value in [("collection_errors", ["bad.py"]), ("exitstatus", 1)]:
            data = copy.deepcopy(self.data)
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_outcomes(data)

    def test_junit_failure_error_and_empty_fail(self):
        for tag in ("failure", "error"):
            body = junit_body(self.ids)
            ET.SubElement(body[0], tag)
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Unsuccessful"):
                self.check_junit(body)
        with self.assertRaisesRegex(ValueError, "No test cases"):
            self.check_junit(ET.Element("testsuites"))

    def test_broader_skips_are_counted_and_cross_checked(self):
        data = typed_outcomes(["test_example.py::test_example"])
        data["reports"][1].update(outcome="skipped", wasxfail=True)
        self.check_outcomes(data)
        body = junit_body(data["collected"])
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            self.check_junit(body, data)
        ET.SubElement(body[0], "skipped")
        self.assertEqual(self.check_junit(body, data)["skipped"], 1)
        data["reports"][1].update(outcome="passed")
        with self.assertRaisesRegex(ValueError, "XPASS"):
            self.check_outcomes(data)

    def test_count_preserving_skip_swap_is_rejected_per_identity(self):
        ids = ["test_required.py::test_required", "test_optional.py::test_optional"]
        data = typed_outcomes(ids)
        data["reports"][4]["outcome"] = "skipped"
        self.check_outcomes(data)  # A valid broader-suite optional call skip.
        body = junit_body(ids)
        ET.SubElement(body[0], "skipped")  # Same totals, wrong test skipped.
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            self.check_junit(body, data)

    def test_consistent_optional_skips_join_by_identity_not_order(self):
        ids = ["test_required.py::test_required", "test_optional.py::test_optional"]
        for phase in ("setup", "call", "teardown"):
            for xfail in (False, True):
                data = typed_outcomes(ids)
                for report in data["reports"]:
                    if report["nodeid"] == ids[1] and report["when"] == phase:
                        report.update(outcome="skipped", wasxfail=xfail)
                if phase == "setup":
                    data["reports"] = [r for r in data["reports"]
                                       if not (r["nodeid"] == ids[1] and r["when"] == "call")]
                body = junit_body(list(reversed(ids)))
                ET.SubElement(body[0], "skipped")
                with self.subTest(phase=phase, xfail=xfail):
                    self.check_outcomes(data)
                    self.assertEqual(
                        self.check_junit(body, data),
                        {"tests": 2, "failures": 0, "errors": 0, "skipped": 1},
                    )

    def test_malformed_optional_phase_states_are_rejected_by_both_readers(self):
        variants = []
        data = typed_outcomes(["test_optional.py::test_optional"])
        for mode in ("duplicate", "missing", "setup_skip_with_call", "multiple_skips",
                     "unknown_phase", "unknown_outcome", "untyped_xfail"):
            changed = copy.deepcopy(data)
            if mode == "duplicate":
                changed["reports"].append(changed["reports"][1])
            elif mode == "missing":
                changed["reports"].pop()
            elif mode == "setup_skip_with_call":
                changed["reports"][0]["outcome"] = "skipped"
            elif mode == "multiple_skips":
                changed["reports"][1]["outcome"] = "skipped"
                changed["reports"][2]["outcome"] = "skipped"
            elif mode == "unknown_phase":
                changed["reports"][1]["when"] = "unknown"
            elif mode == "unknown_outcome":
                changed["reports"][1]["outcome"] = "unknown"
            else:
                changed["reports"][1]["wasxfail"] = "false"
            variants.append((mode, changed))
        for mode, changed in variants:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    self.check_outcomes(changed)
                with self.assertRaises(ValueError):
                    self.check_junit(junit_body(changed["collected"]), changed)

    def test_malformed_junit_statuses_are_rejected(self):
        ids = ["test_optional.py::test_optional"]
        data = typed_outcomes(ids)
        data["reports"][1]["outcome"] = "skipped"
        for tags in (("skipped", "skipped"), ("passed",), ("unknown",)):
            body = junit_body(ids)
            for tag in tags:
                ET.SubElement(body[0], tag)
            with self.subTest(tags=tags), self.assertRaisesRegex(ValueError, "Malformed JUnit"):
                self.check_junit(body, data)


class RealPytestHookTests(unittest.TestCase):
    """Actual pytest subprocess probes; fixtures are temporary synthetic tests."""

    def run_probe(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_probe.py").write_text(source)
            (root / "pytest.ini").write_text("[pytest]\nxfail_strict=true\n")
            outcomes = root / "outcomes.json"
            report = root / "report.xml"
            result = subprocess.run(
                [sys.executable, str(Path(ci.__file__).resolve()), "--pytest",
                 str(outcomes), "-c", str(root / "pytest.ini"), str(root / "test_probe.py"),
                 f"--junitxml={report}", "-q"],
                capture_output=True, text=True,
            )
            return result.returncode, json.loads(outcomes.read_text()), ET.parse(report).getroot()

    def validate_probe(self, data, xml=None, *, corrections=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.json"
            path.write_text(json.dumps(data))
            result = ci.outcome_result(path, corrections=corrections)
            if xml is not None:
                report = Path(directory) / "report.xml"
                ET.ElementTree(xml).write(report)
                ci.junit_result(report, result, corrections=corrections)
            return result

    def test_actual_strict_false_xpass_is_rejected_despite_plain_junit_pass(self):
        code, data, xml = self.run_probe(
            "import pytest\n@pytest.mark.xfail(strict=False)\ndef test_probe(): pass\n")
        self.assertEqual(code, 0)  # Global xfail_strict cannot override the marker.
        self.assertFalse(list(xml.find(".//testcase")))
        self.assertTrue(any(r["wasxfail"] and r["outcome"] == "passed" for r in data["reports"]))
        with self.assertRaisesRegex(ValueError, "XPASS"):
            self.validate_probe(data)

    def test_actual_xfail_is_reported_as_skip_and_rejected_for_corrections(self):
        code, data, xml = self.run_probe(
            "import pytest\n@pytest.mark.xfail(strict=False)\ndef test_probe(): assert False\n")
        self.assertEqual(code, 0)
        self.assertIsNotNone(xml.find(".//skipped"))
        self.validate_probe(data, xml)  # Broader expected failures stay explicit skips.
        # Use the real outcome for a reviewed ID to isolate the correction skip gate.
        first = ci.INVENTORY.read_text().splitlines()[0]
        full = typed_outcomes(ci.INVENTORY.read_text().splitlines())
        full["reports"][:3] = [{**r, "nodeid": first} for r in data["reports"]]
        with self.assertRaisesRegex(ValueError, "must not skip or xfail"):
            self.validate_probe(full, corrections=True)

    def test_actual_ordinary_pass_has_all_phases(self):
        code, data, xml = self.run_probe("def test_probe(): pass\n")
        self.assertEqual(code, 0)
        self.validate_probe(data, xml)
        self.assertEqual([r["when"] for r in data["reports"]], ["setup", "call", "teardown"])
        self.assertEqual(len(xml.findall(".//testcase")), 1)

    def test_actual_setup_call_and_teardown_skips_join_per_identity(self):
        code, data, xml = self.run_probe(
            "import pytest\n"
            "@pytest.fixture\n"
            "def setup_skip(): pytest.skip('optional setup')\n"
            "@pytest.fixture\n"
            "def teardown_skip():\n"
            "    yield\n"
            "    pytest.skip('optional teardown')\n"
            "def test_required(): pass\n"
            "def test_setup(setup_skip): pass\n"
            "def test_call(): pytest.skip('optional call')\n"
            "def test_teardown(teardown_skip): pass\n"
        )
        self.assertEqual(code, 0)
        self.validate_probe(data, xml)
        self.assertEqual(list(ci.outcome_statuses(data).values()).count("skipped"), 3)
        self.assertEqual(len(xml.findall(".//skipped")), 3)
        # Keep the same count and identities but move one skip to the required pass.
        cases = xml.findall(".//testcase")
        skipped = next(case for case in cases if case.find("skipped") is not None)
        marker = skipped.find("skipped")
        skipped.remove(marker)
        next(case for case in cases if case.get("name") == "test_required").append(marker)
        with self.assertRaisesRegex(ValueError, "status disagrees"):
            self.validate_probe(data, xml)


if __name__ == "__main__":
    unittest.main()