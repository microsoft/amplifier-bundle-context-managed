"""Fail-closed checks for the CI JUnit verifier; no runtime fixtures changed."""

from pathlib import Path
import tempfile
import unittest

from component_ci import junit_result


class JUnitEvidenceTests(unittest.TestCase):
    def check_report(self, body, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xml"
            path.write_text(body)
            return junit_result(path, **kwargs)

    def test_correction_modules_are_required(self):
        with self.assertRaisesRegex(ValueError, "Missing correction module"):
            self.check_report(
                '<testsuites><testcase classname="other"/></testsuites>',
                corrections=True,
            )

    def test_correction_skips_and_xfails_fail(self):
        with self.assertRaisesRegex(ValueError, "must not skip"):
            self.check_report(
                '<testsuites><testcase><skipped type="pytest.xfail"/>'
                '</testcase></testsuites>', corrections=True,
            )

    def test_empty_report_fails(self):
        with self.assertRaisesRegex(ValueError, "No test cases"):
            self.check_report("<testsuites/>")

    def test_failures_and_errors_fail(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Unsuccessful"):
                self.check_report(f"<testsuites><testcase><{tag}/></testcase></testsuites>")

    def test_broader_skips_are_reported_not_hidden(self):
        counts = self.check_report(
            "<testsuites><testcase/><testcase><skipped/></testcase></testsuites>")
        self.assertEqual(counts, {"tests": 2, "failures": 0, "errors": 0, "skipped": 1})

    def test_both_correction_modules_pass_without_skips(self):
        counts = self.check_report(
            '<testsuites><testcase classname="tests.test_summary_request_release"/>'
            '<testcase classname="tests.test_preliminary_request_release"/>'
            '</testsuites>', corrections=True,
        )
        self.assertEqual(counts, {"tests": 2, "failures": 0, "errors": 0, "skipped": 0})


if __name__ == "__main__":
    unittest.main()