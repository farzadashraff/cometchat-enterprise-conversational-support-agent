"""Human-readable console output and a machine-readable JSON artifact
for one `EvalReport` (Phase 7 brief §20-21).

Neither format ever includes API keys, raw order records, full prompts,
or full conversation transcripts — only case ids, categories, pass/fail,
and short assertion detail strings that are themselves designed (see
`assertions.py`) to never echo back matched sensitive substrings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aster_row_agent.evaluation.runner import EvalReport


def render_console(report: EvalReport) -> str:
    lines = [f"ASTER & ROW EVALUATION — {report.run_label}", ""]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"[{status}] {result.case_id} ({result.category})")
        if result.error:
            lines.append(f"    ERROR: {result.error}")
        for assertion in result.assertions:
            if not assertion.passed:
                lines.append(f"    FAILED {assertion.name}: {assertion.detail}")

    lines.append("")
    lines.append(
        f"Summary: Passed: {report.passed_count}  Failed: {report.failed_count}  "
        f"Total: {report.total}"
    )
    lines.append("")
    lines.append("Category summary:")
    header = f"{'CATEGORY':<28}{'PASS':>6}{'FAIL':>6}{'TOTAL':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for category, (passed, total) in report.category_breakdown().items():
        lines.append(f"{category:<28}{passed:>6}{total - passed:>6}{total:>7}")
    lines.append("-" * len(header))
    lines.append(
        f"{'OVERALL':<28}{report.passed_count:>6}{report.failed_count:>6}{report.total:>7}"
    )
    return "\n".join(lines)


def to_json_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "run_label": report.run_label,
        "generated_at": datetime.now(UTC).isoformat(),
        "total": report.total,
        "passed": report.passed_count,
        "failed": report.failed_count,
        "category_breakdown": {
            category: {"passed": passed, "total": total}
            for category, (passed, total) in report.category_breakdown().items()
        },
        "cases": [
            {
                "id": result.case_id,
                "category": result.category,
                "passed": result.passed,
                "error": result.error,
                "assertions": [
                    {"name": a.name, "passed": a.passed, "detail": a.detail}
                    for a in result.assertions
                ],
            }
            for result in report.results
        ],
    }
