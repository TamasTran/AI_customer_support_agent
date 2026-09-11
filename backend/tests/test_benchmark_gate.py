"""No DB, no Ollama — pure logic test for the benchmark's --min-pass-rate CI gate."""
from scripts.benchmark_agent import print_report


def _report(passed_flags: list[bool]) -> dict:
    return {
        "model": "test-model",
        "run_at": "2026-01-01T00:00:00Z",
        "results": [
            {
                "id": f"case-{i}",
                "category": "test",
                "passed": passed,
                "detail": "",
                "note": "",
                "latency_s": 1.0,
                "tokens_per_sec": None,
                "llm_calls": 1,
                "reply_preview": "",
            }
            for i, passed in enumerate(passed_flags)
        ],
    }


def test_print_report_returns_100_when_all_pass(capsys):
    rate = print_report(_report([True, True, True]))
    capsys.readouterr()
    assert rate == 100.0


def test_print_report_returns_0_when_all_fail(capsys):
    rate = print_report(_report([False, False]))
    capsys.readouterr()
    assert rate == 0.0


def test_print_report_returns_correct_fraction(capsys):
    rate = print_report(_report([True, True, True, False]))
    capsys.readouterr()
    assert rate == 75.0


def test_print_report_handles_empty_results_without_crashing(capsys):
    rate = print_report(_report([]))
    capsys.readouterr()
    assert rate == 0.0
