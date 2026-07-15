"""scripts/prior_art_check.py (methodology section 16)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prior_art_check as pac  # noqa: E402


def test_search_strings_match_methodology_section_16_exactly():
    assert pac.SEARCH_STRINGS == [
        "neighbourhood conditioned delta Mamba",
        "local convolution modulate Mamba step size",
        "convolution conditioned selective parameters SSM",
        "spatially varying step-size state space model",
        "depthwise conv modulate dt visual Mamba",
        "scan dynamics conditioned convolution segmentation",
    ]


def test_is_suspicious_hit_true_when_all_three_term_groups_present():
    title = "Local Depthwise Convolution Modulates Mamba Step-Size"
    snippet = "We propose conditioning the Delta parameter of a state space model on local structure."
    assert pac.is_suspicious_hit(title, snippet) is True


def test_is_suspicious_hit_false_when_missing_a_term_group():
    # conv + ssm terms present, but no delta/step-size/dt term
    title = "A Depthwise Convolutional Mamba Block for Segmentation"
    snippet = "We add a convolution branch before the Mamba scan."
    assert pac.is_suspicious_hit(title, snippet) is False


def test_is_suspicious_hit_false_for_unrelated_text():
    assert pac.is_suspicious_hit("A Survey of Transformers", "Attention is all you need.") is False


def test_try_import_ddgs_returns_none_when_not_installed():
    # duckduckgo-search is NOT a project dependency (kept minimal, see module
    # docstring) -- on this dev machine (and any fresh install) it must be
    # absent, so the optional path degrades gracefully.
    assert pac._try_import_ddgs() is None


def test_check_one_string_captures_query_exception_without_raising():
    class _FailingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            raise RuntimeError("network down")

    outcome = pac.check_one_string(_FailingDDGS, "some query", 8)
    assert outcome["error"] is not None
    assert "network down" in outcome["error"]
    assert outcome["results"] == []
    assert outcome["flagged"] == []


def test_check_one_string_flags_suspicious_results_only():
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [
                {"title": "Unrelated paper", "href": "http://a", "body": "nothing relevant here"},
                {"title": "Local conv modulates Mamba delta", "href": "http://b", "body": "a state space model with depthwise convolution conditioning step-size"},
            ]

    outcome = pac.check_one_string(_FakeDDGS, "q", 8)
    assert outcome["error"] is None
    assert len(outcome["results"]) == 2
    assert len(outcome["flagged"]) == 1
    assert outcome["flagged"][0]["href"] == "http://b"


def test_print_report_no_query_returns_empty_and_does_not_crash(capsys):
    result = pac.print_report(do_query=False)
    assert result == []
    out = capsys.readouterr().out
    for q in pac.SEARCH_STRINGS:
        assert q in out


def test_print_report_with_query_but_no_ddgs_installed_returns_empty(capsys):
    result = pac.print_report(do_query=True)
    assert result == []
    out = capsys.readouterr().out
    assert "not installed" in out


def test_print_report_prints_pivot_warning_when_flagged(monkeypatch, capsys):
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [{"title": "Local conv modulates Mamba delta", "href": "http://flag", "body": "depthwise convolution conditioning the state space model step-size"}]

    monkeypatch.setattr(pac, "_try_import_ddgs", lambda: _FakeDDGS)

    result = pac.print_report(do_query=True)
    out = capsys.readouterr().out
    assert "PIVOT-BEFORE-SUBMISSION WARNING" in out
    assert "http://flag" in out
    assert any(o["flagged"] for o in result)


def test_print_report_no_pivot_warning_when_nothing_flagged(monkeypatch, capsys):
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [{"title": "Unrelated paper", "href": "http://a", "body": "nothing relevant"}]

    monkeypatch.setattr(pac, "_try_import_ddgs", lambda: _FakeDDGS)

    result = pac.print_report(do_query=True)
    out = capsys.readouterr().out
    assert "PIVOT-BEFORE-SUBMISSION WARNING" not in out
    assert "No flagged hits" in out
    assert not any(o["flagged"] for o in result)
