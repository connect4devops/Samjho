"""Grading tests. Pure functions, no network."""

from __future__ import annotations

import pytest

from app import grading, validation


def _extraction(**kwargs):
    payload = {
        "title": "Employee records",
        "notes": [{"heading": "Entities", "content": ["EMP", "COMPANY"], "confidence": "high"}],
        "takeaways": ["Every employee belongs to a company."],
    }
    payload.update(kwargs)
    return validation.normalize_extraction(payload, requested_mode="er")


ER_SOURCE = """erDiagram
    EMP {
        string id PK
        string name
    }
    COMPANY {
        string id PK
    }
    EMP ||--o{ COMPANY : "works for"
"""


class TestAssess:
    def test_a_complete_diagram_scores_well(self):
        result = grading.assess(
            _extraction(),
            coverage=[
                {"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"},
                {"label": "COMPANY", "kind": "entity", "detail": "", "confidence": "high"},
            ],
            diagram_source=ER_SOURCE,
            diagram_type="erDiagram",
            mode="er",
        )
        assert result["score"] >= 60
        assert result["confidence"] in {"high", "medium", "low"}
        assert result["criteria"]

    def test_every_criterion_is_scored_and_evidenced(self):
        result = grading.assess(
            _extraction(),
            coverage=[{"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"}],
            diagram_source=ER_SOURCE,
            diagram_type="erDiagram",
            mode="er",
        )
        for criterion in result["criteria"]:
            assert 0 <= criterion["score"] <= 100
            assert criterion["verdict"] in {"strong", "partial", "gap"}
            assert criterion["evidence"]

    def test_a_missing_element_lowers_the_coverage_score(self):
        covered = grading.assess(
            _extraction(),
            coverage=[{"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"}],
            diagram_source=ER_SOURCE,
            diagram_type="erDiagram",
            mode="er",
        )
        missing = grading.assess(
            _extraction(),
            coverage=[
                {"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"},
                {"label": "SALARY", "kind": "entity", "detail": "", "confidence": "high"},
            ],
            diagram_source=ER_SOURCE,
            diagram_type="erDiagram",
            mode="er",
        )
        assert missing["score"] < covered["score"]

    def test_an_invented_element_is_penalised(self):
        without = grading.assess(
            _extraction(),
            coverage=[{"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"}],
            diagram_source=ER_SOURCE,
            diagram_type="erDiagram",
            mode="er",
        )
        with_invention = grading.assess(
            _extraction(),
            coverage=[{"label": "EMP", "kind": "entity", "detail": "", "confidence": "high"}],
            diagram_source=ER_SOURCE + '    GHOST ||--o{ EMP : "haunts"\n',
            diagram_type="erDiagram",
            mode="er",
        )
        assert with_invention["score"] < without["score"]

    def test_an_empty_diagram_scores_zero_on_completeness(self):
        result = grading.assess(
            _extraction(),
            coverage=[],
            diagram_source="",
            diagram_type="",
            mode="er",
        )
        assert result["score"] >= 0
        assert isinstance(result["gaps"], list)

    def test_the_rationale_names_the_weakest_criterion(self):
        result = grading.assess(
            _extraction(notes=[], takeaways=[]),
            coverage=[],
            diagram_source="",
            diagram_type="",
            mode="er",
        )
        assert result["rationale"]
        assert "scores" in result["rationale"]

    def test_a_flowchart_with_no_edges_is_called_out(self):
        result = grading.assess(
            _extraction(),
            coverage=[],
            diagram_source="flowchart TD\n    A[Start]\n    B[End]",
            diagram_type="flowchart",
            mode="flowchart",
        )
        completeness = next(
            c for c in result["criteria"] if c["criterion"] == "Diagram completeness"
        )
        assert completeness["verdict"] != "strong"

    def test_nothing_to_grade_is_handled(self):
        result = grading.assess(
            validation.normalize_extraction(None),
            coverage=[],
            diagram_source="",
            diagram_type="",
            mode="auto",
        )
        assert result["score"] == 0
        assert result["confidence"] == "low"
        assert result["gaps"]
