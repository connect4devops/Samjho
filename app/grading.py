"""Objective grading of an extraction.

A model's own confidence is not a grade. The failure this app exists to
catch is a model that is confidently wrong, so every criterion here is
computed from artifacts the app actually holds: the diagram that passed
validation, the source inventory the model reported seeing, the notes that
were extracted, and the changes the app had to make. The model cannot talk
itself up, because it does not get to write the score.

Each criterion answers a different question, so a low score says *what* is
weak rather than just "weak":

* ``Source coverage``      -- did the reconstruction include every element of
  the drawing?
* ``Diagram completeness`` -- does the diagram carry the detail, or is it a
  set of empty boxes?
* ``Structure captured``   -- did the mode produce the artifacts it owes?
* ``Relationship integrity`` -- is everything connected the way it should be?
* ``Faithfulness``         -- does the diagram contain anything that was not
  in the drawing? (Invention is penalised; flagging an ambiguity is not.)
"""

from __future__ import annotations

from typing import Any

from . import diagram as diagram_mod

# What each mode owes the reader. A mode that produces none of these has
# not done its job, whatever the model says in prose.
_MODE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "notes": ("notes",),
    "flowchart": ("diagram", "notes"),
    "mindmap": ("diagram",),
    "architecture": ("diagram", "notes"),
    "actions": ("action_items",),
    "concept": ("diagram", "notes"),
    "er": ("diagram", "sql_schema"),
    "auto": ("notes", "diagram"),
}

_CONFIDENCE_HIGH = 80
_CONFIDENCE_MEDIUM = 55


def _verdict(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 55:
        return "partial"
    return "gap"


def _criterion(name: str, score: float | None, evidence: str) -> dict[str, Any] | None:
    if score is None:
        return None
    score = max(0.0, min(100.0, float(score)))
    return {
        "criterion": name,
        "score": int(round(score)),
        "verdict": _verdict(score),
        "evidence": evidence,
    }


def _extraction_text(extraction: Any) -> str:
    """Every word the extraction contains, as one searchable blob."""
    parts: list[str] = []
    for note in getattr(extraction, "notes", []) or []:
        parts.append(str(note.get("heading", "")))
        parts.extend(str(line) for line in note.get("content", []) or [])
    parts.extend(str(item) for item in getattr(extraction, "takeaways", []) or [])
    parts.extend(str(item.get("task", "")) for item in getattr(extraction, "action_items", []) or [])
    parts.append(str(getattr(extraction, "summary", "")))
    parts.append(str(getattr(extraction, "sql_schema", "") or ""))
    return "\n".join(parts).lower()


def _artifact_present(extraction: Any, name: str) -> bool:
    if name == "notes":
        return bool(getattr(extraction, "notes", []))
    if name == "takeaways":
        return bool(getattr(extraction, "takeaways", []))
    if name == "action_items":
        return bool(getattr(extraction, "action_items", []))
    if name == "diagram":
        diagram = getattr(extraction, "diagram", None)
        return bool(diagram and diagram.get("mermaid"))
    if name == "sql_schema":
        return bool(getattr(extraction, "sql_schema", None))
    return False


def _source_coverage(
    extraction: Any, coverage: list[dict[str, str]], diagram_source: str
) -> tuple[float | None, str]:
    """How much of what the model saw actually made it into the answer."""
    if not coverage:
        return None, "The model did not report an inventory of the drawing."
    text = _extraction_text(extraction)
    diagram_low = (diagram_source or "").lower()
    found = 0
    missing: list[str] = []
    for item in coverage:
        label = str(item.get("label", "")).strip().lower()
        if not label:
            continue
        if label in text or (diagram_low and label in diagram_low):
            found += 1
        else:
            missing.append(item.get("label", ""))
    score = 100.0 * found / len(coverage)
    evidence = f"{found} of {len(coverage)} reported elements appear in the answer."
    if missing:
        evidence += " Missing: " + ", ".join(missing[:5]) + "."
    return score, evidence


def _diagram_completeness(
    diagram_type: str, diagram_source: str
) -> tuple[float | None, str]:
    """Does the diagram carry detail, or is it a row of empty boxes?"""
    if not diagram_source:
        return None, "There is no diagram to check."

    if diagram_type == "erDiagram":
        parsed = diagram_mod.parse_er(diagram_source)
        entities = parsed["entities"]
        relationships = parsed["relationships"]
        if not entities:
            return 0.0, "The diagram declares no entities."
        with_attributes = [e for e in entities if e["attributes"]]
        labeled = [r for r in relationships if r["label"]]
        entity_share = 100.0 * len(with_attributes) / len(entities)
        relation_share = (
            100.0 * len(labeled) / len(relationships) if relationships else 100.0
        )
        score = 0.6 * entity_share + 0.4 * relation_share
        evidence = (
            f"{len(with_attributes)} of {len(entities)} entities carry attributes and "
            f"{len(labeled)} of {len(relationships)} relationships are labelled."
        )
        return score, evidence

    labels = diagram_mod.diagram_labels(diagram_source)
    if not labels:
        return 0.0, "The diagram has no labelled elements."
    edges = sum(
        1
        for line in diagram_source.splitlines()
        if "-->" in line or "---" in line or "-.->" in line or "==>" in line
    )
    # A diagram with labels but no edges is a list, not a flow.
    if edges == 0:
        return 40.0, "The diagram has labelled boxes but no connections between them."
    score = min(100.0, 60.0 + 40.0 * min(edges, 8) / 8.0)
    return score, f"{len(labels)} labelled elements and {edges} connections."


def _structure_captured(extraction: Any, mode: str) -> tuple[float | None, str]:
    """Did the mode produce the artifacts it owes?"""
    target = mode if mode in _MODE_ARTIFACTS else "auto"
    expected = _MODE_ARTIFACTS[target]
    present = [name for name in expected if _artifact_present(extraction, name)]
    score = 100.0 * len(present) / len(expected)
    missing = [name for name in expected if name not in present]
    evidence = f"{len(present)} of {len(expected)} expected parts are present."
    if missing:
        evidence += " Missing: " + ", ".join(missing) + "."
    return score, evidence


def _relationship_integrity(
    diagram_type: str, diagram_source: str
) -> tuple[float | None, str]:
    """Is everything in the diagram connected to something?"""
    if not diagram_source:
        return None, "There is no diagram to check."

    if diagram_type == "erDiagram":
        parsed = diagram_mod.parse_er(diagram_source)
        entities = parsed["entities"]
        if len(entities) < 2:
            return None, "A single entity has no relationships to check."
        connected: set[str] = set()
        for relationship in parsed["relationships"]:
            connected.add(relationship["from"])
            connected.add(relationship["to"])
        score = 100.0 * len(connected) / len(entities)
        return score, (
            f"{len(connected)} of {len(entities)} entities take part in a relationship."
        )

    labels = diagram_mod.diagram_labels(diagram_source)
    if len(labels) < 2:
        return None, "A single element has no connections to check."
    linked = 0
    for line in diagram_source.splitlines():
        for label in labels:
            if label and label in line and ("-->" in line or "---" in line):
                linked += 1
                break
    score = min(100.0, 100.0 * linked / len(labels))
    return score, f"{linked} of {len(labels)} labelled elements sit on a connection."


def _faithfulness(
    extraction: Any, coverage: list[dict[str, str]], diagram_source: str
) -> tuple[float | None, str]:
    """Penalise content that was not in the drawing.

    Flagging an ambiguity is honest and is not penalised here. Silently
    adding an element that the drawing never had is.
    """
    if not coverage:
        return None, "Without an inventory there is nothing to compare against."
    if not diagram_source:
        return None, "There is no diagram to check."

    known = [str(item.get("label", "")).strip().lower() for item in coverage]
    known = [label for label in known if label]
    if not known:
        return None, "Without an inventory there is nothing to compare against."

    diagram_labels = [
        label.lower() for label in diagram_mod.diagram_elements(diagram_source)
    ]
    if not diagram_labels:
        return None, "The diagram has no labelled elements to compare."

    invented = [
        label
        for label in diagram_labels
        if not any(label in known_item or known_item in label for known_item in known)
    ]
    score = max(0.0, 100.0 - 25.0 * len(invented))
    evidence = "Nothing in the diagram is missing from the reported inventory."
    if invented:
        evidence = (
            f"{len(invented)} element(s) in the diagram are not in the reported "
            "inventory: " + ", ".join(invented[:5]) + "."
        )
    return score, evidence


def assess(
    extraction: Any,
    *,
    coverage: list[dict[str, str]] | None = None,
    diagram_source: str = "",
    diagram_type: str = "",
    mode: str = "auto",
) -> dict[str, Any]:
    """Grade an extraction against the drawing it came from.

    Returns a dict with an overall ``score`` (0-100), a ``confidence``
    band, one entry per ``criteria``, a plain-language ``rationale`` and a
    list of ``gaps`` -- the things worth looking at.
    """
    coverage = coverage or []
    criteria: list[dict[str, Any]] = []
    gaps: list[str] = []

    checks = (
        (
            "Source coverage",
            _source_coverage(extraction, coverage, diagram_source),
            1.5,
        ),
        (
            "Diagram completeness",
            _diagram_completeness(diagram_type, diagram_source),
            1.5,
        ),
        ("Structure captured", _structure_captured(extraction, mode), 1.0),
        (
            "Relationship integrity",
            _relationship_integrity(diagram_type, diagram_source),
            1.0,
        ),
        ("Faithfulness", _faithfulness(extraction, coverage, diagram_source), 1.0),
    )

    for name, (score, evidence), weight in checks:
        entry = _criterion(name, score, evidence)
        if entry is None:
            continue
        criteria.append(entry)
        if entry["verdict"] == "gap":
            gaps.append(f"{name}: {entry['evidence']}")

    if not criteria:
        return {
            "score": 0,
            "confidence": "low",
            "criteria": [],
            "rationale": "There was not enough output to grade.",
            "gaps": ["The extraction produced nothing to grade."],
        }

    total_weight = 0.0
    weighted = 0.0
    for entry in criteria:
        weight = next(w for name, _c, w in checks if name == entry["criterion"])
        weighted += entry["score"] * weight
        total_weight += weight
    score = weighted / total_weight if total_weight else 0.0

    confidence = (
        "high"
        if score >= _CONFIDENCE_HIGH
        else "medium"
        if score >= _CONFIDENCE_MEDIUM
        else "low"
    )

    weakest = min(criteria, key=lambda entry: entry["score"])
    rationale = (
        f"This reconstruction scores {int(round(score))} out of 100. "
        f"The weakest area is {weakest['criterion'].lower()}: {weakest['evidence']}"
    )

    return {
        "score": int(round(score)),
        "confidence": confidence,
        "criteria": criteria,
        "rationale": rationale,
        "gaps": gaps,
    }
