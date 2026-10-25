"""Mermaid pipeline tests. Pure functions, no network."""

import pytest

from app import diagram as dgm

GOOD = "flowchart TD\n    A[Start] --> B[Next]"
GOOD_ER = """erDiagram
    USER ||--o{ ORDER : places
    USER {
        int id PK
        string name
    }"""
GOOD_MINDMAP = """mindmap
  root((Ideas))
    Alpha
    Beta"""


class TestStripFences:
    def test_plain_source_unchanged(self):
        assert dgm.strip_fences(GOOD) == GOOD

    def test_mermaid_fence_removed(self):
        assert dgm.strip_fences("```mermaid\n" + GOOD + "\n```") == GOOD

    def test_bare_fence_removed(self):
        assert dgm.strip_fences("```\n" + GOOD + "\n```") == GOOD

    def test_leading_prose_discarded(self):
        text = "Sure! Here is the diagram:\n\n" + GOOD
        assert dgm.strip_fences(text).startswith("flowchart TD")

    def test_empty_input(self):
        assert dgm.strip_fences("") == ""


class TestValidateMermaid:
    def test_valid_flowchart(self):
        out = dgm.validate_mermaid(GOOD)
        assert out["type"] == "flowchart"
        assert out["mermaid"].endswith("\n")

    def test_graph_alias_normalized_to_flowchart(self):
        out = dgm.validate_mermaid("graph LR\n    A --> B")
        assert out["type"] == "flowchart"

    def test_mindmap_supported(self):
        out = dgm.validate_mermaid(GOOD_MINDMAP)
        assert out["type"] == "mindmap"

    def test_er_diagram_supported(self):
        out = dgm.validate_mermaid(GOOD_ER)
        assert out["type"] == "erDiagram"

    def test_fenced_valid_source_accepted(self):
        assert dgm.validate_mermaid("```mermaid\n" + GOOD + "\n```")["type"] == "flowchart"

    def test_empty_rejected(self):
        with pytest.raises(dgm.DiagramError, match="empty"):
            dgm.validate_mermaid("   ")

    def test_none_rejected(self):
        with pytest.raises(dgm.DiagramError, match="No diagram"):
            dgm.validate_mermaid(None)

    def test_oversized_rejected(self):
        big = "flowchart TD\n" + "\n".join(f"    N{i} --> N{i+1}" for i in range(500))
        with pytest.raises(dgm.DiagramError, match="limit is"):
            dgm.validate_mermaid(big)

    def test_too_many_lines_rejected(self):
        many = "flowchart TD\n" + "\n".join(f"    N{i}[n{i}] --> M{i}[m{i}]" for i in range(260))
        with pytest.raises(dgm.DiagramError, match="limit is"):
            dgm.validate_mermaid(many)

    def test_unsupported_type_rejected(self):
        with pytest.raises(dgm.DiagramError, match="Unsupported diagram type"):
            dgm.validate_mermaid("pie title Pets\n    \"Dogs\" : 386")

    def test_script_tag_rejected(self):
        bad = "flowchart TD\n    A[Start] --> B[<script>alert(1)</script>]"
        with pytest.raises(dgm.DiagramError, match="rejected for safety"):
            dgm.validate_mermaid(bad)

    def test_unbalanced_brackets_rejected(self):
        bad = "flowchart TD\n    A[Start --> B[Next]"
        with pytest.raises(dgm.DiagramError, match="Unbalanced"):
            dgm.validate_mermaid(bad)

    def test_unbracketed_node_label_with_spaces_rejected(self):
        bad = "flowchart TD\n    Write --> Invalidate cache"
        with pytest.raises(dgm.DiagramError, match="bracketed when it contains spaces"):
            dgm.validate_mermaid(bad)


class TestRepairPrompt:
    def test_repair_prompt_contains_error_and_type(self):
        prompt = dgm.build_repair_prompt("flowchart TD\nA --> B", "Node label error", expected_type="flowchart")
        assert "Node label error" in prompt
        assert "flowchart TD" in prompt

    def test_repair_prompt_for_er(self):
        prompt = dgm.build_repair_prompt("erDiagram\nA ||--o{ B", "Unbalanced quotes", expected_type="erDiagram")
        assert "erDiagram" in prompt


class TestErRepair:
    """The failure in the field: an ER diagram whose attributes have no data
    type passes every generic check and then dies in the browser with
    "Expecting 'ATTRIBUTE_WORD', got 'BLOCK_STOP'"."""

    BROKEN = """erDiagram
    EMP {
        name
        id
    }
    COMPANY {
        name
        id
        emp_id
    }
    EMP ||--o{ COMPANY : "works for"
"""

    def test_a_bare_attribute_is_given_a_type(self):
        out = dgm.validate_mermaid(self.BROKEN)
        assert "string name" in out["mermaid"]
        assert "string emp_id" in out["mermaid"]

    def test_the_repair_is_reported_as_a_change(self):
        out = dgm.validate_mermaid(self.BROKEN)
        assert out["changes"], "a repair that is not reported is a silent overwrite"
        assert any("data type" in change for change in out["changes"])

    def test_the_change_says_which_entity_and_attribute(self):
        out = dgm.validate_mermaid(self.BROKEN)
        joined = " ".join(out["changes"])
        assert "EMP" in joined and "emp_id" in joined

    def test_a_valid_er_diagram_is_left_alone(self):
        out = dgm.validate_mermaid(GOOD_ER)
        assert out["changes"] == []
        assert "int id PK" in out["mermaid"]

    def test_an_entity_name_with_a_space_keeps_its_label(self):
        source = 'erDiagram\n    "Employee Record" {\n        string id PK\n    }\n'
        out = dgm.validate_mermaid(source)
        assert 'Employee_Record["Employee Record"]' in out["mermaid"]
        assert any("Renamed the entity" in change for change in out["changes"])

    def test_an_attribute_name_with_a_space_is_joined(self):
        source = "erDiagram\n    A {\n        string first name\n    }\n"
        out = dgm.validate_mermaid(source)
        assert "string first_name" in out["mermaid"]

    def test_a_key_without_a_type_is_repaired(self):
        source = "erDiagram\n    A {\n        id PK\n    }\n"
        out = dgm.validate_mermaid(source)
        assert "string id PK" in out["mermaid"]

    def test_an_invalid_cardinality_is_rejected_with_a_usable_message(self):
        source = 'erDiagram\n    A ||--?{ B : "weird"\n'
        with pytest.raises(dgm.DiagramError, match="not a valid relationship"):
            dgm.validate_mermaid(source)

    def test_an_unclosed_attribute_block_is_rejected(self):
        source = "erDiagram\n    A {\n        string id\n"
        with pytest.raises(dgm.DiagramError, match="never"):
            dgm.validate_mermaid(source)

    def test_a_relationship_without_a_label_gets_one(self):
        source = "erDiagram\n    A ||--o{ B\n"
        out = dgm.validate_mermaid(source)
        assert 'relates to' in out["mermaid"]
        assert out["changes"]

    def test_relationship_labels_with_spaces_are_quoted(self):
        source = 'erDiagram\n    A ||--o{ B : works for\n'
        out = dgm.validate_mermaid(source)
        assert '"works for"' in out["mermaid"]

    def test_er_warnings_flag_an_entity_without_attributes(self):
        source = 'erDiagram\n    A ||--o{ B : "has"\n'
        out = dgm.validate_mermaid(source)
        assert any("no attributes" in w for w in out["warnings"])

    def test_the_er_repair_prompt_teaches_the_er_rules(self):
        prompt = dgm.build_repair_prompt("erDiagram", "boom", expected_type="erDiagram")
        assert "type name" in prompt
        assert "||--o{" in prompt
        # A flowchart rule is meaningless for an ER diagram.
        assert "brackets" not in prompt

    def test_parse_er_reports_entities_attributes_and_relationships(self):
        parsed = dgm.parse_er(GOOD_ER)
        assert [e["id"] for e in parsed["entities"]] == ["USER", "ORDER"]
        assert parsed["entities"][0]["attributes"][0] == {
            "type": "int",
            "name": "id",
            "keys": ["PK"],
            "comment": "",
        }
        assert parsed["relationships"][0]["from"] == "USER"
        assert parsed["relationships"][0]["cardinality"] == "||--o{"


class TestCoverage:
    def test_er_labels_are_the_entities_and_attributes(self):
        labels = dgm.diagram_labels(GOOD_ER)
        assert "USER" in labels and "id" in labels

    def test_flowchart_labels_are_the_node_labels(self):
        labels = dgm.diagram_labels("flowchart TD\n    A[Start here] --> B[Next step]")
        assert labels == ["Start here", "Next step"]

    def test_missing_labels_are_reported(self):
        missing = dgm.missing_labels("flowchart TD\n    A[Start]", ["Start", "Finish"])
        assert missing == ["Finish"]

    def test_nothing_is_missing_when_everything_is_there(self):
        assert dgm.missing_labels(GOOD_ER, ["USER", "ORDER"]) == []
