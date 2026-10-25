"""Mermaid sanitization and validation.

The diagram is an enhancement. Structured notes are the product. Everything
here is written so that a hostile or broken diagram can never take down the
notes -- the worst case is that we show the source in a code block.

ER diagrams get their own treatment, because Mermaid's ER grammar is much
stricter than its flowchart grammar and the two mistakes models make most
often are invisible to the generic checks:

  1. Every attribute inside an entity block must be written ``type name``.
     A bare ``name`` is a hard parse error -- Mermaid reads the first word
     as the data type and then demands a second one ("Expecting
     'ATTRIBUTE_WORD', got 'BLOCK_STOP'").
  2. Entity identifiers are ALPHANUM: letters, digits, underscore and dash.
     A name containing a space must be written ``NAME["Display Name"]``.

Both are repaired deterministically here, and every repair is reported back
as a *change*, so the app can show the user what it altered in their own
sketch instead of silently overwriting it.
"""

from __future__ import annotations

import re

# Diagram types we render. Anything else is rejected rather than trusted.
SUPPORTED_TYPES = {"flowchart", "graph", "mindmap", "sequenceDiagram", "erDiagram"}

# Conservative ceiling. A hand-drawn whiteboard should never need more.
MAX_MERMAID_CHARS = 5000
MAX_MERMAID_LINES = 250

# Anchored form, used to confirm the source really is a diagram.
_TYPE_RE = re.compile(r"^\s*(flowchart|graph|mindmap|sequenceDiagram|erDiagram)\b", re.IGNORECASE)
# Unanchored form, used to find where the diagram starts when prose precedes it.
_TYPE_ANYWHERE_RE = re.compile(
    r"(?:^|\n)\s*(flowchart|graph|mindmap|sequenceDiagram|erDiagram)\b", re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Anything that could execute or exfiltrate. Mermaid has no legitimate need for
# these in a diagram body, so their presence means we discard the input.
_DANGEROUS = (
    "<script",
    "</script",
    "javascript:",
    "<iframe",
    "<object",
    "<embed",
    "<img",
    "<svg",
    "onerror=",
    "onload=",
    "onclick=",
    "data:text/html",
)


class DiagramError(ValueError):
    """Raised when Mermaid source cannot be accepted. Message is user-facing."""


def strip_fences(source: str) -> str:
    """Remove an optional outer ```mermaid fence and surrounding prose."""
    if not source:
        return ""
    text = source.strip()

    fenced = _FENCE_RE.findall(text)
    if fenced:
        # Take the longest fenced block; a short one is usually stray prose.
        text = max(fenced, key=len).strip()

    # If prose surrounds a diagram declaration, keep from the declaration on.
    match = _TYPE_ANYWHERE_RE.search(text)
    if match:
        text = text[match.start() :].strip()

    return text


def _reject_unsafe(source: str) -> None:
    lowered = source.lower()
    for token in _DANGEROUS:
        if token in lowered:
            raise DiagramError(
                "The diagram contained markup or script-like content and was rejected "
                "for safety."
            )


def _check_shape(source: str) -> None:
    if not source:
        raise DiagramError("The diagram was empty.")
    if len(source) > MAX_MERMAID_CHARS:
        raise DiagramError(
            f"The diagram was {len(source)} characters; the limit is "
            f"{MAX_MERMAID_CHARS}. It was omitted rather than truncated."
        )
    lines = [ln for ln in source.splitlines() if ln.strip()]
    if len(lines) > MAX_MERMAID_LINES:
        raise DiagramError(
            f"The diagram had {len(lines)} lines; the limit is {MAX_MERMAID_LINES}."
        )


def _check_types(source: str) -> str:
    match = _TYPE_RE.match(source)
    if not match:
        head = source.splitlines()[0][:60] if source.splitlines() else source[:60]
        raise DiagramError(
            f"Unsupported diagram type (starts with '{head}'). "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}."
        )
    declared = match.group(1).lower()
    # Normalize aliases and casing
    if declared == "graph":
        return "flowchart"
    if declared == "erdiagram":
        return "erDiagram"
    if declared == "sequencediagram":
        return "sequenceDiagram"
    if declared == "mindmap":
        return "mindmap"
    return declared


# ---------------------------------------------------------------------------
# ER diagram support
# ---------------------------------------------------------------------------

# Mermaid's ER cardinality tokens. The left token describes the parent side,
# the right token the child side.
ER_CARDINALITIES = {
    "||": "exactly one",
    "|o": "zero or one",
    "o|": "zero or one",
    "}o": "zero or more",
    "o{": "zero or more",
    "|{": "one or more",
    "}|": "one or more",
}

# Relationship line types: `--` identifying, `..` non-identifying.
ER_REL_TYPES = {
    "--": "identifying",
    "..": "non-identifying",
    "to": "identifying",
    "optionally to": "non-identifying",
    ".-": "non-identifying",
    "-.": "non-identifying",
}

ER_KEYS = ("PK", "FK", "UK")

# Data types a model reaches for. Used only to tell a `type name` pair apart
# from a two-word attribute name; an unrecognised word is left alone rather
# than guessed at.
ER_SCALAR_TYPES = frozenset(
    {
        "string", "text", "varchar", "nvarchar", "char", "nchar", "clob",
        "json", "jsonb", "xml", "uuid", "guid", "blob", "bytea", "binary",
        "int", "integer", "bigint", "smallint", "tinyint", "serial",
        "bigserial", "number", "numeric", "decimal", "float", "double",
        "real", "money", "bool", "boolean", "bit", "date", "datetime",
        "timestamp", "timestamptz", "time", "interval", "year", "enum",
        "set", "array", "url", "email", "phone", "percent", "ratio",
    }
)

_ER_ALNUM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9\-_]*$")
_ER_KEY_SET = frozenset(ER_KEYS)

# `PARENT ||--o{ CHILD` -- the middle group is cardinality, line type and
# cardinality again.
_ER_REL_RE = re.compile(
    r"^(?P<left>.*?)"
    r"\s*(?P<c1>\|\||\|o|o\||\}o|o\{|\|\{|\}\|)"
    r"\s*(?P<rt>optionally to|--|\.\.|to|\.-|\-\.)"
    r"\s*(?P<c2>\|\||\|o|o\||\}o|o\{|\|\{|\}\|)"
    r"\s*(?P<right>.*)$"
)

# `NAME {` or `NAME["Display Name"] {`
_ER_HEADER_RE = re.compile(
    r'^(?P<ident>[A-Za-z_][A-Za-z0-9\-_]*)(\["(?P<label>[^"]*)"\])?$'
)

# `type name [PK[,FK]] ["comment"]`
_ER_ATTRIBUTE_RE = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_\[\]\(\)]*)"
    r"\s+(?P<name>[A-Za-z_][A-Za-z0-9\-_]*)"
    r"(\s+(?P<keys>PK|FK|UK)(,\s*(?:PK|FK|UK))*)?"
    r'(\s+"(?P<comment>[^"]*)")?$'
)


def _split_quoted(line: str) -> list[str]:
    """Split on whitespace, keeping a quoted run together as one token."""
    tokens: list[str] = []
    current: list[str] = []
    in_quote = False
    for char in line:
        if char == '"':
            in_quote = not in_quote
            current.append(char)
            continue
        if char.isspace() and not in_quote:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _split_relationship(line: str) -> tuple[str, str] | None:
    """Split ``A ||--o{ B : label`` into its left half and its label."""
    in_quote = False
    cut = -1
    for index, char in enumerate(line):
        if char == '"':
            in_quote = not in_quote
        elif char == ":" and not in_quote:
            cut = index
    if cut == -1:
        return None
    return line[:cut].strip(), line[cut + 1 :].strip()


class _ErRewriter:
    """Rewrites an ER source into the strict subset Mermaid can parse.

    Holds the name mapping so an entity written one way in its box and
    another way in a relationship still lands on the same identifier.
    """

    def __init__(self) -> None:
        self.ids: dict[str, str] = {}
        self.labels: dict[str, str | None] = {}
        self.used: set[str] = set()
        self.changes: list[str] = []

    def identifier(self, raw: str) -> tuple[str, str | None]:
        """The identifier for an entity name, plus a display label if needed."""
        key = raw.strip().strip('"').strip()
        if key in self.ids:
            return self.ids[key], self.labels.get(key)

        if _ER_ALNUM_RE.match(key):
            ident = key
        else:
            ident = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_") or "ENTITY"
            if not (ident[0].isalpha() or ident[0] == "_"):
                ident = "_" + ident
        base, suffix = ident, 2
        while ident in self.used:
            ident = f"{base}_{suffix}"
            suffix += 1
        self.used.add(ident)
        self.ids[key] = ident
        self.labels[key] = None if ident == key else key
        return ident, self.labels[key]


def _is_relationship_line(line: str) -> bool:
    """True when a line is meant to be an ER relationship.

    Detected loosely on purpose: a relationship with a broken cardinality is
    still a relationship, and it must be reported as one instead of being
    mistaken for an entity block.
    """
    without_quotes = re.sub(r'"[^"]*"', "", line)
    return "--" in without_quotes or ".." in without_quotes


def _er_attribute(line: str) -> tuple[str, list[str]]:
    """Repair one ER attribute line into ``type name [KEY] ["comment"]``.

    Returns the repaired line and the notes describing what was changed.
    """
    tokens = _split_quoted(line)
    if not tokens:
        return line, []

    comment = None
    if len(tokens) > 1 and tokens[-1].startswith('"') and tokens[-1].endswith('"'):
        comment = tokens[-1]
        tokens = tokens[:-1]

    keys: list[str] = []
    if tokens and "," in tokens[-1]:
        parts = [part.strip().upper() for part in tokens[-1].split(",") if part.strip()]
        if parts and all(part in _ER_KEY_SET for part in parts):
            keys = parts
            tokens = tokens[:-1]
    while tokens and tokens[-1].upper() in _ER_KEY_SET:
        keys.insert(0, tokens[-1].upper())
        tokens = tokens[:-1]

    if not tokens:
        return line, []

    notes: list[str] = []
    if len(tokens) == 1:
        fixed = f"string {tokens[0]}"
        notes.append(f"type added to `{tokens[0]}`")
    elif len(tokens) == 2:
        # Already a type and a name (or a name and a key, handled above).
        fixed = f"{tokens[0]} {tokens[1]}"
    elif tokens[0].lower() in ER_SCALAR_TYPES:
        name = "_".join(tokens[1:])
        fixed = f"{tokens[0]} {name}"
        notes.append(f"`{' '.join(tokens[1:])}` renamed to `{name}`")
    else:
        name = "_".join(tokens)
        fixed = f"string {name}"
        notes.append(f"`{' '.join(tokens)}` renamed to `string {name}`")

    if keys:
        fixed += " " + ",".join(keys)
    if comment:
        fixed += " " + comment
    return fixed, notes


def _er_header(header: str, rewriter: _ErRewriter) -> tuple[str, str | None]:
    """Repair an entity header into an identifier plus an optional label."""
    name, alias = header, None
    bracket = header.find("[")
    if bracket != -1:
        name = header[:bracket].strip()
        alias = header[bracket + 1 :].rsplit("]", 1)[0].strip().strip('"').strip()

    ident, derived = rewriter.identifier(name)
    if derived:
        rewriter.changes.append(
            f"Renamed the entity `{name}` to `{ident}`: Mermaid entity identifiers "
            "cannot contain spaces. The original name is kept as the label under "
            "the box."
        )
    return ident, alias or derived


def _er_relationship(line: str, rewriter: _ErRewriter) -> str:
    """Repair one ER relationship line."""
    split = _split_relationship(line)
    if split is None:
        # No colon: the relationship has no label at all.
        left, label = line, ""
    else:
        left, label = split
    match = _ER_REL_RE.match(left)
    if match is None:
        # Leave a malformed relationship alone; the validator reports it with
        # the exact line, which is more useful than a mangled rewrite.
        return line

    left_id, _ = rewriter.identifier(match.group("left"))
    right_id, _ = rewriter.identifier(match.group("right"))
    spec = f'{match.group("c1")}{match.group("rt")}{match.group("c2")}'

    original = label
    if not label:
        label = '"relates to"'
        rewriter.changes.append(
            "Added the label \"relates to\" to a relationship that had none: "
            "Mermaid requires a label after the colon."
        )
    elif not (label.startswith('"') and label.endswith('"')):
        label = f'"{label}"'
        if " " in original:
            rewriter.changes.append(
                f"Quoted the relationship label `{original}` so Mermaid reads it as "
                "one label instead of two."
            )
    return f"{left_id} {spec} {right_id} : {label}"


def normalize_er_source(source: str) -> tuple[str, list[str]]:
    """Rewrite an ER source into syntax Mermaid can parse.

    Returns the rewritten source and the list of changes made to it. Each
    change is something the app did to the user's own drawing, so it is
    surfaced rather than applied silently.
    """
    rewriter = _ErRewriter()
    output: list[str] = []
    in_block = False
    current_entity = ""
    seen_header = False
    # Attribute edits are collected per entity so the user is told "3
    # attributes of EMP" once, not once per attribute.
    edits: dict[str, list[str]] = {}

    for raw_line in source.splitlines():
        line = raw_line.strip()
        indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        if not line:
            continue
        if not seen_header and line.lower().startswith("erdiagram"):
            seen_header = True
            output.append(line)
            continue
        if line.startswith("%%") or line.lower().startswith(("acctitle:", "accdescr:")):
            output.append(line)
            continue

        if in_block:
            if line.startswith("}"):
                in_block = False
                output.append(indent + "}")
                current_entity = ""
                continue
            fixed, notes = _er_attribute(line)
            if notes:
                edits.setdefault(current_entity, []).extend(notes)
            output.append(indent + fixed)
            continue

        if _is_relationship_line(line):
            output.append(_er_relationship(line, rewriter))
            continue

        if "{" in line:
            header, _, rest = line.partition("{")
            ident, label = _er_header(header.strip(), rewriter)
            current_entity = label or ident
            output.append(
                f'{indent}{ident}["{label}"] {{' if label else f"{indent}{ident} {{"
            )
            in_block = not rest.strip().startswith("}")
            if rest.strip():
                # `NAME { string id PK` -- an attribute on the header line.
                fixed, notes = _er_attribute(rest.strip())
                if notes:
                    edits.setdefault(current_entity, []).extend(notes)
                output.append(indent + "    " + fixed)
            continue

        ident, _ = rewriter.identifier(line)
        output.append(ident)

    for entity, notes in edits.items():
        typed = [note for note in notes if note.startswith("type added")]
        renamed = [note for note in notes if not note.startswith("type added")]
        if typed:
            names = ", ".join(note.split("`")[1] for note in typed)
            rewriter.changes.append(
                f"Added the data type `string` to {len(typed)} attribute(s) of "
                f"`{entity}` ({names}): Mermaid requires a type in front of every "
                "ER attribute name."
            )
        if renamed:
            names = ", ".join(note.split("`")[1] for note in renamed)
            rewriter.changes.append(
                f"Rewrote {len(renamed)} attribute name(s) of `{entity}` ({names}): "
                "Mermaid attribute names cannot contain spaces."
            )

    return "\n".join(output), rewriter.changes


def parse_er(source: str) -> dict:
    """Parse a normalized ER source into entities and relationships.

    Never raises: unusable lines are collected as problems so the caller can
    report them all at once.
    """
    entities: list[dict] = []
    by_id: dict[str, dict] = {}
    relationships: list[dict] = []
    problems: list[str] = []
    current: dict | None = None

    for number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue
        if line.lower().startswith(("erdiagram", "acctitle:", "accdescr:")):
            continue

        if current is not None:
            if line.startswith("}"):
                current = None
                continue
            match = _ER_ATTRIBUTE_RE.match(line)
            if match is None:
                problems.append(
                    f"Line {number}: `{line}` is not a valid ER attribute. Write it as "
                    "`type name`, for example `string name`."
                )
                continue
            keys = [
                part.strip()
                for part in (match.group("keys") or "").split(",")
                if part.strip()
            ]
            current["attributes"].append(
                {
                    "type": match.group("type"),
                    "name": match.group("name"),
                    "keys": keys,
                    "comment": match.group("comment") or "",
                }
            )
            continue

        if line.endswith("{"):
            header = line[:-1].strip()
            match = _ER_HEADER_RE.match(header)
            if match is None:
                problems.append(
                    f"Line {number}: `{header}` is not a valid entity name. Use letters, "
                    "digits and underscores only."
                )
                current = None
                continue
            ident = match.group("ident")
            entity = {
                "id": ident,
                "label": match.group("label") or ident,
                "attributes": [],
                "line": number,
            }
            entities.append(entity)
            by_id[ident] = entity
            current = entity
            continue

        split = _split_relationship(line)
        if split is not None and _is_relationship_line(line):
            left, label = split
            match = _ER_REL_RE.match(left)
            if match is None:
                problems.append(
                    f"Line {number}: `{line}` is not a valid relationship. Write it as "
                    "`PARENT ||--o{ CHILD : \"label\"`, where the cardinality is one of "
                    "`||`, `|o`, `o{` or `|{`."
                )
                continue
            relationships.append(
                {
                    "from": match.group("left").strip(),
                    "to": match.group("right").strip(),
                    "cardinality": (
                        f'{match.group("c1")}{match.group("rt")}{match.group("c2")}'
                    ),
                    "label": label.strip('"'),
                    "line": number,
                }
            )
            continue

        if not _ER_ALNUM_RE.match(line):
            problems.append(
                f"Line {number}: `{line}` is not a valid entity name. Use letters, "
                "digits and underscores only."
            )

    if current is not None:
        problems.append(
            f"Line {current['line']}: the attribute block of `{current['id']}` is never "
            "closed with `}`."
        )

    # An entity that only appears in a relationship still gets a box in the
    # rendered diagram, so it belongs in the entity list -- with the flag
    # that says nobody gave it any attributes.
    for relationship in relationships:
        for side in ("from", "to"):
            name = relationship[side]
            if name in by_id:
                continue
            entity = {
                "id": name,
                "label": name,
                "attributes": [],
                "declared": False,
                "line": relationship["line"],
            }
            entities.append(entity)
            by_id[name] = entity

    return {
        "entities": entities,
        "relationships": relationships,
        "problems": problems,
    }


def _er_soft_warnings(parsed: dict) -> list[str]:
    """Problems that do not stop an ER diagram from rendering."""
    entities = parsed["entities"]
    relationships = parsed["relationships"]
    warnings: list[str] = []

    if not entities and not relationships:
        return warnings

    connected: set[str] = set()
    for relationship in relationships:
        connected.add(relationship["from"])
        connected.add(relationship["to"])
        if not relationship["label"]:
            warnings.append(
                f"The relationship `{relationship['from']}` → "
                f"`{relationship['to']}` has no label, so the diagram does not say "
                "what the link means."
            )

    for entity in entities:
        if not entity["attributes"]:
            warnings.append(
                f"Entity `{entity['label']}` has no attributes, so its box is empty."
            )
        elif entity["id"] not in connected and len(entities) > 1:
            warnings.append(
                f"Entity `{entity['label']}` is not connected to anything, which usually "
                "means a relationship was missed."
            )

    return warnings


def _check_er(source: str) -> list[str]:
    """Validate a normalized ER source. Returns soft warnings."""
    parsed = parse_er(source)
    if parsed["problems"]:
        raise DiagramError(parsed["problems"][0])
    if not parsed["entities"] and not parsed["relationships"]:
        raise DiagramError("The ER diagram declared no entities.")
    return _er_soft_warnings(parsed)


def _er_warnings(source: str) -> list[str]:
    """Soft warnings for an ER source that is already known to be valid."""
    return _er_soft_warnings(parse_er(source))


def _check_syntax_heuristics(source: str) -> None:
    """Cheap structural checks that catch the most common generation failures."""
    first = source.splitlines()[0].strip()
    if first.lower().startswith("erdiagram"):
        # ER has its own grammar; the flowchart rules below do not apply.
        _check_er(source)
        return
    check_source = source

    # Bracket/quote balance. Unbalanced delimiters are the single most common
    # defect in model-generated Mermaid and render as a hard failure.
    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if check_source.count(opener) != check_source.count(closer):
            raise DiagramError(
                f"Unbalanced '{opener}{closer}' in the diagram "
                f"({check_source.count(opener)} vs {check_source.count(closer)})."
            )
    if check_source.count('"') % 2 != 0:
        raise DiagramError("Unbalanced double quotes in the diagram.")

    body = "\n".join(source.splitlines()[1:]).strip()
    if first.lower().startswith(("flowchart", "graph")) and not body:
        raise DiagramError("The diagram declared no nodes.")

    if "-->" not in source and "---" not in source and "-" not in source:
        # mindmap, sequenceDiagram, and erDiagram legitimately have different syntax.
        if not first.lower().startswith(("mindmap", "sequencediagram", "erdiagram")):
            raise DiagramError("The diagram contains no relationships or steps.")

    # Only run flowchart-specific node label validation on flowcharts
    if first.lower().startswith(("flowchart", "graph")):
        _check_node_labels(source)


# A bare node id containing a space is the single most common way a model
# produces Mermaid that balances fine and still will not parse:
#   Write --> Invalidate cache      <- rejected by mermaid, needs Write[Write]
_LINK_SIDE = re.compile(r"(?:-->|---|-\.->|==>)\s*(.+)$")
_LABELLED = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|\"[^\"]*\"|'[^']*'")


def _check_node_labels(source: str) -> None:
    """Reject unbracketed multi-word node references on link lines."""
    for line in source.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        match = _LINK_SIDE.search(stripped)
        if not match:
            continue
        tail = _LABELLED.sub("", match.group(1)).strip()
        # Remove edge decorations the model may append, e.g. "--> B ~~~ yes" or "|text|".
        tail = re.sub(r"[~|].*$", "", tail).strip()
        if not tail:
            continue
        if " " in tail:
            raise DiagramError(
                f"Node label must be bracketed when it contains spaces: "
                f"'{tail}' (found on line: {stripped[:60]})."
            )


def validate_mermaid(
    source: str, explanation: str = "", changes: list[str] | None = None
) -> dict:
    """Validate Mermaid source. Raises DiagramError if unusable.

    Returns {"type", "mermaid", "explanation", "changes", "warnings"} on
    success. ``changes`` lists the edits the app made to the model's source
    so the caller can show them to the user; ``warnings`` lists soft
    problems that do not stop the diagram from rendering.
    """
    if source is None:
        raise DiagramError("No diagram was provided.")

    cleaned = strip_fences(source)
    _check_shape(cleaned)
    _reject_unsafe(cleaned)
    diagram_type = _check_types(cleaned)

    applied: list[str] = []
    warnings: list[str] = []

    if diagram_type == "erDiagram":
        # Normalize before validating: the repairs are what make the source
        # parseable, and the validator checks the repaired form.
        cleaned, applied = normalize_er_source(cleaned)
        _check_shape(cleaned)
        _reject_unsafe(cleaned)

    _check_syntax_heuristics(cleaned)

    if diagram_type == "erDiagram":
        warnings = _er_warnings(cleaned)

    # Normalize: single trailing newline, no tabs.
    cleaned = cleaned.replace("\t", "    ").rstrip() + "\n"

    if changes is not None:
        changes.extend(applied)

    return {
        "type": diagram_type,
        "mermaid": cleaned,
        "explanation": explanation.strip() if isinstance(explanation, str) else "",
        "changes": applied,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Coverage: does the diagram actually show what the source contains?
# ---------------------------------------------------------------------------

# `A[Label]`, `A(Label)`, `A{Label}`, `A[[Label]]`
_NODE_LABEL_RE = re.compile(r"\[([^\[\]]+)\]|\(([^()]+)\)|\{([^{}]+)\}")


def diagram_labels(source: str) -> list[str]:
    """Every label the diagram shows, as written.

    Used to check that the reconstruction covers the elements the model
    reported seeing in the source image.
    """
    if not source:
        return []
    first = source.splitlines()[0].strip().lower()
    labels: list[str] = []

    if first.startswith("erdiagram"):
        parsed = parse_er(source)
        for entity in parsed["entities"]:
            labels.append(entity["label"])
            labels.extend(attr["name"] for attr in entity["attributes"])
        for relationship in parsed["relationships"]:
            if relationship["label"]:
                labels.append(relationship["label"])
        return labels

    for line in source.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        for match in _NODE_LABEL_RE.finditer(stripped):
            label = next((group for group in match.groups() if group), "")
            label = label.strip().strip('"').strip("'")
            if label:
                labels.append(label)
    return labels


def diagram_elements(source: str) -> list[str]:
    """The top-level elements of the diagram: the boxes a reader sees.

    For an ER diagram these are the entities, not their attributes; for a
    flowchart they are the nodes. This is the granularity the source
    inventory is written at, so it is the one a fidelity check compares.
    """
    if not source:
        return []
    first = source.splitlines()[0].strip().lower()
    if first.startswith("erdiagram"):
        return [entity["label"] for entity in parse_er(source)["entities"]]
    return diagram_labels(source)


def missing_labels(source: str, labels: list[str]) -> list[str]:
    """Labels that do not appear anywhere in the diagram source."""
    if not source:
        return list(labels)
    lowered = source.lower()
    missing: list[str] = []
    for label in labels:
        token = str(label).strip()
        if not token:
            continue
        if token.lower() not in lowered:
            missing.append(token)
    return missing


def build_repair_prompt(source: str, error: str, expected_type: str = "flowchart") -> str:
    """Message sent back to the model for the single corrective retry.

    The requirements are type-specific: a flowchart rule ("wrap every label
    with a space in brackets") is meaningless for an ER diagram, and an ER
    rule ("every attribute needs a data type") is what actually breaks ER
    output. Sending the wrong rule back is how a repair round reproduces the
    same broken diagram.
    """
    if expected_type == "erDiagram":
        return (
            "Your Mermaid ER diagram was rejected by the validator with this error:\n"
            f"{error}\n\n"
            "Return ONLY a corrected Mermaid ER diagram, with no prose and no code fence.\n"
            "The syntax rules you must follow:\n"
            "  - The first line must be exactly: erDiagram\n"
            "  - Entity names use letters, digits and underscores only. If the original "
            "name contains a space, write it as: NAME[\"Display Name\"]\n"
            "  - EVERY attribute inside an entity block must be written as `type name`, "
            "for example `string name` or `int id`. A bare `name` on its own is a "
            "syntax error and Mermaid will refuse to draw the diagram.\n"
            "  - A key may follow the name: `string id PK` or `string emp_id FK`.\n"
            "  - Relationships are written: PARENT ||--o{ CHILD : \"label\"\n"
            "  - Cardinalities: `||` exactly one, `|o` zero or one, `o{` zero or more, "
            "`|{` one or more. The separator is `--` (identifying) or `..` "
            "(non-identifying).\n"
            "  - The label after the colon must be in double quotes.\n"
            "  - Keep every entity, attribute and relationship from your previous "
            "attempt; only fix the syntax.\n"
            "  - Keep it under 60 lines.\n\n"
            f"The diagram you attempted:\n{source}\n\n"
            "Return only the corrected diagram."
        )

    type_instruction = "First line must be: flowchart TD"
    if expected_type == "mindmap":
        type_instruction = "First line must be: mindmap"
    elif expected_type == "sequenceDiagram":
        type_instruction = "First line must be: sequenceDiagram"

    return (
        "Your Mermaid diagram was rejected by the validator with this error:\n"
        f"{error}\n\n"
        "Return ONLY a corrected Mermaid diagram, with no prose and no code fence.\n"
        "Requirements:\n"
        f"  - {type_instruction}\n"
        "  - EVERY flowchart node label containing a space MUST be wrapped in brackets, e.g. Node[Node Name].\n"
        "  - Brackets, quotes, and braces must balance.\n"
        "  - No HTML, <script>, <style>, or any markup tags.\n"
        "  - Keep it under 50 lines.\n\n"
        f"The diagram you attempted:\n{source}\n\n"
        "Return only the corrected diagram."
    )
