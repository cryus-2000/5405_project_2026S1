"""Lightweight rule-based query decomposition for Charades-STA sentences."""

from dataclasses import dataclass, field
import re
from typing import Iterable


SEQUENTIAL_RELATIONS = {
    "then",
    "and then",
    "afterward",
    "afterwards",
    "next",
    "finally",
    "before",
    "after",
}

PARALLEL_RELATIONS = {
    "and",
    "while",
    "as",
}

BOUNDARY_PATTERN = re.compile(
    r"\s*(,|;|\band then\b|\bthen\b|\bafterwards?\b|\bnext\b|\bfinally\b|"
    r"\bbefore\b|\bafter\b|\bwhile\b|\bas\b|\band\b)\s*",
    re.IGNORECASE,
)

MARKER_PATTERN = re.compile(
    r"^(,|;|and then|then|afterwards?|next|finally|before|after|while|as|and)$",
    re.IGNORECASE,
)

SUBJECT_PATTERN = re.compile(
    r"^(a person|the person|person|someone|somebody|one person|another person|"
    r"the other person|other person|two people|a man|the man|a woman|the woman|"
    r"he|she|they)\b",
    re.IGNORECASE,
)

LEADING_FILLER_PATTERN = re.compile(
    r"^(a video of|a clip of|there is|there are|we see|the video shows)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryClause:
    text: str
    index: int
    relation_to_previous: str = "start"
    marker: str | None = None
    source: str = "rule"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRelation:
    source_index: int
    target_index: int
    relation: str
    marker: str | None = None


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    clauses: list[QueryClause]
    relations: list[QueryRelation]
    backend: str = "rule"

    @property
    def sub_queries(self) -> list[str]:
        return [clause.text for clause in self.clauses]

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "sub_queries": self.sub_queries,
            "relations": [
                {
                    "source_index": relation.source_index,
                    "target_index": relation.target_index,
                    "relation": relation.relation,
                    "marker": relation.marker,
                }
                for relation in self.relations
            ],
            "backend": self.backend,
        }


def normalize_query(query: str) -> str:
    query = query.strip()
    query = LEADING_FILLER_PATTERN.sub("", query)
    query = re.sub(r"\s+", " ", query)
    return query.rstrip(".")


def relation_from_marker(marker: str | None) -> str:
    if marker is None:
        return "unknown"
    marker = marker.strip().lower()
    if marker in SEQUENTIAL_RELATIONS or marker in {",", ";"}:
        return "sequential"
    if marker in PARALLEL_RELATIONS:
        return "parallel"
    return "unknown"


def infer_subject(text: str) -> str | None:
    match = SUBJECT_PATTERN.match(text.strip())
    return match.group(1) if match else None


def repair_clause_subject(text: str, subject: str | None) -> str:
    text = text.strip()
    if not text or subject is None or infer_subject(text) is not None:
        return text
    return f"{subject} {text}"


def deduplicate_clauses(clauses: Iterable[QueryClause]) -> list[QueryClause]:
    seen = set()
    result = []
    for clause in clauses:
        key = clause.text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(
            QueryClause(
                text=clause.text,
                index=len(result),
                relation_to_previous=clause.relation_to_previous if result else "start",
                marker=clause.marker if result else None,
                source=clause.source,
                metadata=clause.metadata,
            )
        )
    return result


def build_relations(clauses: list[QueryClause]) -> list[QueryRelation]:
    relations = []
    for index, clause in enumerate(clauses[1:], start=1):
        relations.append(
            QueryRelation(
                source_index=index - 1,
                target_index=index,
                relation=clause.relation_to_previous,
                marker=clause.marker,
            )
        )
    return relations


class RuleBasedQueryParser:
    backend = "rule"

    def parse(self, query: str) -> ParsedQuery:
        """Split a query on simple temporal/connective markers.

        Example:
        "person opens the door and then turns on the light"
        becomes two clauses with a sequential relation. If the second clause
        loses the subject during splitting, we copy the previous subject back in
        so SigLIP2 receives a complete phrase.
        """
        normalized = normalize_query(query)
        pieces = [piece for piece in BOUNDARY_PATTERN.split(normalized) if piece and piece.strip()]

        clauses = []
        current_marker = None
        subject = infer_subject(normalized)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if MARKER_PATTERN.fullmatch(piece):
                current_marker = piece.lower()
                continue

            text = repair_clause_subject(piece, subject)
            if not text:
                continue
            subject = infer_subject(text) or subject
            relation = "start" if not clauses else relation_from_marker(current_marker)
            clauses.append(
                QueryClause(
                    text=text,
                    index=len(clauses),
                    relation_to_previous=relation,
                    marker=current_marker if clauses else None,
                    source=self.backend,
                )
            )
            current_marker = None

        if not clauses:
            clauses = [
                QueryClause(
                    text=normalized,
                    index=0,
                    relation_to_previous="start",
                    source=self.backend,
                )
            ]

        clauses = deduplicate_clauses(clauses)
        return ParsedQuery(
            original=query,
            clauses=clauses,
            relations=build_relations(clauses),
            backend=self.backend,
        )


def parse_query(query: str, backend: str = "rule") -> ParsedQuery:
    """Parse one raw query; 'auto' is accepted as an alias for the rule parser."""
    if backend not in {"rule", "auto"}:
        raise ValueError(f"Unsupported query parser backend: {backend}. Use 'rule'.")
    return RuleBasedQueryParser().parse(query)
