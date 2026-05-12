"""Query decomposition for Charades-STA sentences.

The original paper uses AllenNLP as the raw-query parser. This project replaces
that parser with spaCy dependency parsing for Windows compatibility.
"""

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

SPACY_MODEL_NAME = "en_core_web_sm"
SUBJECT_DEPS = {"nsubj", "nsubjpass", "expl"}
VERB_POS = {"VERB", "AUX"}
SEPARATE_ACTION_DEPS = {"conj", "advcl", "relcl"}
MARKER_TOKEN_DEPS = {"advmod", "cc", "mark"}
MARKER_WORDS = {
    word
    for marker in SEQUENTIAL_RELATIONS | PARALLEL_RELATIONS
    for word in marker.split()
}
_SPACY_NLP = None


@dataclass(frozen=True)
class QueryClause:
    text: str
    index: int
    relation_to_previous: str = "start"
    marker: str | None = None
    source: str = "spacy"
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
    backend: str = "spacy"

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


def clean_phrase(text: str) -> str:
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;")


def load_spacy_model(model_name: str = SPACY_MODEL_NAME):
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Install it with "
            "`D:\\Anaconda\\envs\\RP\\python.exe -m pip install spacy`."
        ) from exc
    try:
        _SPACY_NLP = spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model `{model_name}` is not installed. Install it with "
            f"`D:\\Anaconda\\envs\\RP\\python.exe -m spacy download {model_name}`."
        ) from exc
    return _SPACY_NLP


def token_text(tokens) -> str:
    return clean_phrase(" ".join(token.text for token in sorted(tokens, key=lambda item: item.i)))


class SpacyQueryParser:
    backend = "spacy"

    def __init__(self, model_name: str = SPACY_MODEL_NAME):
        self.nlp = load_spacy_model(model_name)

    def parse(self, query: str) -> ParsedQuery:
        """Extract verb-centered simple queries with spaCy dependency parsing."""
        normalized = normalize_query(query)
        doc = self.nlp(normalized)
        subject = infer_subject(normalized) or self._document_subject(doc)
        clauses = []
        previous_end = -1

        for verb in self._action_verbs(doc):
            phrase = self._verb_phrase(verb, subject)
            if not phrase:
                continue

            subject = self._subject_for_verb(verb) or subject
            marker = None if not clauses else self._marker_between(doc, previous_end, verb.i)
            relation = "start" if not clauses else relation_from_marker(marker)
            clauses.append(
                QueryClause(
                    text=phrase,
                    index=len(clauses),
                    relation_to_previous=relation,
                    marker=marker,
                    source=self.backend,
                    metadata={"verb": verb.lemma_, "verb_index": verb.i},
                )
            )
            previous_end = max(previous_end, self._phrase_end(verb))

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

    def _action_verbs(self, doc):
        verbs = [
            token
            for token in doc
            if token.pos_ == "VERB" and not self._has_verbal_ancestor(token)
        ]
        if verbs:
            return verbs
        return [
            token
            for token in doc
            if token.pos_ == "AUX" and token.dep_ == "ROOT"
        ]

    def _has_verbal_ancestor(self, token) -> bool:
        for ancestor in token.ancestors:
            if ancestor.pos_ in VERB_POS and token.dep_ not in SEPARATE_ACTION_DEPS:
                return True
        return False

    def _verb_phrase(self, verb, inherited_subject: str | None) -> str:
        tokens = set()
        self._collect_action_tokens(verb, tokens)
        phrase = token_text(tokens)
        subject = self._subject_for_verb(verb) or inherited_subject
        return repair_clause_subject(phrase, subject)

    def _collect_action_tokens(self, token, tokens: set) -> None:
        if token.is_punct or token.dep_ == "cc":
            return
        if token.dep_ in MARKER_TOKEN_DEPS and token.lower_ in MARKER_WORDS:
            return
        tokens.add(token)
        for child in token.children:
            if child.pos_ in VERB_POS and child.dep_ in SEPARATE_ACTION_DEPS:
                continue
            self._collect_action_tokens(child, tokens)

    def _subject_for_verb(self, verb) -> str | None:
        for child in verb.children:
            if child.dep_ in SUBJECT_DEPS:
                return token_text(child.subtree)
        for ancestor in verb.ancestors:
            for child in ancestor.children:
                if child.dep_ in SUBJECT_DEPS:
                    return token_text(child.subtree)
        return None

    def _document_subject(self, doc) -> str | None:
        for token in doc:
            if token.dep_ in SUBJECT_DEPS:
                return token_text(token.subtree)
        return None

    def _marker_between(self, doc, previous_end: int, verb_index: int) -> str | None:
        between = doc[max(previous_end + 1, 0) : verb_index].text.lower()
        for marker in sorted(SEQUENTIAL_RELATIONS | PARALLEL_RELATIONS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(marker)}\b", between):
                return marker
        if "," in between:
            return ","
        if ";" in between:
            return ";"
        return None

    def _phrase_end(self, verb) -> int:
        tokens = set()
        self._collect_action_tokens(verb, tokens)
        return max(token.i for token in tokens)


def parse_query(query: str, backend: str = "spacy") -> ParsedQuery:
    """Parse one raw query with the spaCy dependency parser."""
    if backend == "spacy":
        return SpacyQueryParser().parse(query)
    raise ValueError(f"Unsupported query parser backend: {backend}. Use 'spacy'.")
