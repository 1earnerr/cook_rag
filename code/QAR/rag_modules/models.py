"""Shared evidence and answer contracts; independent of any corpus domain."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return f"[{self.evidence_id}] {self.title} / {self.section}\n{self.content}"


@dataclass
class RAGAnswer:
    answer: str
    sources: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StreamEvent:
    event: str  # sources / token / done
    text: str = ""
    sources: list[Evidence] = field(default_factory=list)
