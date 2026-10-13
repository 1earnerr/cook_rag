"""Stable HTTP request and response contracts for the QAR API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


NonEmptyQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FilterScalar = str | int | float | bool
FilterValue = FilterScalar | list[FilterScalar]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(APIModel):
    query: NonEmptyQuery
    filters: dict[str, FilterValue] = Field(default_factory=dict)

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, filters: dict[str, FilterValue]) -> dict[str, FilterValue]:
        if any(not key.strip() for key in filters):
            raise ValueError("filter key must not be empty")
        if any(isinstance(value, list) and not value for value in filters.values()):
            raise ValueError("filter list must not be empty")
        return filters


class SourceResponse(APIModel):
    source_id: str
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str


class RetrievalInfo(APIModel):
    mode: Literal["hybrid"] = "hybrid"
    count: int = Field(ge=0)


class ChatResponse(APIModel):
    answer: str
    sources: list[SourceResponse]
    retrieval: RetrievalInfo


class HealthResponse(APIModel):
    status: Literal["ok", "degraded"]
    embedding_loaded: bool
    index_loaded: bool
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    detail: str | None = None


class ReindexResponse(APIModel):
    success: bool
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    index_schema_version: int = Field(ge=1)
    elapsed_seconds: float = Field(ge=0)
