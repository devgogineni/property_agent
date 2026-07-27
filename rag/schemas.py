from typing import Literal

from pydantic import BaseModel, Field


class ExtractedQuery(BaseModel):
    """Location/property entities extracted from a user's free-text question."""

    postcode: str | None = Field(
        default=None,
        description="A full UK postcode, e.g. 'BR6 1AA'. Only set this for a complete postcode - use postcode_prefix for a partial one.",
    )
    postcode_prefix: str | None = Field(
        default=None,
        description="A partial postcode / postcode district (the outward code, or just the first few letters/digits), e.g. 'BR6', 'SE19'. Set this when the user gives an area code rather than a full postcode.",
    )
    street: str | None = None
    locality: str | None = None
    town: str | None = None
    district: str | None = None
    county: str | None = None
    property_type: str | None = None  # D=Detached, S=Semi-detached, T=Terraced, F=Flat, O=Other
    new_build: bool | None = None
    min_price: int | None = None
    max_price: int | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    request_id: int
    answer: str
    extracted_query: ExtractedQuery
    resolved_location: dict | None = None
    num_ppd_matches: int
    num_hpi_matches: int
    total_cost_usd: float
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    relevance_explanation: str


class FeedbackRequest(BaseModel):
    request_id: int
    score: Literal[1, -1]


class PromptLogEntry(BaseModel):
    id: int
    created_at: str
    question: str
    answer: str | None = None
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    feedback_positive: int = 0
    feedback_negative: int = 0
    judge_relevance: str | None = None
    judge_explanation: str | None = None


class LLMCallMetricEntry(BaseModel):
    id: int
    request_id: str
    created_at: str
    call_type: str
    model: str
    prompt: str
    instructions: str
    answer: str | None = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_s: float
    cost_usd: float


class DashboardSummary(BaseModel):
    total_conversations: int
    avg_response_time_s: float
    total_cost_usd: float
    avg_tokens: float
    positive_feedback: int = 0
    negative_feedback: int = 0
    relevant_count: int = 0
    partly_relevant_count: int = 0
    non_relevant_count: int = 0


class ConversationMetricPoint(BaseModel):
    id: int
    created_at: str
    cost_usd: float
    response_time_s: float
