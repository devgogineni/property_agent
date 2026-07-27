"""LLM-as-a-judge relevance check, following the llm-zoomcamp built-in-judge
lesson:
https://github.com/DataTalksClub/llm-zoomcamp/blob/main/05-monitoring/lessons/09-built-in-judge.md

Online judging: only the question and generated answer are available (no
reference answer to compare against), so the verdict has to reason from the
answer's content alone - the explanation field is asked for before the
label to force that reasoning.
"""

from typing import Literal

from pydantic import BaseModel

JUDGE_INSTRUCTIONS = """
You are an expert evaluator for a RAG system that answers UK property
questions grounded in real Price Paid Data sales and House Price Index
statistics.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

JUDGE_PROMPT_TEMPLATE = """
Question: {question}
Generated Answer: {answer}
""".strip()


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str
