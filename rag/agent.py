import sqlite3
import time

from utils.evaluation_utils import LLMCallRecord, RAGWithUsage, llm_structured_retry
from rag.judge import JUDGE_INSTRUCTIONS, JUDGE_PROMPT_TEMPLATE, RelevanceVerdict
from rag.resolver import LocationResolver
from rag.retriever import fetch_hpi_rows, fetch_ppd_rows
from rag.schemas import ExtractedQuery

INSTRUCTIONS = """
You are a UK property buying assistant. Answer the user's question about a
property or area using ONLY the CONTEXT provided below, which contains real
historical Price Paid Data sales and House Price Index trend statistics.

Ground your answer in the numbers given - cite specific sale prices, dates,
and trend percentages where relevant. If the context contains no matching
records, say so plainly rather than guessing or making up figures.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

EXTRACTION_INSTRUCTIONS = """
Extract structured location and property filters from a UK property
question. Only fill fields the user actually mentioned or implied - leave
everything else null. street/locality/town/district/county should be
copied as plain text as the user wrote them (don't invent values).

For postcodes: if the user gives a complete postcode (e.g. "BR6 1AA",
"SE19 3NF"), put it in `postcode`. If they only give a partial postcode /
postcode area or district (e.g. "BR6", "SE19", "properties in the SE19
area"), put it in `postcode_prefix` instead - never guess or complete a
partial postcode into a full one.

property_type should be one of: detached, semi-detached, terraced, flat,
other - or null if unspecified.

to fetch HPI information when postcode is provided, firt determine the region (also called district) from post code 
and use this region info to derove HPI data.
""".strip()


ESTATE_TYPE_LABELS = {"F": "Freehold", "L": "Leasehold"}


def _format_ppd_row(row: sqlite3.Row) -> str:
    # saon = secondary addressable object name (e.g. flat/unit within a
    # building), paon = primary addressable object name (house number or
    # name) - shown together so a specific unit within a building is
    # distinguishable from the building itself.
    address_parts = [
        p for p in (row["saon"], row["paon"], row["street"], row["locality"], row["town"]) if p
    ]
    address = " ".join(address_parts)
    new_build = "new build" if row["new_build"] == "Y" else "existing"
    estate_type = ESTATE_TYPE_LABELS.get(row["estate_type"], row["estate_type"])
    return (
        f"- {row['deed_date']} | £{row['price_paid']:,} | {row['property_type']} "
        f"({new_build}, {estate_type}) | {address}, {row['district']}, {row['county']}, "
        f"{row['postcode']}"
    )


def _format_hpi_row(row: sqlite3.Row) -> str:
    return (
        f"- {row['period_date']} | {row['region_name']} | avg price £{row['average_price']:,.0f} "
        f"| 1m change {row['change_1m_pct']}% | 12m change {row['change_12m_pct']}%"
    )


class PropertyRAGAgent(RAGWithUsage):
    """Estate-agent RAG: extract entities -> resolve locations -> SQL retrieve
    -> build context -> call the OpenAI LLM.

    Subclasses RAGWithUsage (itself RAGBase) to reuse build_prompt/llm/rag/
    cost-tracking as-is; only search() and build_context() are overridden.
    `index` is repurposed to mean the SQLite connection - the durable store
    this agent fetches context from - rather than a minsearch Index.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        llm_client,
        resolver: LocationResolver,
        model: str = "gpt-5.4-mini",
        extraction_model: str = "gpt-5.4-mini",
        ppd_limit: int = 20,
        hpi_limit: int = 20,
    ):
        super().__init__(
            index=conn,
            llm_client=llm_client,
            instructions=INSTRUCTIONS,
            prompt_template=PROMPT_TEMPLATE,
            course=None,
            model=model,
        )
        self.resolver = resolver
        self.extraction_model = extraction_model
        self.ppd_limit = ppd_limit
        self.hpi_limit = hpi_limit
        self.last_extracted: ExtractedQuery | None = None
        self.last_resolved: dict | None = None
        self.last_search_results: dict | None = None

    def extract(self, query: str) -> ExtractedQuery:
        start_time = time.time()
        extracted, usage = llm_structured_retry(
            self.llm_client,
            EXTRACTION_INSTRUCTIONS,
            query,
            ExtractedQuery,
            model=self.extraction_model,
        )
        response_time = time.time() - start_time

        self._record_call(
            model=self.extraction_model,
            prompt=query,
            instructions=EXTRACTION_INSTRUCTIONS,
            answer=extracted.model_dump_json(),
            usage=usage,
            response_time=response_time,
            call_type="extraction",
        )
        return extracted

    def judge_relevance(self, query: str, answer: str) -> tuple[RelevanceVerdict, LLMCallRecord]:
        prompt = JUDGE_PROMPT_TEMPLATE.format(question=query, answer=answer)
        start_time = time.time()
        verdict, usage = llm_structured_retry(
            self.llm_client,
            JUDGE_INSTRUCTIONS,
            prompt,
            RelevanceVerdict,
            model=self.extraction_model,
        )
        response_time = time.time() - start_time

        call_record = self._record_call(
            model=self.extraction_model,
            prompt=prompt,
            instructions=JUDGE_INSTRUCTIONS,
            answer=verdict.model_dump_json(),
            usage=usage,
            response_time=response_time,
            call_type="judge",
        )
        return verdict, call_record

    def resolve_location(self, extracted: ExtractedQuery) -> dict:
        resolved_ppd = None
        for field in ("town", "district", "county", "locality"):
            text = getattr(extracted, field)
            if text:
                candidate = self.resolver.resolve_best(text, field=field)
                if candidate:
                    resolved_ppd = candidate
                    break

        region_text = extracted.town or extracted.district or extracted.county or extracted.locality
        resolved_region = self.resolver.resolve_best(region_text, field="region_name") if region_text else None

        return {"ppd": resolved_ppd, "region": resolved_region}

    def search(self, query: str, num_results: int = 20) -> dict:
        extracted = self.extract(query)
        resolved = self.resolve_location(extracted)

        ppd_rows = fetch_ppd_rows(self.index, extracted, resolved["ppd"], limit=self.ppd_limit)
        hpi_rows = fetch_hpi_rows(self.index, extracted, resolved["region"], limit=self.hpi_limit)

        self.last_extracted = extracted
        self.last_resolved = resolved
        self.last_search_results = {"ppd_rows": ppd_rows, "hpi_rows": hpi_rows}
        return self.last_search_results

    def build_context(self, search_results: dict) -> str:
        ppd_rows = search_results.get("ppd_rows", [])
        hpi_rows = search_results.get("hpi_rows", [])

        sections = []

        if ppd_rows:
            sections.append(
                "## Recent Property Sales\n" + "\n".join(_format_ppd_row(r) for r in ppd_rows)
            )
        else:
            sections.append("## Recent Property Sales\nNo matching sales records found.")

        if hpi_rows:
            sections.append(
                "## House Price Index Trend\n" + "\n".join(_format_hpi_row(r) for r in hpi_rows)
            )
        else:
            sections.append("## House Price Index Trend\nNo matching regional index data found.")

        return "\n\n".join(sections)
