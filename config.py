import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "estate_agent.db"))
# Separate from DB_PATH on purpose - DB_PATH is a derived artifact that gets
# deleted/rebuilt from the source CSVs (see db/connection.py), which would
# wipe prompt history if it lived in the same file.
PROMPT_LOG_DB_PATH = os.getenv(
    "PROMPT_LOG_DB_PATH", str(BASE_DIR / "data" / "prompt_log.db")
)
LOCATION_INDEX_PATH = os.getenv(
    "LOCATION_INDEX_PATH", str(BASE_DIR / "data" / "location_index.pkl")
)
EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH", str(BASE_DIR / "models" / "Xenova" / "all-MiniLM-L6-v2")
)

HPI_CSV_PATH = os.getenv(
    "HPI_CSV_PATH",
    str(BASE_DIR / "data" / "house_price_index_data" / "UK-HPI-full-file-2026-04.csv"),
)
PPD_DIR = os.getenv("PPD_DIR", str(BASE_DIR / "data" / "ppd"))

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4-mini")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-5.4-mini")

PPD_INGEST_DEFAULT_LIMIT = int(os.getenv("PPD_INGEST_DEFAULT_LIMIT", "400000"))
PPD_INGEST_CHUNKSIZE = int(os.getenv("PPD_INGEST_CHUNKSIZE", "200000"))
