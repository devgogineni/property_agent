import re
import sqlite3

from rag.schemas import ExtractedQuery

UK_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$")

PROPERTY_TYPE_ALIASES = {
    "detached": "D", "d": "D",
    "semi-detached": "S", "semi detached": "S", "semi": "S", "s": "S",
    "terraced": "T", "terrace": "T", "t": "T",
    "flat": "F", "flats": "F", "apartment": "F", "f": "F",
    "other": "O", "o": "O",
}


def _normalize_property_type(value: str | None) -> str | None:
    if not value:
        return None
    return PROPERTY_TYPE_ALIASES.get(value.strip().lower())


def fetch_ppd_rows(
    conn: sqlite3.Connection,
    query: ExtractedQuery,
    resolved: dict | None,
    limit: int = 200,
) -> list[sqlite3.Row]:
    clauses = []
    params: list = []

    postcode_text = None
    if query.postcode:
        postcode_text = query.postcode.strip().upper()
    elif query.postcode_prefix:
        postcode_text = query.postcode_prefix.strip().upper()[:3]

    if postcode_text:
        if UK_POSTCODE_RE.match(postcode_text):
            clauses.append("postcode = ?")
            params.append(postcode_text)
        else:
            clauses.append("postcode LIKE ?")
            params.append(f"{postcode_text}%")

    if query.street:
        clauses.append("street LIKE ?")
        params.append(f"%{query.street.strip().upper()}%")

    # UK addresses conflate these levels in colloquial use (e.g. "Croydon" is
    # commonly called a town but PPD files it under `district`, with
    # `town=LONDON`) - AND-ing each field independently misses real matches
    # where the extracted/resolved value landed on a different column than
    # expected. OR them together instead: match if ANY level agrees.
    geo_terms = set()
    for field in ("town", "district", "county", "locality"):
        value = getattr(query, field)
        if resolved and resolved.get("field") == field:
            value = resolved["value"]
        if value:
            geo_terms.add(value.strip().upper())

    if geo_terms:
        geo_terms = list(geo_terms)
        placeholders = ", ".join("?" for _ in geo_terms)
        geo_clause = " OR ".join(
            f"{field} IN ({placeholders})" for field in ("town", "district", "county", "locality")
        )
        clauses.append(f"({geo_clause})")
        params.extend(geo_terms * 4)

    property_type = _normalize_property_type(query.property_type)
    if property_type:
        clauses.append("property_type = ?")
        params.append(property_type)

    if query.new_build is not None:
        clauses.append("new_build = ?")
        params.append("Y" if query.new_build else "N")

    if query.min_price is not None:
        clauses.append("price_paid >= ?")
        params.append(query.min_price)

    if query.max_price is not None:
        clauses.append("price_paid <= ?")
        params.append(query.max_price)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT unique_id, price_paid, deed_date, postcode, property_type, new_build,
               estate_type, saon, paon, street, locality, town, district, county
        FROM ppd
        {where}
        ORDER BY deed_date DESC
        LIMIT ?
    """
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def fetch_hpi_rows(
    conn: sqlite3.Connection,
    query: ExtractedQuery,
    resolved_region: dict | None,
    limit: int = 3,
) -> list[sqlite3.Row]:
    region_name = resolved_region["value"] if resolved_region else None
    if not region_name:
        region_name = query.county or query.district or query.town

    if not region_name:
        return []

    sql = """
        SELECT region_name, area_code, period_date, average_price, price_index,
               change_1m_pct, change_12m_pct, sales_volume,
               detached_price, semi_detached_price, terraced_price, flat_price
        FROM hpi
        WHERE region_name = ?
        ORDER BY period_date DESC
        LIMIT ?
    """
    return conn.execute(sql, [region_name.strip(), limit]).fetchall()
