import argparse

import pandas as pd

import config
from db.connection import create_indexes, create_tables, get_connection
from utils.data_validation_utils import clean_hpi_dataframe

HPI_COLUMN_MAP = {
    "Date": "period_date",
    "RegionName": "region_name",
    "AreaCode": "area_code",
    "AveragePrice": "average_price",
    "Index": "price_index",
    "IndexSA": "index_sa",
    "1m%Change": "change_1m_pct",
    "12m%Change": "change_12m_pct",
    "AveragePriceSA": "average_price_sa",
    "SalesVolume": "sales_volume",
    "DetachedPrice": "detached_price",
    "DetachedIndex": "detached_index",
    "Detached1m%Change": "detached_change_1m_pct",
    "Detached12m%Change": "detached_change_12m_pct",
    "SemiDetachedPrice": "semi_detached_price",
    "SemiDetachedIndex": "semi_detached_index",
    "SemiDetached1m%Change": "semi_detached_change_1m_pct",
    "SemiDetached12m%Change": "semi_detached_change_12m_pct",
    "TerracedPrice": "terraced_price",
    "TerracedIndex": "terraced_index",
    "Terraced1m%Change": "terraced_change_1m_pct",
    "Terraced12m%Change": "terraced_change_12m_pct",
    "FlatPrice": "flat_price",
    "FlatIndex": "flat_index",
    "Flat1m%Change": "flat_change_1m_pct",
    "Flat12m%Change": "flat_change_12m_pct",
    "CashPrice": "cash_price",
    "CashIndex": "cash_index",
    "Cash1m%Change": "cash_change_1m_pct",
    "Cash12m%Change": "cash_change_12m_pct",
    "CashSalesVolume": "cash_sales_volume",
    "MortgagePrice": "mortgage_price",
    "MortgageIndex": "mortgage_index",
    "Mortgage1m%Change": "mortgage_change_1m_pct",
    "Mortgage12m%Change": "mortgage_change_12m_pct",
    "MortgageSalesVolume": "mortgage_sales_volume",
    "FTBPrice": "ftb_price",
    "FTBIndex": "ftb_index",
    "FTB1m%Change": "ftb_change_1m_pct",
    "FTB12m%Change": "ftb_change_12m_pct",
    "FOOPrice": "foo_price",
    "FOOIndex": "foo_index",
    "FOO1m%Change": "foo_change_1m_pct",
    "FOO12m%Change": "foo_change_12m_pct",
    "NewPrice": "new_price",
    "NewIndex": "new_index",
    "New1m%Change": "new_change_1m_pct",
    "New12m%Change": "new_change_12m_pct",
    "NewSalesVolume": "new_sales_volume",
    "OldPrice": "old_price",
    "OldIndex": "old_index",
    "Old1m%Change": "old_change_1m_pct",
    "Old12m%Change": "old_change_12m_pct",
    "OldSalesVolume": "old_sales_volume",
}

NUMERIC_COLUMNS = [v for k, v in HPI_COLUMN_MAP.items() if k not in ("Date", "RegionName", "AreaCode")]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df = df.rename(columns=HPI_COLUMN_MAP)

    df["period_date"] = pd.to_datetime(df["period_date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return clean_hpi_dataframe(df)


def ingest(csv_path: str, db_path: str) -> int:
    df = load_and_clean(csv_path)

    conn = get_connection(db_path)
    try:
        create_tables(conn)
        conn.execute("DELETE FROM hpi")
        df.to_sql("hpi", conn, if_exists="append", index=False)
        create_indexes(conn)
        conn.commit()
        return conn.execute("SELECT COUNT(*) FROM hpi").fetchone()[0]
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the UK HPI dataset into SQLite")
    parser.add_argument("--csv", default=config.HPI_CSV_PATH)
    parser.add_argument("--db", default=config.DB_PATH)
    args = parser.parse_args()

    count = ingest(args.csv, args.db)
    print(f"Ingested {count} hpi rows into {args.db}")


if __name__ == "__main__":
    main()
