import argparse
import os
import pandas as pd
from typing import Optional
import psycopg2
from pathlib import Path



pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)      # allow wide output
pd.set_option("display.max_colwidth", None)

def load_master_sheet(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        sheet_name="MASTER",
        header=None,          # very important (no structured header)
        engine="openpyxl"
    )
    
    # Remove fully empty rows
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    
    return df

def parse_key_value_section(df: pd.DataFrame) -> pd.DataFrame:
    metadata = {}

    # Fields that may have multiple values in same row
    multi_value_fields = {
        "Rating methodologies applied",
        "Industry risk",
        "Industry risk score",
        "Industry weight"
    }

    for _, row in df.iterrows():
        label = row[1]

        # Stop when metrics table begins
        if isinstance(label, str) and "[Scope Credit Metrics]" in label:
            break

        if pd.isna(label):
            continue

        label = str(label).strip()

        # Collect all non-null values from column 2 onward
        row_values = row[2:]
        non_null_values = [
            v for v in row_values
            if pd.notna(v) and str(v).strip() != ""
        ]

        if label in multi_value_fields:
            # Store as list (preserves order)
            if label == "Industry weight":
                non_null_values = [f"{str(int(float(v)*100))}%" for v in non_null_values]
            metadata[label] = non_null_values
        else:
            # Single value expected
            metadata[label] = non_null_values[0] if non_null_values else None

    metadata_df = pd.DataFrame([metadata])
    return metadata_df

def parse_credit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Find metrics start
    metrics_start_idx = df[df[1].astype(str).str.contains(r"\[Scope Credit Metrics\]", na=False)].index[0]
    
    # Extract header row (years)
    header_row = df.iloc[metrics_start_idx]
    
    years = header_row[2:]   # Years start from column 2
    years = years.dropna().tolist()
    
    records = []
    
    # Loop through rows below header
    for i in range(metrics_start_idx + 1, len(df)):
        row = df.iloc[i]
        metric_name = row[1]
        
        if pd.isna(metric_name):
            continue
        
        values = row[2:2+len(years)]
        
        for year, value in zip(years, values):
            if pd.notna(value) and value != "Locked":
                records.append({
                    "metric_name": metric_name,
                    "year": year,
                    "value": value
                })
    
    metrics_df = pd.DataFrame(records)
    return metrics_df


### Database helper functions (write to Postgres)

def ensure_master_tables(conn: psycopg2.extensions.connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company (
            company_id INTEGER PRIMARY KEY,
            entity_name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            sector TEXT,
            country TEXT,
            currency TEXT,
            accounting_principles TEXT,
            business_year TEXT,
            UNIQUE (entity_name, version)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submission (
            submission_id SERIAL PRIMARY KEY,
            company_id INTEGER NOT NULL REFERENCES company(company_id),
            version INTEGER NOT NULL,
            upload_timestamp TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rating_submission (
            submission_id INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            industry_risk_score TEXT,
            business_risk_profile TEXT,
            financial_risk_profile TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS submission_methodology (
            submission_id INTEGER NOT NULL,
            methodology_name TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS business_risk_detail (
            submission_id INTEGER NOT NULL,
            business_risk_profile TEXT,
            blended_industry_risk_profile TEXT,
            competitive_positioning TEXT,
            market_share TEXT,
            diversification TEXT,
            operating_profitability TEXT,
            sector_company_specific_factors_1 TEXT,
            sector_company_specific_factors_2 TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS financial_risk_detail (
            submission_id INTEGER NOT NULL,
            financial_risk_profile TEXT,
            leverage TEXT,
            interest_cover TEXT,
            cash_flow_cover TEXT,
            liquidity TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS industry_risk (
            submission_id INTEGER NOT NULL,
            industry_name TEXT,
            score TEXT,
            weight TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS credit_metrics (
            submission_id INTEGER NOT NULL,
            metric_name TEXT,
            year INTEGER,
            value REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_history (
            history_id SERIAL PRIMARY KEY,
            company_id INTEGER NOT NULL,
            entity_name TEXT,
            version INTEGER,
            sector TEXT,
            country TEXT,
            currency TEXT,
            accounting_principles TEXT,
            business_year TEXT,
            change_type TEXT,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def get_or_create_company_id(conn: psycopg2.extensions.connection, entity_name: str, version: int = 1, company_info: Optional[dict] = None) -> int:
    """Get company_id for a given entity_name.

    Lookup is performed only by entity_name (ignoring version). If any row exists for
    the entity_name we reuse that company_id. Otherwise we allocate a new company_id
    (max(company_id)+1) and insert a new row with the provided version.
    """
    cur = conn.cursor()
    # 1) Look for an existing company_id in company_history by entity_name (latest entry)
    cur.execute(
        "SELECT company_id FROM company_history WHERE entity_name = %s ORDER BY changed_at DESC LIMIT 1",
        (entity_name,)
    )
    hist_row = cur.fetchone()
    history_exists = hist_row is not None
    if history_exists:
        company_id = hist_row[0]
    else:
        # No history entry: allocate a new company_id based on company_history max
        cur.execute("SELECT MAX(company_id) FROM company_history")
        row = cur.fetchone()
        max_id = row[0] if row and row[0] is not None else 0
        company_id = int(max_id) + 1

    # 2) Insert a new history record for this company snapshot (insert vs update)
    change_type = 'update' if history_exists else 'insert'
    sector = company_info.get("sector") if company_info else None
    country = company_info.get("country") if company_info else None
    currency = company_info.get("currency") if company_info else None
    accounting_principles = company_info.get("accounting_principles") if company_info else None
    business_year = company_info.get("business_year") if company_info else None
    try:
        cur.execute(
            "INSERT INTO company_history (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year, change_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year, change_type)
        )
        conn.commit()
    except Exception:
        # If history insert fails for some reason, continue but don't block the flow
        pass

    # 3) Upsert the company table: update if company_id exists, otherwise insert
    cur.execute("SELECT company_id FROM company WHERE company_id = %s", (company_id,))
    existing_company = cur.fetchone()
    if existing_company:
        # Update existing company row: set version and any provided non-None metadata fields
        updates = ["version = %s"]
        params = [version]
        if company_info:
            field_map = {
                "sector": "sector",
                "country": "country",
                "currency": "currency",
                "accounting_principles": "accounting_principles",
                "business_year": "business_year",
            }
            for k, col in field_map.items():
                v = company_info.get(k)
                if v is not None:
                    updates.append(f"{col} = %s")
                    params.append(v)
        params.append(company_id)
        cur.execute(f"UPDATE company SET {', '.join(updates)} WHERE company_id = %s", tuple(params))
        conn.commit()
    else:
        # Insert a new company row with the computed company_id
        cur.execute(
            "INSERT INTO company (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year),
        )
        conn.commit()

    return company_id
    sector = company_info.get("sector") if company_info else None
    country = company_info.get("country") if company_info else None
    currency = company_info.get("currency") if company_info else None
    accounting_principles = company_info.get("accounting_principles") if company_info else None
    business_year = company_info.get("business_year") if company_info else None
    cur.execute(
        "INSERT INTO company (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (new_id, entity_name, version, sector, country, currency, accounting_principles, business_year),
    )
    conn.commit()
    # Record history for the newly created company row
    try:
        cur.execute(
            "INSERT INTO company_history (company_id, entity_name, version, sector, country, currency, accounting_principles, business_year, change_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (new_id, entity_name, version, sector, country, currency, accounting_principles, business_year, 'insert')
        )
        conn.commit()
    except Exception:
        # If history table is not available for some reason, ignore but keep the company row
        pass
    return new_id


def create_submission(conn: psycopg2.extensions.connection, company_id: int, version: int, upload_ts: str) -> int:
    cur = conn.cursor()
    # Use SERIAL/sequence and RETURNING to get an atomic submission_id
    cur.execute("INSERT INTO submission (company_id, version, upload_timestamp) VALUES (%s, %s, %s) RETURNING submission_id", (company_id, version, upload_ts))
    row = cur.fetchone()
    conn.commit()
    return int(row[0])


def insert_rating_submission(conn: psycopg2.extensions.connection, submission_id: int, company_id: int, version: int, industry_risk_score: Optional[float] = None, business_risk_profile: Optional[str] = None, financial_risk_profile: Optional[str] = None) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO rating_submission (submission_id, company_id, version, industry_risk_score, business_risk_profile, financial_risk_profile) VALUES (%s, %s, %s, %s, %s, %s)", (submission_id, company_id, version, industry_risk_score, business_risk_profile, financial_risk_profile))
    conn.commit()


def insert_submission_methodology(conn: psycopg2.extensions.connection, submission_id: int, methodology_name: str) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO submission_methodology (submission_id, methodology_name) VALUES (%s, %s)", (submission_id, methodology_name))
    conn.commit()


def insert_business_risk_detail(conn: psycopg2.extensions.connection, submission_id: int, business_risk_profile: str, blended_industry_risk_profile: str, competitive_positioning: str, market_share: str, diversification: str, operating_profitability: str, sector_company_specific_factors_1: str, sector_company_specific_factors_2: str) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO business_risk_detail (submission_id, business_risk_profile, blended_industry_risk_profile, competitive_positioning, market_share, diversification, operating_profitability, sector_company_specific_factors_1, sector_company_specific_factors_2) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (submission_id, business_risk_profile, blended_industry_risk_profile, competitive_positioning, market_share, diversification, operating_profitability, sector_company_specific_factors_1, sector_company_specific_factors_2))
    conn.commit()


def insert_financial_risk_detail(conn: psycopg2.extensions.connection, submission_id: int, financial_risk_profile: str, leverage: str, interest_cover: str, cash_flow_cover: str, liquidity: str) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO financial_risk_detail (submission_id, financial_risk_profile, leverage, interest_cover, cash_flow_cover, liquidity) VALUES (%s, %s, %s, %s, %s, %s)", (submission_id, financial_risk_profile, leverage, interest_cover, cash_flow_cover, liquidity))
    conn.commit()


def insert_industry_risk(conn: psycopg2.extensions.connection, submission_id: int, industry_name: str, score: Optional[float], weight: Optional[float]) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO industry_risk (submission_id, industry_name, score, weight) VALUES (%s, %s, %s, %s)", (submission_id, industry_name, score, weight))
    conn.commit()


def insert_credit_metric(conn: psycopg2.extensions.connection, submission_id: int, metric_name: str, year: Optional[int], value: Optional[float]) -> None:
    cur = conn.cursor()
    cur.execute("INSERT INTO credit_metrics (submission_id, metric_name, year, value) VALUES (%s, %s, %s, %s)", (submission_id, metric_name, year, value))
    conn.commit()





def process_file(file_path: str) -> None:
    """Load an Excel file, parse metadata and credit metrics, write results to Postgres and print summary.

    The DB connection is provided via `conn` when called from `main`.
    """
    # kept for backward-compatibility when called standalone without DB
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {file_path}")
        return
    df = load_master_sheet(str(p))
    metadata_df = parse_key_value_section(df)
    credit_metrics_df = parse_credit_metrics(df)

    print(f"\n--- Parsed file: {file_path} ---\n")
    print("Metadata:")
    print(metadata_df.to_string(index=False))
    print("\nCredit metrics:")
    print(credit_metrics_df.to_string(index=False))
    print("complete")


def process_file_and_store(file_path: str, conn: psycopg2.extensions.connection) -> None:
    """Parse file and store parsed results into Postgres using helpers defined below."""
    p = Path(file_path)
    if not p.exists():
        print(f"File not found: {file_path}")
        return

    df = load_master_sheet(str(p))
    metadata_df = parse_key_value_section(df)
    credit_metrics_df = parse_credit_metrics(df)

    # Flatten metadata row to a dict
    metadata = metadata_df.iloc[0].to_dict() if not metadata_df.empty else {}

    # Determine original filename, company_name and version from filename if available
    original_name = p.name
    stem = p.stem
    version = 1
    company_name = stem
    if "_" in stem:
        last = stem.rsplit("_", 1)[-1]
        if last.isdigit():
            version = int(last)
            company_name = stem.rsplit("_", 1)[0]

    upload_ts = pd.Timestamp.utcnow().isoformat()

    # Ensure master tables exist
    ensure_master_tables(conn)

    # Get or create company_id
    # Allow metadata to provide entity_name override if present
    entity_name_key = None
    for k in metadata.keys():
        if isinstance(k, str) and k.strip().lower() in ("entity name", "entity_name", "company", "company name"):
            entity_name_key = k
            break

    entity_name = metadata.get(entity_name_key) if entity_name_key else company_name
    company_info = {}
    # try to map some standard fields from metadata (case-insensitive)
    def get_meta_field(name_variants):
        for v in name_variants:
            for k in metadata.keys():
                if isinstance(k, str) and k.strip().lower() == v.strip().lower():
                    return metadata.get(k)
        return None
    company_info["sector"] = get_meta_field(["CorporateSector"]) or None
    company_info["country"] = get_meta_field(["Country of origin"]) or None
    company_info["currency"] = get_meta_field(["Reporting Currency/Units"]) or None
    company_info["accounting_principles"] = get_meta_field(["accounting principles"]) or None
    company_info["business_year"] = get_meta_field(["End of business year"]) or None

    company_id = get_or_create_company_id(conn, entity_name, version, company_info)

    # Create submission (auto incremented id) and record upload timestamp on submission
    submission_id = create_submission(conn, company_id, version, upload_ts)

    # Insert rating_submission summary (try to get some fields)
    # Helper to coerce possibly-list metadata into a scalar float or string
    def _first_or_none(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return x[0] if x else None
        return x

    def _to_float_or_none(x):
        val = _first_or_none(x)
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None


    industry_risk_score = metadata.get("Industry risk score")
    business_risk_profile = _first_or_none(metadata.get("Business risk profile") or metadata.get("Business risk"))
    financial_risk_profile = _first_or_none(metadata.get("Financial risk profile") or metadata.get("Financial risk"))

    insert_rating_submission(conn, submission_id, company_id, version, industry_risk_score, business_risk_profile, financial_risk_profile)

    # Methodologies
    methodologies = metadata.get("Rating methodologies applied") or metadata.get("Rating methodologies")
    if methodologies is not None:
        if not isinstance(methodologies, (list, tuple)):
            methodologies = [methodologies]
        for m in methodologies:
            insert_submission_methodology(conn, submission_id, str(m))

    # Industry risk details: try to align Industry risk, Industry risk score, Industry weight
    ind_names = metadata.get("Industry risk")
    ind_scores = metadata.get("Industry risk score")
    ind_weights = metadata.get("Industry weight")
    if ind_names is not None:
        # ensure lists
        if not isinstance(ind_names, (list, tuple)):
            ind_names = [ind_names]
        if ind_scores is None:
            ind_scores = [None] * len(ind_names)
        elif not isinstance(ind_scores, (list, tuple)):
            ind_scores = [ind_scores]
        if ind_weights is None:
            ind_weights = [None] * len(ind_names)
        elif not isinstance(ind_weights, (list, tuple)):
            ind_weights = [ind_weights]

        for name, score, weight in zip(ind_names, ind_scores, ind_weights):
            try:
                s = score if score is not None else None
            except Exception:
                s = None
            try:
                w = weight if weight is not None else None
            except Exception:
                w = None
            insert_industry_risk(conn, submission_id, str(name), str(s), str(w))

    def map_fields(fields, metadata):
        dict_out = {}
        for field in fields:
            dict_out[field.translate(str.maketrans({"(": "", ")": "", "/": "_", "-": "_", " ": "_"})).lower()] = metadata.get(field)
        return dict_out

    business_risk_fields  = [
    "Business risk profile",
    "(Blended) Industry risk profile",
    "Competitive Positioning",
    "Market share",
    "Diversification",
    "Operating profitability",
    "Sector/company-specific factors (1)",
    "Sector/company-specific factors (2)"
    ]

    financial_risk_fields  = [
    "Financial risk profile",
    "Leverage",
    "Interest cover",
    "Cash flow cover",
    "Liquidity"
    ]

    financial_risk_profile_data = map_fields(financial_risk_fields, metadata)
    business_risk_profile_data = map_fields(business_risk_fields, metadata)

    insert_business_risk_detail(conn, submission_id, **business_risk_profile_data)
    insert_financial_risk_detail(conn, submission_id, **financial_risk_profile_data)

    # Credit metrics: insert rows
    def _parse_year_to_int(y) -> Optional[int]:
        if pd.isna(y) or y is None:
            return None
        # If it's already an int
        if isinstance(y, int):
            return y
        # If it's a float that is an integer value (e.g., 2020.0)
        if isinstance(y, float):
            try:
                if float(y).is_integer():
                    return int(y)
            except Exception:
                return None
        # Try to parse numeric forms in string
        s = str(y).strip()
        # Try direct int
        try:
            return int(s)
        except Exception:
            pass
        # Try float then int if whole number (handles '2020.0')
        try:
            f = float(s.replace(",", ""))
            if f.is_integer():
                return int(f)
        except Exception:
            pass
        # Extract 4-digit year (e.g., '2025E' -> 2025)
        import re

        m = re.search(r"(19|20)\d{2}", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None

    for _, row in credit_metrics_df.iterrows():
        metric_name = row.get("metric_name")
        year = row.get("year")
        value = row.get("value")
        try:
            val = float(value)
        except Exception:
            try:
                val = float(str(value).replace(",", ""))
            except Exception:
                val = None
        parsed_year = _parse_year_to_int(year)
        if metric_name is not None:
            insert_credit_metric(conn, submission_id, str(metric_name), parsed_year, val)

    print(f"Stored to DB: submission_id={submission_id}, company_id={company_id}, version={version}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse one or more Excel files and print extracted tables")
    p.add_argument("files", nargs="+", help="Path(s) to Excel .xlsm file(s) to parse")
    p.add_argument(
        "--db-url",
        "-d",
        type=str,
        default=os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@localhost:5432/mydatabase"),
        help="Postgres DSN/URL (default: from DATABASE_URL or postgresql://myuser:mypassword@localhost:5432/mydatabase)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Connect to Postgres and ensure master tables
    db_url = args.db_url
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        ensure_master_tables(conn)
    except Exception as exc:
        print(f"Failed to connect or initialize DB: {exc}")
        conn = None

    for f in args.files:
        if conn:
            try:
                process_file_and_store(f, conn)
            except Exception as exc:
                print(f"Error processing and storing {f}: {exc}")
        else:
            process_file(f)

    if conn:
        conn.close()


if __name__ == "__main__":
    main()