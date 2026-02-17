from __future__ import annotations

import os
from typing import Any, Dict, List

import psycopg2
from fastapi import FastAPI, HTTPException
from typing import Optional
from fastapi import Query


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://myuser:mypassword@postgres:5432/mydatabase"
)

app = FastAPI(title="Company API")


def get_conn():
    return psycopg2.connect(DATABASE_URL)


def rows_to_dicts(cursor, rows) -> List[Dict[str, Any]]:
    cols = [c.name for c in cursor.description]
    return [{cols[i]: r[i] for i in range(len(cols))} for r in rows]


@app.get("/companies")
def list_companies(company_id: Optional[str] = Query(None)):
    conn = get_conn()
    try:
        cur = conn.cursor()

        if company_id:
            # Return latest version for specific company
            cur.execute(
                "SELECT * FROM company WHERE company_id = %s ORDER BY version DESC LIMIT 1",
                (company_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Company not found")
            return rows_to_dicts(cur, [row])[0]

        # Otherwise return all companies
        cur.execute("SELECT * FROM company")
        rows = cur.fetchall()
        return rows_to_dicts(cur, rows)

    finally:
        conn.close()




# -------------------------------------------------------
# GET /companies/versions?company_id=2
# -------------------------------------------------------
@app.get("/companies/versions")
def list_versions(company_id: Optional[str] = Query(None)):
    conn = get_conn()
    try:
        cur = conn.cursor()

        if company_id:
            cur.execute(
                """
                SELECT *
                FROM submission
                WHERE company_id = %s
                ORDER BY version DESC
                """,
                (company_id,),
            )
        else:
            cur.execute(
                """
                SELECT *
                FROM submission
                ORDER BY version DESC
                """
            )

        rows = cur.fetchall()
        return rows_to_dicts(cur, rows)

    finally:
        conn.close()


# -------------------------------------------------------
# GET /companies/history?company_id=2
# -------------------------------------------------------
@app.get("/companies/history")
def company_history(company_id: Optional[str] = Query(None)):
    conn = get_conn()
    try:
        cur = conn.cursor()

        if company_id:
            cur.execute(
                """
                SELECT *
                FROM company_history
                WHERE company_id = %s
                ORDER BY changed_at ASC
                """,
                (company_id,),
            )
        else:
            cur.execute(
                """
                SELECT *
                FROM company_history
                ORDER BY changed_at ASC
                """
            )

        rows = cur.fetchall()
        return rows_to_dicts(cur, rows)

    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
