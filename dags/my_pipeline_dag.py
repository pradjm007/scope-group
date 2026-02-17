from __future__ import annotations

import os
from datetime import datetime
from airflow.decorators import dag, task
from airflow.sensors.python import PythonSensor
from pathlib import Path
import importlib.util

# Adjust these via environment in Airflow
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/opt/airflow/upload_folder")
TRACKER_FOLDER = os.getenv("TRACKER_FOLDER", "/opt/airflow/files_tracker")
DB_URL = os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@postgres:5432/mydatabase")


def has_xlsm_files() -> bool:
    """Module-level callable used by the sensor to detect any .xlsm files.

    Placing this at module level ensures Airflow can serialize/import it cleanly.
    """
    p = Path(UPLOAD_FOLDER)
    try:
        if not p.exists():
            return False
        return any(p.glob("*.xlsm"))
    except Exception:
        return False


@dag(
    schedule="*/10 * * * *",   # every 10 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["file_trigger"]
)
def script_pipeline():

    wait_for_file = PythonSensor(
        task_id="wait_for_file",
        python_callable=has_xlsm_files,
        poke_interval=10,   # check every 10 seconds
        timeout=60,         # stop after 1 minute
        mode="reschedule",  # recommended so it doesn’t block a worker
    )

    @task
    def run_uploader_task() -> list[str]:
        # Simple import by file path and call
    # Use the project path where your scripts live in the Airflow container
        uploader_path = "/opt/airflow/project/0_raw_uploader_and_track.py"
        spec = importlib.util.spec_from_file_location("uploader", uploader_path)
        uploader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uploader)

        processed = uploader.process_upload_folder(Path(UPLOAD_FOLDER), DB_URL, Path(TRACKER_FOLDER))
        return processed or []

    @task
    def run_parser_task(file_paths: list[str]) -> None:
        if not file_paths:
            print("No files to parse")
            return

    # Use the project path where your scripts live in the Airflow container
        parser_path = "/opt/airflow/project/1_extract_and_process.py"
        spec = importlib.util.spec_from_file_location("parser", parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)

        import psycopg2
        conn = psycopg2.connect(DB_URL)
        try:
            for p in file_paths:
                print(f"Parsing: {p}")
                parser.process_file_and_store(str(p), conn)
        finally:
            conn.close()

    uploader_task = run_uploader_task()
    wait_for_file >> uploader_task
    run_parser_task(uploader_task)



dag = script_pipeline()
