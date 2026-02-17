"""Simple uploader that reads .xlsm files from an upload folder and
tracks them in a Postgres `files_history_table` table.

Schema (files_history_table):
	file_name TEXT
	company_name TEXT
	upload_timestamp TEXT (ISO 8601)
	version INTEGER

Behavior:
 - For each .xlsm file in the upload folder, check whether `file_name` exists
	 in the table. If not present, insert a record with version=1.
 - If present, increment the version (max(version)+1) and insert a new record.

This script uses psycopg2 to talk to Postgres and moves processed files to
a tracker folder. It also provides a small CLI for folder/db configuration.
"""

from __future__ import annotations

import argparse
import os
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import subprocess
import sys


TABLE_NAME = "files_history_table"


def ensure_table(conn: Any) -> None:
	"""Create files_history_table if it does not exist (Postgres).

	Uses a DB-API connection (psycopg2) and commits the DDL.
	"""
	cur = conn.cursor()
	cur.execute(
		f"""
		CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
			id SERIAL PRIMARY KEY,
			original_name TEXT NOT NULL,
			company_name TEXT NOT NULL,
			upload_timestamp TEXT NOT NULL,
			version INTEGER NOT NULL
		)
		"""
	)
	conn.commit()


# Version is inferred from filename (trailing _<number>), so we don't query DB for it.


def insert_record(
	conn: Any,
	original_name: str,
	company_name: str,
	upload_ts: str,
	version: int,
) -> None:
	cur = conn.cursor()
	cur.execute(
		f"INSERT INTO {TABLE_NAME} (original_name, company_name, upload_timestamp, version) VALUES (%s, %s, %s, %s)",
		(original_name, company_name, upload_ts, version),
	)
	conn.commit()


def process_file(conn: Any, file_path: Path, tracker_folder: Path) -> None:
	"""Process a single file: infer version and company_name from filename, copy file to tracker, insert record, and call parser.

	Expected filename format for versioning: <company>_<maybe_more>_<version> (version is the integer after the last underscore).
	Company name is defined as everything except the final '_<number>' segment. Example: company_A_2 -> company_A (company_name) and version 2.
	"""
	original_name = file_path.name
	# Determine version from filename: trailing _<number> on the stem. Example: company_A_2 -> version 2
	upload_ts = datetime.utcnow().isoformat()


	stem = file_path.stem  # filename without suffix
	version = 1
	company_name = stem
	if "_" in stem:
		last = stem.rsplit("_", 1)[-1]
		if last.isdigit():
			version = int(last)
			company_name = stem.rsplit("_", 1)[0]

	print(f"Processing {original_name} (inferred version={version}, company_name={company_name})")

	# Ensure tracker folder exists and copy the file preserving its original name
	tracker_folder.mkdir(parents=True, exist_ok=True)
	dest = tracker_folder / original_name
	try:
		import shutil

		shutil.move(str(file_path), str(dest))
		print(f"Moved {original_name} -> {dest}")
	except Exception as exc:
		print(f"Failed to move file {original_name} to {dest}: {exc}")

	insert_record(conn, original_name, company_name, upload_ts, version)

	# Return the destination path so callers (e.g., a DAG) can act on processed files
	return str(dest)


def process_upload_folder(upload_folder: Path, db_url: str, tracker_folder: Optional[Path] = None) -> list[str]:
	"""Iterate over .xlsm files in upload_folder and process them.

	Files are processed one by one and each upload is recorded in Postgres.
	Returns a list of destination file paths moved to the tracker folder.
	"""
	if not upload_folder.exists() or not upload_folder.is_dir():
		raise FileNotFoundError(f"Upload folder not found: {upload_folder}")

	# Connect to Postgres using the provided DSN / URL
	conn = psycopg2.connect(db_url)
	try:
		ensure_table(conn)

		files = sorted(upload_folder.glob("*.xlsm"))
		if not files:
			print("No .xlsm files found in upload folder.")
			return []

		if tracker_folder is None:
			# Default to a files_tracker directory in the current working directory
			tracker_folder = Path.cwd() / "files_tracker"

		processed: list[str] = []

		for p in files:
			# If this filename already exists in the table, move the source file to rejected and skip
			try:
				cur = conn.cursor()
				cur.execute(f"SELECT 1 FROM {TABLE_NAME} WHERE original_name = %s LIMIT 1", (p.name,))
				row = cur.fetchone()
				if row is not None:
					# Entry already exists -> move the uploaded source file to an upload_folder/rejected folder to avoid reprocessing
					try:
						rejected_folder = upload_folder / "rejected"
						rejected_folder.mkdir(parents=True, exist_ok=True)
						import shutil

						dest_rejected = rejected_folder / p.name
						shutil.move(str(p), str(dest_rejected))
						print(f"File {p.name} already recorded in DB; moved source file to rejected: {dest_rejected}")
					except Exception as rm_exc:
						print(f"Failed to move duplicate source file {p} to rejected/: {rm_exc}")
					continue

				# Not found in DB -> process normally
				try:
					dest = process_file(conn, p, tracker_folder)
					if dest:
						processed.append(dest)
				except Exception as pf_exc:
					print(f"process_file failed for {p.name}: {pf_exc}")
			except Exception as exc:
				print(f"Failed processing {p.name}: {exc}")

		return processed
	finally:
		conn.close()


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Upload .xlsm files and track history in PostgreSQL")
	p.add_argument("--upload-folder", "-u", type=Path, required=True, help="Folder containing .xlsm files")
	p.add_argument(
		"--db-url",
		"-d",
		type=str,
		default=os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@localhost:5432/mydatabase"),
		help=(
			"Postgres DSN/URL (default: from DATABASE_URL or "
			"postgresql://myuser:mypassword@localhost:5432/mydatabase)"
		),
	)
	p.add_argument(
		"--tracker-folder",
		"-t",
		type=Path,
		default=None,
		help="Folder where versioned files will be moved (default: ./files_tracker)",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	process_upload_folder(args.upload_folder, args.db_url, args.tracker_folder)


if __name__ == "__main__":
	main()

