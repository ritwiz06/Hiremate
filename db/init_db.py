import sqlite3
from pathlib import Path

# Resolve the path to a local SQLite file so no server is needed (SQLite is file-based) [web:166]
DB_PATH = Path(__file__).with_name("jobs.db")

# DDL includes:
# - Base jobs table to store postings
# - FTS5 virtual table for fast full-text search over text fields
# - Triggers to keep FTS in sync automatically on INSERT/UPDATE/DELETE [web:166]
DDL = """
-- Use WAL mode for better concurrency and crash safety on a single file DB [web:166]
PRAGMA journal_mode=WAL;

-- Main table for job postings; job_id deduplicates same LinkedIn jobs [web:166]
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT UNIQUE,             -- Parsed from LinkedIn job URL when available [web:171]
  title TEXT,
  company TEXT,
  location TEXT,
  url TEXT,
  description TEXT,
  date_posted TEXT,               -- As displayed on page (e.g., "1 week ago") [web:171]
  scraped_at TEXT DEFAULT (datetime('now'))  -- UTC timestamp when row was inserted [web:166]
);

-- FTS5 index for ultra-fast text queries on large text columns [web:166]
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
  title, company, location, description, url,
  content='jobs',                -- Link FTS table to base table (contentless pattern) [web:166]
  content_rowid='id'             -- Rowid matches jobs.id so triggers can sync [web:166]
);

-- Keep FTS updated after INSERT into jobs [web:166]
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
  INSERT INTO jobs_fts(rowid, title, company, location, description, url)
  VALUES (new.id, new.title, new.company, new.location, new.description, new.url);
END;

-- Keep FTS updated after DELETE from jobs (mark as deleted) [web:166]
CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, location, description, url)
  VALUES('delete', old.id, old.title, old.company, old.location, old.description, old.url);
END;

-- Keep FTS updated after UPDATE on jobs (remove old then add new) [web:166]
CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
  INSERT INTO jobs_fts(jobs_fts, rowid, title, company, location, description, url)
  VALUES('delete', old.id, old.title, old.company, old.location, old.description, old.url);
  INSERT INTO jobs_fts(rowid, title, company, location, description, url)
  VALUES (new.id, new.title, new.company, new.location, new.description, new.url);
END;
"""

if __name__ == "__main__":
  # Ensure the db directory exists (safe if it already exists) [web:166]
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  # Create/connect and run schema script in one transaction [web:166]
  with sqlite3.connect(DB_PATH) as conn:
    conn.executescript(DDL)
    print(f"Initialized DB at {DB_PATH}")
