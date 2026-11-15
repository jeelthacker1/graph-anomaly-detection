"""Offline backup, integrity inspection, and selected-feedback export."""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["backup", "inspect", "export-feedback"])
    parser.add_argument("--database", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.database).resolve()
    if not source.is_file():
        raise SystemExit("Database does not exist")
    with sqlite3.connect(source.as_uri()+"?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        if args.operation == "inspect":
            previous, valid, rows = "0"*64, True, 0
            for row in db.execute("SELECT * FROM audit ORDER BY seq"):
                digest = hashlib.sha256((previous+row["payload"]).encode()).hexdigest()
                if row["previous_hash"] != previous or row["hash"] != digest:
                    valid = False
                    break
                previous, rows = row["hash"], rows+1
            report = {"sqlite_integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
                      "audit_valid": valid, "audit_rows_checked": rows,
                      "model_version": db.execute("SELECT value FROM meta WHERE key='model_version'").fetchone()[0]}
            print(json.dumps(report, indent=2))
            if not valid or report["sqlite_integrity"] != "ok":
                raise SystemExit(1)
        else:
            if not args.output:
                raise SystemExit("--output is required")
            target = Path(args.output)
            if target.exists():
                raise SystemExit("Refusing to overwrite an existing output")
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.operation == "backup":
                with sqlite3.connect(target) as backup:
                    db.backup(backup)
            else:
                rows = db.execute("""SELECT payment_id,status,version,updated,actor FROM cases
                                     WHERE status IN ('closed_fraud','closed_legitimate') ORDER BY updated""")
                with target.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["payment_id", "review_disposition", "case_version", "updated", "actor"])
                    writer.writerows(tuple(row) for row in rows)
            print(f"Saved {args.operation} output to {target}")


if __name__ == "__main__":
    main()
