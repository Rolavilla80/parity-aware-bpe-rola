import csv
import os
from collections import Counter

CSV_FILES = [
    r"benchmark_outputs\enzh_100_none_log.csv",
    r"benchmark_outputs\enzh_100_max_log.csv",
    r"benchmark_outputs\enzh_100_local_log.csv",
    r"benchmark_outputs\enzh_100_rescan_log.csv",
]

OUTPUT_SUMMARY = r"benchmark_outputs\merge_log_summary.csv"

def summarize_csv(path: str) -> dict:
    category_counter = Counter()
    first_mixed = None
    first_bad = None
    total = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            category = row["category"].strip().lower()
            step = int(row["step"])

            category_counter[category] += 1

            if category == "mixed" and first_mixed is None:
                first_mixed = step
            if category == "bad" and first_bad is None:
                first_bad = step

    good = category_counter["good"]
    mixed = category_counter["mixed"]
    bad = category_counter["bad"]

    def pct(x: int) -> float:
        return (x / total * 100) if total else 0.0

    return {
        "file": os.path.basename(path),
        "total": total,
        "good": good,
        "mixed": mixed,
        "bad": bad,
        "good_pct": round(pct(good), 2),
        "mixed_pct": round(pct(mixed), 2),
        "bad_pct": round(pct(bad), 2),
        "first_mixed": first_mixed,
        "first_bad": first_bad,
    }

def main():
    rows = []

    for path in CSV_FILES:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        rows.append(summarize_csv(path))

    if not rows:
        print("No CSV files found.")
        return

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "total", "good", "mixed", "bad",
                "good_pct", "mixed_pct", "bad_pct",
                "first_mixed", "first_bad"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to: {OUTPUT_SUMMARY}")
    for row in rows:
        print(row)

if __name__ == "__main__":
    main()