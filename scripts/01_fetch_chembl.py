"""Download all ChEMBL IC50 activities for hERG (CHEMBL240, KCNH2).

Pages are checkpointed to data/raw/pages/, so an interrupted run resumes
where it stopped. The merged file is written only when every record the
API reports has been downloaded.
"""
import csv, gzip, json, sys, time, urllib.request
from pathlib import Path

TARGET = "CHEMBL240"
BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000
ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "raw" / "pages"
OUT = ROOT / "data" / "raw" / "chembl240_ic50.csv.gz"


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # network hiccup: back off and retry
            if i == tries - 1:
                raise
            print(f"retry {i + 1}: {e}", file=sys.stderr)
            time.sleep(5 * (i + 1))


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    target = get(f"{BASE}/target/{TARGET}.json")
    status = get(f"{BASE}/status.json")
    assert "KCNH2" in target["pref_name"], target["pref_name"]
    assert target["organism"] == "Homo sapiens", target["organism"]

    query = f"target_chembl_id={TARGET}&standard_type=IC50"
    total = get(f"{BASE}/activity.json?{query}&limit=1")["page_meta"]["total_count"]
    for offset in range(0, total, PAGE):
        page = PAGES / f"{offset:06d}.json"
        if page.exists():
            continue
        data = get(f"{BASE}/activity.json?{query}&limit={PAGE}&offset={offset}&order_by=activity_id")
        tmp = page.with_suffix(".tmp")
        tmp.write_text(json.dumps(data["activities"]))
        tmp.rename(page)
        print(f"{offset + len(data['activities'])}/{total}", flush=True)

    rows = [a for p in sorted(PAGES.glob("*.json")) for a in json.loads(p.read_text())]
    ids = {a["activity_id"] for a in rows}
    if len(ids) != total:
        sys.exit(f"incomplete: {len(ids)} unique activities, API reports {total}")
    fields = sorted({k for a in rows for k in a})
    with gzip.open(OUT, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    meta = {"target": TARGET, "pref_name": target["pref_name"], "organism": target["organism"],
            "chembl_version": status["chembl_db_version"], "n_activities": total,
            "downloaded": time.strftime("%Y-%m-%d")}
    (ROOT / "data" / "raw" / "download_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
