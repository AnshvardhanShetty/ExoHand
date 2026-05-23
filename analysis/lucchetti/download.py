"""
Download the Lucchetti et al. 2025 dataset from FigShare.

22 articles total: 10 healthy (HS_01..HS_10) + 10 stroke (ST_01..ST_10) + 2 MATLAB helpers.
Each article exposes files via the FigShare API; we save under
data/lucchetti/{kind}/{code}/<filename>.

Resumable — files are skipped if already present with matching size or SHA256.

Usage:
    python analysis/lucchetti/download.py
    python analysis/lucchetti/download.py --only stroke
    python analysis/lucchetti/download.py --only HS_03 ST_07
    python analysis/lucchetti/download.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "analysis" / "lucchetti" / "article_index.json"
DATA_ROOT = PROJECT_ROOT / "data" / "lucchetti"

API_BASE = "https://api.figshare.com/v2/articles"


def http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "exohand-lucchetti-download/1.0"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def download_file(url: str, dest: Path, expected_size: int = None) -> bool:
    """Stream-download a single file. Returns True if downloaded, False on skip."""
    if dest.exists():
        actual = dest.stat().st_size
        if expected_size is None or actual == expected_size:
            print(f"  SKIP (already present, {actual:,} bytes): {dest.name}")
            return False
        print(f"  RESIZE (have {actual:,}, want {expected_size:,}): re-downloading {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "exohand-lucchetti-download/1.0"})
    bytes_written = 0
    with urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)
    tmp.rename(dest)
    print(f"  OK ({bytes_written:,} bytes): {dest.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset filter: 'healthy', 'stroke', 'code', or specific codes like HS_03 ST_07")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded without fetching.")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        print(f"Article index missing: {INDEX_PATH}", file=sys.stderr)
        sys.exit(1)

    index = json.loads(INDEX_PATH.read_text())
    articles = index["articles"]

    if args.only:
        filt = set(args.only)
        articles = [a for a in articles if a["kind"] in filt or a["code"] in filt]
        if not articles:
            print(f"No articles match filter: {args.only}", file=sys.stderr)
            sys.exit(1)

    print(f"Selected {len(articles)} of {len(index['articles'])} articles")
    print(f"License: {index['license']}  Paper DOI: {index['paper_doi']}\n")

    total_bytes = 0
    total_files = 0
    for art in articles:
        print(f"[{art['code']}] article {art['id']} ({art['kind']}) — {art['title']}")
        try:
            files_meta = http_get_json(f"{API_BASE}/{art['id']}/files")
        except (URLError, HTTPError) as e:
            print(f"  FAILED to fetch file list: {e}", file=sys.stderr)
            continue

        dest_dir = DATA_ROOT / art["kind"] / art["code"]
        for f_info in files_meta:
            name = f_info["name"]
            url = f_info["download_url"]
            size = f_info.get("size")
            total_files += 1
            total_bytes += size or 0
            if args.dry_run:
                print(f"  PLAN ({size:,} bytes): {name}")
                continue
            try:
                download_file(url, dest_dir / name, expected_size=size)
            except (URLError, HTTPError) as e:
                print(f"  FAILED to download {name}: {e}", file=sys.stderr)

    print(f"\n{'PLANNED' if args.dry_run else 'DONE'}: {total_files} files, {total_bytes:,} bytes "
          f"(~{total_bytes / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
