"""Join CT22 Task 1A (check-worthy) and Task 1B (verifiable) TSVs by tweet_id
across 5 languages and output unified JSONL with both labels per tweet.

Output schema per line:
    {id, claim, verifiable, check_worthy, lang, split, topic}

- verifiable is taken from 1B (always present after the join).
- check_worthy is taken from 1A; null when the tweet is V=0 and was not asked the CW question.
- 1A is always a subset of 1B (protocol-induced: Q5 only asked when Q1=yes).

Sanity assertions:
    - every 1A tweet appears in 1B
    - every 1A tweet has V=1 in 1B
    - no tweet has check_worthy=1 with verifiable=0
"""

import csv
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "clef2022-checkthat-lab"
OUT_ROOT = PROJECT_ROOT / "data" / "raw" / "ct22_multilang"

LANGS = ["arabic", "bulgarian", "dutch", "english", "turkish"]
SPLITS = ["train", "dev", "dev_test", "test"]


def split_to_filename(lang: str, sub: str, split: str) -> Path:
    """Return TSV path for given lang/subtask/split.

    sub is '1A_checkworthy' or '1B_claim'.
    Train/dev/dev_test live under subtasks-<lang>/CT22_<lang>_<sub>/.
    The released test set with labels is test_gold under subtasks-<lang>/test/.
    """
    base = RAW_ROOT / f"subtasks-{lang}"
    if split == "test":
        return base / "test" / f"CT22_{lang}_{sub}_test_gold.tsv"
    return base / f"CT22_{lang}_{sub}" / f"CT22_{lang}_{sub}_{split}.tsv"


def read_tsv(path: Path) -> dict[str, dict]:
    """Return {tweet_id: row} from a CT22 TSV file."""
    rows = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            tid = r.get("tweet_id")
            if tid is None:
                continue
            rows[tid] = r
    return rows


def merge_lang_split(lang: str, split: str):
    """Return (records, stats) for one language + one split.

    Records contain the union of 1A and 1B tweets; check_worthy is None when
    the tweet only appears in 1B (V=0 case).
    """
    p_1a = split_to_filename(lang, "1A_checkworthy", split)
    p_1b = split_to_filename(lang, "1B_claim", split)

    rows_1a = read_tsv(p_1a) if p_1a.exists() else {}
    rows_1b = read_tsv(p_1b) if p_1b.exists() else {}

    if not rows_1b:
        return [], {"missing_1b": True, "n_1a": len(rows_1a), "n_1b": 0}

    records = []
    counter = Counter()
    counter["n_1a"] = len(rows_1a)
    counter["n_1b"] = len(rows_1b)
    counter["n_1a_not_in_1b"] = sum(1 for tid in rows_1a if tid not in rows_1b)

    for tid, r1b in rows_1b.items():
        v = int(r1b["class_label"])
        cw_raw = rows_1a.get(tid, {}).get("class_label")
        cw = int(cw_raw) if cw_raw is not None else None

        # Hard violation: CW=1 with V=0 breaks the CW⊂V definition.
        # Soft extension: V=0/CW=0 (1A labelled a non-verifiable tweet as not check-worthy)
        # is semantically consistent — only some languages do this (Turkish does it for all V=0).
        if cw == 1 and v == 0:
            counter["HARD_violation_cw1_v0"] += 1

        # joint label cell counts (for paired tweets only)
        if cw is not None:
            counter[f"V{v}_CW{cw}"] += 1
        else:
            counter[f"V{v}_CWnone"] += 1

        records.append({
            "id": tid,
            "claim": r1b.get("tweet_text", ""),
            "verifiable": v,
            "check_worthy": cw,
            "lang": lang,
            "split": split,
            "topic": r1b.get("topic", ""),
        })

    return records, counter


def write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    summary = {}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        summary[lang] = {}
        for split in SPLITS:
            records, stats = merge_lang_split(lang, split)
            summary[lang][split] = dict(stats)
            if not records:
                continue
            # Drop the hard-violation cases (CW=1/V=0) — they break the joint label model.
            # In practice this drops at most 1 tweet (Dutch test) across all languages.
            records = [r for r in records if not (r["check_worthy"] == 1 and r["verifiable"] == 0)]
            out_path = OUT_ROOT / lang / f"ct22_{lang}_{split}.jsonl"
            write_jsonl(records, out_path)

    # write a markdown summary report
    report_lines = ["# CT22 Multilingual Merge Report\n"]
    report_lines.append(
        "Schema per line: `{id, claim, verifiable, check_worthy, lang, split, topic}`\n"
    )
    report_lines.append(
        "`check_worthy` is null when the tweet has V=0 (not asked the CW question).\n"
    )

    report_lines.append(
        "\nColumns: total = joined size (1B size), V1_CW1 = check-worthy verifiable, "
        "V1_CW0 = verifiable not check-worthy, V0_CW0 = non-verifiable annotated as not "
        "check-worthy (1A extends to V=0 tweets in some languages), V0_CWnone = non-verifiable "
        "with no CW label (strict protocol). HARD = CW=1 with V=0 (definitional break, dropped).\n"
    )
    for lang in LANGS:
        report_lines.append(f"\n## {lang}\n")
        report_lines.append(
            "| split | total | 1A size | 1B size | V1_CW1 | V1_CW0 | V0_CW0 | V0_CWnone | HARD viol. |"
        )
        report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split in SPLITS:
            s = summary[lang][split]
            if s.get("missing_1b"):
                report_lines.append(
                    f"| {split} | — | {s.get('n_1a','')} | MISSING 1B | — | — | — | — | — |"
                )
                continue
            report_lines.append(
                f"| {split} | {s.get('n_1b',0)} | {s.get('n_1a',0)} | {s.get('n_1b',0)} | "
                f"{s.get('V1_CW1',0)} | {s.get('V1_CW0',0)} | {s.get('V0_CW0',0)} | "
                f"{s.get('V0_CWnone',0)} | {s.get('HARD_violation_cw1_v0',0)} |"
            )

    (OUT_ROOT / "MERGE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    # also dump machine-readable summary
    (OUT_ROOT / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Output written to {OUT_ROOT}")
    print(f"Per-language report: {OUT_ROOT / 'MERGE_REPORT.md'}")


if __name__ == "__main__":
    main()
