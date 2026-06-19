"""Build structured-text cards from an extracted textbook circuit corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optcpv.textbook_surrogate import TextbookCorpusIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="textbook_circuit_corpus")
    parser.add_argument("--likely-only", action="store_true")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus).resolve()
    out_dir = corpus_dir / "structured_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = TextbookCorpusIndex(corpus_dir, include_all_figures=not args.likely_only)
    cards = corpus.cards

    _write_jsonl(out_dir / "figure_cards.jsonl", [card.to_dict() for card in cards])
    (out_dir / "figure_cards.txt").write_text("\n\n---\n\n".join(card.structured_text for card in cards), encoding="utf-8")
    _write_json(out_dir / "style_guide.json", corpus.style_guide(cards))
    _write_json(out_dir / "manifest.json", {"corpus": str(corpus_dir), "card_count": len(cards), "likely_only": args.likely_only})
    _write_category_views(out_dir, cards)
    print(json.dumps({"card_count": len(cards), "out_dir": str(out_dir)}, indent=2))


def _write_category_views(out_dir: Path, cards) -> None:
    by_family: dict[str, list] = {}
    for card in cards:
        for family in card.visual_families or ("generic",):
            by_family.setdefault(family, []).append(card)
    family_dir = out_dir / "by_family"
    family_dir.mkdir(exist_ok=True)
    for family, family_cards in sorted(by_family.items()):
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in family)
        (family_dir / f"{safe}.txt").write_text("\n\n---\n\n".join(card.structured_text for card in family_cards), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
