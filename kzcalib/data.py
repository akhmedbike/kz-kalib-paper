"""CoNLL-U loading shared by every model in the benchmark.

Tokenisation rules match the CRF baseline's original training parser exactly:
comment lines skipped, multiword token ranges (``1-2``) skipped so that only
syntactic words are evaluated, empty nodes (``8.1``) are absent from UD_Kazakh-KTB.
That parser yielded 7,376 / 1,566 / 1,594 tokens on the canonical split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

UPOS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
LABEL2ID = {l: i for i, l in enumerate(UPOS_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class Sentence:
    sent_id: str
    tokens: list[str]
    upos: list[str]

    def __len__(self) -> int:
        return len(self.tokens)


def parse_conllu(filepath: str | Path) -> list[Sentence]:
    """Parse a CoNLL-U file into sentences of syntactic words."""
    sentences: list[Sentence] = []
    current_tokens: list[str] = []
    current_upos: list[str] = []
    sent_id = ""

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# sent_id"):
                    sent_id = line.split("=", 1)[1].strip()
                continue
            if not line.strip():
                if current_tokens:
                    sentences.append(Sentence(sent_id, current_tokens, current_upos))
                    current_tokens, current_upos = [], []
                continue
            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:
                continue
            if len(parts) >= 4:
                current_tokens.append(parts[1])
                current_upos.append(parts[3])

    if current_tokens:
        sentences.append(Sentence(sent_id, current_tokens, current_upos))
    return sentences


def load_split(data_dir: str | Path) -> dict[str, list[Sentence]]:
    data_dir = Path(data_dir)
    return {
        name: parse_conllu(data_dir / f"{name}.conllu")
        for name in ("train", "dev", "test")
    }


def flatten(sentences: list[Sentence]) -> dict[str, list]:
    """Flatten sentences into parallel per-token lists."""
    toks, golds, sids, tidx = [], [], [], []
    for s in sentences:
        for i, (t, g) in enumerate(zip(s.tokens, s.upos)):
            toks.append(t)
            golds.append(g)
            sids.append(s.sent_id)
            tidx.append(i)
    return {"token": toks, "gold": golds, "sent_id": sids, "tok_idx": tidx}


def split_stats(sentences: list[Sentence]) -> dict:
    return {
        "sentences": len(sentences),
        "tokens": sum(len(s) for s in sentences),
    }
