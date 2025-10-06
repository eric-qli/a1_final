#!/usr/bin/env python3
import argparse, sys
from typing import List, Tuple

# ---- minimal CoNLL-U reader (no external libs) ----
def read_conllu(path: str):
    sents = []
    toks, heads, rels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    sents.append({"tokens": toks, "heads": heads, "rels": rels})
                    toks, heads, rels = [], [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                # skip multiword tokens and empty nodes for simplicity
                continue
            i = int(cols[0])       # 1-based index (ROOT is implicit 0)
            tok = cols[1]
            head = int(cols[6])
            rel  = cols[7]
            toks.append(tok)
            heads.append(head)
            rels.append(rel)
    if toks:
        sents.append({"tokens": toks, "heads": heads, "rels": rels})
    return sents

def differs(h1, r1, h2, r2):
    if len(h1)!=len(h2): return True
    for i in range(len(h1)):
        if h1[i]!=h2[i] or r1[i]!=r2[i]:
            return True
    return False

def wrong_vs_gold(pred_h, pred_r, gold_h, gold_r):
    for i in range(len(gold_h)):
        if pred_h[i]!=gold_h[i] or pred_r[i]!=gold_r[i]:
            return True
    return False

def pretty_diff(tokens, gold_h, gold_r, p_h, p_r, name):
    rows = []
    for i, tok in enumerate(tokens, start=1):
        if p_h[i-1]!=gold_h[i-1] or p_r[i-1]!=gold_r[i-1]:
            rows.append(
                f"  {name}: {i}:{tok}  got {p_h[i-1]}:{p_r[i-1]}  | gold {gold_h[i-1]}:{gold_r[i-1]}"
            )
    return "\n".join(rows) if rows else f"  {name}: (no diffs)"

def main():
    ap = argparse.ArgumentParser(description="Find sentences where Q1 and Q2 disagree and both are wrong vs gold.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--q1", required=True, help="Transition-based outputs (.conllu)")
    ap.add_argument("--q2", required=True, help="Graph-based outputs (.conllu)")
    ap.add_argument("--k", type=int, default=3, help="How many examples to print")
    args = ap.parse_args()

    gold = read_conllu(args.gold)
    q1   = read_conllu(args.q1)
    q2   = read_conllu(args.q2)

    if not (len(gold)==len(q1)==len(q2)):
        print("Mismatch in number of sentences across files.", file=sys.stderr)
        print(f"gold={len(gold)}, q1={len(q1)}, q2={len(q2)}", file=sys.stderr)
        sys.exit(1)

    hits: List[Tuple[int,int]] = []  # (sent_index, num_arc_diffs_between_q1_q2)
    for idx in range(len(gold)):
        g, a, b = gold[idx], q1[idx], q2[idx]
        # length alignment check
        if not (len(g["tokens"])==len(a["tokens"])==len(b["tokens"])):
            continue
        both_wrong = wrong_vs_gold(a["heads"], a["rels"], g["heads"], g["rels"]) \
                     and wrong_vs_gold(b["heads"], b["rels"], g["heads"], g["rels"])
        if not both_wrong:
            continue
        # count disagreements between Q1 and Q2
        diffs = sum((a["heads"][i]!=b["heads"][i]) or (a["rels"][i]!=b["rels"][i])
                    for i in range(len(a["heads"])))
        if diffs>0:
            hits.append((idx, diffs))

    # sort: most disagreements first
    hits.sort(key=lambda x: -x[1])

    if not hits:
        print("No sentences found that satisfy: Q1≠Q2 and both≠gold.")
        return

    print(f"Found {len(hits)} sentences. Showing top {min(args.k, len(hits))}:\n")
    for rank, (idx, diffs) in enumerate(hits[:args.k], start=1):
        g, a, b = gold[idx], q1[idx], q2[idx]
        toks = g["tokens"]
        print(f"=== Example {rank} | sent #{idx} | disagreements(Q1 vs Q2): {diffs} ===")
        print("Sentence:", " ".join(toks))
        print("\n-- Differences vs GOLD --")
        print(pretty_diff(toks, g["heads"], g["rels"], a["heads"], a["rels"], "Q1 (transition)"))
        print(pretty_diff(toks, g["heads"], g["rels"], b["heads"], b["rels"], "Q2 (graph)"))
        print("\n-- Q1 vs Q2 (where they differ) --")
        for i,(h1,r1,h2,r2) in enumerate(zip(a["heads"],a["rels"],b["heads"],b["rels"]), start=1):
            if h1!=h2 or r1!=r2:
                print(f"  idx {i}:{toks[i-1]}  Q1 {h1}:{r1}  |  Q2 {h2}:{r2}")
        print()

if __name__ == "__main__":
    main()