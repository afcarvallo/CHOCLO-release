"""Export the small inputs the GPU experiment needs (run on the main machine, not the GPU one).

Writes src/gpu_probe/bundle/: entities.csv (same order as the OpenAI embeddings), targets.csv
(entity-level LLM-judge score per evaluated model), pageviews.csv, and the reference results of
the OpenAI-embedding predictor (E14) for comparison.
Usage: python src/gpu_probe/export_bundle.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, RESULTS  # noqa: E402
from probe import targets  # noqa: E402

OUT = Path(__file__).resolve().parent / "bundle"


def main():
    OUT.mkdir(exist_ok=True)
    meta = __import__("pandas").read_csv(CACHE / "embeddings" / "entities.csv")
    meta.to_csv(OUT / "entities.csv", index=False)
    targets().reindex(meta.entity_id).round(5).to_csv(OUT / "targets.csv")
    pv = __import__("pandas").read_csv(RESULTS / "E13_pageviews.csv")[["entity_id", "pageviews_60d"]]
    pv.to_csv(OUT / "pageviews.csv", index=False)
    for f in ["E14_probe.csv", "E14_probe_transfer.csv"]:
        shutil.copy(RESULTS / f, OUT / f)
    print("bundle written to", OUT)


if __name__ == "__main__":
    main()
