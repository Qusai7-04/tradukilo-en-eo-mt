"""Lightweight checks for the published baseline results artifact."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_results_json_has_expected_systems():
    data = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    assert {"Basic Transformer", "mT5-style", "LLaMA-style", "Gemma-style"} <= set(data["results"])
    for record in data["results"].values():
        assert record["bleu"] >= 0
        assert record["chrf"] >= 0
        assert record["params"] > 0
        assert record["infer_s"] >= 0


def test_baseline_is_not_mislabelled_as_pretrained_family_finetuning():
    data = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    assert data["results"]["LLaMA-style"]["bleu"] < 10
    assert data["results"]["MarianMT (pretrained reference)"]["bleu"] > 20


def test_qualitative_examples_are_present():
    data = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    assert data["examples"]
    for row in data["examples"]:
        assert row.get("en")
        assert row.get("ref")
