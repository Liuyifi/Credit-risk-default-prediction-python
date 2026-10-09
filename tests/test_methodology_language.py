from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_narrative_does_not_call_legacy_holdout_pristine():
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "project_notes.md",
        ROOT / "src" / "modeling.py",
        ROOT / "src" / "validation.py",
        ROOT / "notebooks" / "03_model_development.ipynb",
        ROOT / "notebooks" / "04_model_validation.ipynb",
    ]
    forbidden = ("untouched final test", "pristine test", "never-before-seen test", "locked final test")
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    assert not any(phrase in combined for phrase in forbidden)
