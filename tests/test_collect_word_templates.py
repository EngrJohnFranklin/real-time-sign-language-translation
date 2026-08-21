"""Regression tests for word-template collection."""

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
COLLECTOR_PATH = PROJECT_ROOT / "scripts" / "collect_word_templates.py"


def _load_collector_module():
    spec = importlib.util.spec_from_file_location(
        "collect_word_templates", COLLECTOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_sequence_uses_next_index_after_filename_gap(tmp_path, monkeypatch):
    """A missing earlier sample must not cause the highest file to be overwritten."""
    collector = _load_collector_module()
    word_dir = tmp_path / "sorry"
    word_dir.mkdir()

    for sample_index in range(2, 5):
        np.save(word_dir / f"sample_{sample_index:02d}.npy", np.zeros((10, 126)))

    original_sample = word_dir / "sample_04.npy"
    original_contents = original_sample.read_bytes()
    monkeypatch.setattr(collector, "OUTPUT_ROOT", tmp_path)

    recorder = collector.WordTemplateRecorder.__new__(collector.WordTemplateRecorder)
    recorder.current_sequence = [np.zeros(126, dtype=np.float32)] * 10
    recorder.dry_run = False
    recorder.camera_index = 0

    assert recorder._save_sequence("sorry")
    assert (word_dir / "sample_05.npy").exists()
    assert original_sample.read_bytes() == original_contents