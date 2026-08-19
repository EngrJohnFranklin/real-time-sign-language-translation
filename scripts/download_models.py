"""Verify that the application's locally bundled models are available.

This script intentionally performs no network access and never downloads files.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _check_path(path: Path, description: str) -> bool:
	"""Print the local model status and return whether the path exists."""
	if path.exists():
		print(f"[OK] {description}: {path}")
		return True

	print(f"[MISSING] {description}: {path}")
	return False


def verify_models() -> bool:
	"""Verify all model paths required by the offline runtime."""
	checks = [
		(
			PROJECT_ROOT / "model",
			"Bundled English Vosk speech model directory",
		),
		(
			PROJECT_ROOT / "model-tl",
			"Bundled Filipino/Tagalog Vosk speech model directory",
		),
		(
			PROJECT_ROOT / "data" / "models" / "sign_model.pkl",
			"FSL XGBoost sign classifier",
		),
	]

	statuses = [_check_path(path, description) for path, description in checks]
	available = all(statuses)
	if available:
		print("All required local models are available.")
	else:
		print(
			"One or more local models are missing. "
			"Install or place the model files locally before starting the application."
		)
	return available


if __name__ == "__main__":
	sys.exit(0 if verify_models() else 1)
