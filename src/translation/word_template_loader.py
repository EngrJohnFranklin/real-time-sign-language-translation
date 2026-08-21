"""Load and manage word template sequences from disk.

Provides utilities to:
- Load all word templates from data/word_templates/
- Validate template shapes
- Cache templates in memory
- Handle missing or invalid templates gracefully
"""

import logging
import json
import numpy as np
from typing import Dict, Optional, List, Tuple

from utils.paths import get_project_root

logger = logging.getLogger(__name__)

# Project structure
WORD_TEMPLATES_DIR = get_project_root() / "data" / "word_templates"

WORD_CLASSES = [
    "hello", "thank_you", "sorry", "yes", "stop",
    "please", "good", "love", "help", "i_love_you"
]

MOTION_BASED_WORDS = {
    "hello", "thank_you", "sorry", "yes", "please", "good", "help"
}

STATIC_WORDS = {"love", "i_love_you", "stop"}

# Initial recordings with short/long duration outliers. Keep them on disk for
# auditability, but exclude them from the initial recognition template set.
EXCLUDED_TEMPLATE_FILES = {
    ("hello", "sample_01.npy"),
    ("sorry", "sample_02.npy"),
    ("i_love_you", "sample_01.npy"),
}


class WordTemplateManager:
    """Load and cache word templates."""

    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.metadata: Dict[str, dict] = {}
        self.loaded = False

    def load_all_templates(self) -> bool:
        """Load all word templates from disk.
        
        Returns:
            True if at least one template was loaded, False otherwise.
        """
        if not WORD_TEMPLATES_DIR.exists():
            logger.warning(f"Word templates directory does not exist: {WORD_TEMPLATES_DIR}")
            return False
        
        loaded_count = 0
        
        for word in WORD_CLASSES:
            word_dir = WORD_TEMPLATES_DIR / word
            if not word_dir.exists():
                logger.debug(f"No templates found for word '{word}'")
                self.templates[word] = []
                continue
            
            # Load .npy files
            samples = []
            for npy_file in sorted(word_dir.glob("sample_*.npy")):
                if (word, npy_file.name) in EXCLUDED_TEMPLATE_FILES:
                    logger.info("Skipping excluded template: %s", npy_file)
                    continue
                try:
                    template = np.load(str(npy_file))
                    
                    # Validate shape
                    if template.ndim != 2 or template.shape[1] != 126:
                        logger.warning(
                            f"Invalid shape {template.shape} for {npy_file}; "
                            f"expected (T, 126)"
                        )
                        continue
                    
                    samples.append(template)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load {npy_file}: {e}")
            
            self.templates[word] = samples
            
            # Load metadata if available
            metadata_file = word_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        self.metadata[word] = [json.loads(line) for line in f]
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {word}: {e}")
        
        self.loaded = True
        logger.info(f"Loaded {loaded_count} word templates from {WORD_TEMPLATES_DIR}")
        return loaded_count > 0

    def get_templates_for_word(self, word: str) -> List[np.ndarray]:
        """Return list of templates for a word.
        
        Args:
            word: Word label.
        
        Returns:
            List of (T, 126) template arrays, or empty list if none found.
        """
        if not self.loaded:
            self.load_all_templates()
        
        return self.templates.get(word, [])

    def is_motion_based(self, word: str) -> bool:
        """Check if word is motion-based or static."""
        return word in MOTION_BASED_WORDS

    def get_all_templates(self) -> Dict[str, List[np.ndarray]]:
        """Return all loaded templates."""
        if not self.loaded:
            self.load_all_templates()
        return self.templates

    def count_samples_per_word(self) -> Dict[str, int]:
        """Count how many samples are available per word."""
        if not self.loaded:
            self.load_all_templates()
        
        return {word: len(samples) for word, samples in self.templates.items()}
