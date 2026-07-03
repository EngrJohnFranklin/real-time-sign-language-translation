"""
Sign to Text Module
Converts detected sign language gestures into readable text output
"""

from typing import Dict, Optional


class SignToTextConverter:
	"""
	Convert confirmed sign predictions into output text.

	Consecutive duplicate words are ignored to reduce repetitive output.
	"""

	def __init__(self, sign_to_word_map: Optional[Dict[str, str]] = None):
		"""Initialize the converter with an optional sign-to-word mapping."""
		self.sign_to_word_map = sign_to_word_map or {}
		self.words = []
		self.last_word = None

	def _normalize_word(self, sign_name: str) -> str:
		"""
		Convert a sign label to a readable word/phrase.

		Args:
			sign_name: Raw recognized sign label.

		Returns:
			Normalized text token.
		"""
		mapped = self.sign_to_word_map.get(sign_name, sign_name)
		return " ".join(str(mapped).strip().split())

	def add_confirmed_prediction(self, sign_name: str) -> Optional[str]:
		"""
		Add a confirmed sign prediction to the output stream.

		Args:
			sign_name: Confirmed sign label.

		Returns:
			Appended word/phrase if added, otherwise None.
		"""
		word = self._normalize_word(sign_name)
		if not word:
			return None

		if self.last_word and word.lower() == self.last_word.lower():
			return None

		self.words.append(word)
		self.last_word = word
		return word

	def get_text(self) -> str:
		"""Get the full accumulated output text."""
		return " ".join(self.words)

	def clear(self):
		"""Reset converter state and accumulated text."""
		self.words.clear()
		self.last_word = None
