"""Focused tests for closed-set speech phrase routing."""

import pytest

from translation.speech_handler import SpeechResult, VoskSpeechRecognizer


def make_recognizer():
    recognizer = VoskSpeechRecognizer.__new__(VoskSpeechRecognizer)
    recognizer._last_partial_text = ""
    recognizer._last_partial_time = 0.0
    recognizer._cooldown_discard_count = 0
    recognizer.result_callback = None
    recognizer.recognizer = None
    recognizer._reset_calls = 0
    recognizer._cooldown_calls = 0

    def reset():
        recognizer._reset_calls += 1

    def cooldown():
        recognizer._cooldown_calls += 1

    recognizer._reset_recognizer = reset
    recognizer._start_cooldown = cooldown
    return recognizer


def test_multword_phrase_prefix_and_exact_matching():
    recognizer = make_recognizer()

    assert recognizer._match_phrase("thank") == ("prefix", None)
    assert recognizer._match_phrase("thank you") == ("exact", "thank you")
    assert recognizer._match_phrase("i love") == ("prefix", None)
    assert recognizer._match_phrase("i love you") == ("exact", "i love you")


@pytest.mark.parametrize("word", ["hello", "love", "thank you"])
def test_exact_partial_vocabulary_match_is_never_finalized(word):
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append
    result = SpeechResult(word, 0.72, False)

    recognizer._handle_recognition_attempt(
        f'{{"partial":"{word}"}}', result, is_final=False, rms=500
    )

    assert [(item.text, item.is_final) for item in results] == [(word, False)]
    assert recognizer._cooldown_calls == 0


def test_nonexact_partial_is_not_finalized_or_reset():
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append
    result = SpeechResult("banana", 0.5, False)

    recognizer._handle_recognition_attempt(
        '{"partial":"banana"}', result, is_final=False, rms=500
    )

    assert results == []
    assert recognizer._reset_calls == 0
    assert recognizer._cooldown_calls == 0


@pytest.mark.parametrize("word", ["hello", "love", "thank you"])
def test_exact_final_vocabulary_match_is_finalized(word):
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append
    result = SpeechResult(word, 0.72, True)

    recognizer._handle_recognition_attempt(
        f'{{"text":"{word}"}}', result, is_final=True, rms=500
    )

    assert [(item.text, item.is_final) for item in results] == [(word, True)]
    assert recognizer._cooldown_calls == 1


def test_nonexact_final_resets_recognizer_without_finalizing():
    recognizer = make_recognizer()
    result = SpeechResult("banana", 0.5, True)

    recognizer._handle_recognition_attempt(
        '{"text":"banana"}', result, is_final=True, rms=500
    )

    assert recognizer._reset_calls == 1
    assert recognizer._cooldown_calls == 0


def test_empty_partial_waits_without_reset_or_info_attempt():
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append

    assert recognizer._match_phrase("   ") == ("waiting", None)
    recognizer._handle_recognition_attempt(
        '{"partial":""}', None, is_final=False, rms=0
    )

    assert results == []
    assert recognizer._reset_calls == 0
    assert recognizer._cooldown_discard_count == 0
