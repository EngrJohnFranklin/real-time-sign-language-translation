"""Focused tests for closed-set speech phrase routing."""

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


def test_exact_partial_resolves_multword_phrase():
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append
    result = SpeechResult("thank you", 0.72, False)

    recognizer._handle_recognition_attempt(
        '{"partial":"thank you"}', result, is_final=False, rms=500
    )

    assert [(item.text, item.is_final) for item in results] == [("thank you", True)]
    assert recognizer._cooldown_calls == 1


def test_out_of_vocabulary_partial_is_rejected_immediately():
    recognizer = make_recognizer()
    results = []
    recognizer.result_callback = results.append
    result = SpeechResult("banana", 0.5, False)

    recognizer._handle_recognition_attempt(
        '{"partial":"banana"}', result, is_final=False, rms=500
    )

    assert results == []
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
