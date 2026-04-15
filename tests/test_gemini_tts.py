import base64

from src.core.gemini_tts import decode_inline_audio_data, select_speaker_for_segment


def test_decode_inline_audio_data_accepts_bytes():
    payload = b"abc"
    assert decode_inline_audio_data(payload) == b"abc"


def test_decode_inline_audio_data_accepts_base64_string():
    raw = b"pcm-data"
    encoded = base64.b64encode(raw).decode("ascii")
    assert decode_inline_audio_data(encoded) == raw


def test_select_speaker_alternate_mode():
    speaker, text = select_speaker_for_segment(
        "hello world",
        idx=1,
        speaker1_name="Alice",
        speaker2_name="Bob",
        assignment_mode="alternate",
    )
    assert speaker == "Bob"
    assert text == "hello world"


def test_select_speaker_prefix_mode_matches_prefix_name():
    speaker, text = select_speaker_for_segment(
        "Alice: bom dia",
        idx=0,
        speaker1_name="Alice",
        speaker2_name="Bob",
        assignment_mode="prefix",
    )
    assert speaker == "Alice"
    assert text == "bom dia"


def test_select_speaker_prefix_mode_falls_back_to_speaker1():
    speaker, text = select_speaker_for_segment(
        "sem prefixo",
        idx=10,
        speaker1_name="Alice",
        speaker2_name="Bob",
        assignment_mode="prefix",
    )
    assert speaker == "Alice"
    assert text == "sem prefixo"
