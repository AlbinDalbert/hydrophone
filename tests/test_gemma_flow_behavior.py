from __future__ import annotations

import gemma_flow.app as gf


def test_transcription_mode_aliases_and_validation():
    assert gf.selected_transcription_mode(gf.Config(transcription_mode="direct")) == "gemma_direct"
    assert gf.selected_transcription_mode(gf.Config(transcription_mode="two-pass")) == "gemma_two_pass"
    assert gf.selected_transcription_mode(gf.Config(transcription_mode="whisper_cpp")) == "whisper_gemma"
    assert gf.selected_transcription_mode(gf.Config(transcription_mode="whisper_remote")) == "whisper_api_gemma"

    try:
        gf.selected_transcription_mode(gf.Config(transcription_mode="bogus"))
    except RuntimeError as e:
        assert "Unsupported transcription_mode" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_clean_model_output_strips_common_wrappers_and_detects_silence():
    result = gf.clean_model_output("```text\nFinal text: Hello world.\n```")
    assert result.text == "Hello world."
    assert result.empty_reason == ""

    silence = gf.clean_model_output(" [silence] ")
    assert silence.text == ""
    assert silence.empty_reason == "model_silence"

    missing = gf.clean_model_output("Please provide an audio file or link.")
    assert missing.text == ""
    assert missing.empty_reason == "missing_audio"


def test_phrase_replacements_are_applied_case_insensitively():
    cfg = gf.Config(phrase_replacements={"Jammer": "Gemma", "ollama dot land": "ollama.lan"})
    assert gf.apply_phrase_replacements("jammer talks to Ollama dot land", cfg) == "Gemma talks to ollama.lan"


def test_cleanup_falls_back_to_raw_transcript_when_endpoint_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Could not reach Ollama")

    monkeypatch.setattr(gf, "ollama_generate_text", unavailable)

    result = gf.cleanup_transcript_with_gemma(gf.Config(), "Whisper text directly.", None)

    assert result.text == "Whisper text directly."
    assert "Could not reach Ollama" in result.raw
    assert result.empty_reason == ""


def test_cmd_transcribe_file_uses_processing_pipeline_without_paste(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"0" * 2048)

    def fake_transcribe_audio(cfg, audio_path, state):
        assert audio_path == audio
        return gf.TranscriptionResult("hello from test", "raw")

    notifications = []
    monkeypatch.setattr(gf, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(gf, "notify", lambda *args, **kwargs: notifications.append((args, kwargs)))

    code = gf.cmd_transcribe_file(gf.Config(), str(audio), paste=False)

    assert code == 0
    assert capsys.readouterr().out.strip() == "hello from test"
    assert notifications[-1][0][:2] == ("Gemma Flow", "✓ Transcribed")
