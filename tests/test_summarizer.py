from summarizer import generate_summary

class DummyPipeline:
    def __call__(self, text, max_length=None, min_length=None, truncation=True):
        return [{"summary_text": text[: min(50, len(text))] + "..."}]


def test_generate_summary_simple(monkeypatch):
    monkeypatch.setattr("summarizer._get_pipeline", lambda: DummyPipeline())
    long_text = "This is a test. " * 200
    summary = generate_summary(long_text)
    assert isinstance(summary, str)
    assert len(summary) > 0
