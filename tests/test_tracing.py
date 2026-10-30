from types import SimpleNamespace

from app import tracing


def test_client_initializes_from_environment_configured_singleton(monkeypatch):
    client = object()
    calls = []
    monkeypatch.setattr(tracing, "settings", SimpleNamespace(langfuse_public_key="public", langfuse_secret_key="secret"))
    monkeypatch.setattr(tracing, "get_client", lambda *args, **kwargs: calls.append((args, kwargs)) or client)

    assert tracing._client() is client
    assert calls == [((), {})]


def test_finish_embedding_observation_records_usage_and_only_text_count():
    class Observation:
        def update(self, **kwargs):
            self.values = kwargs

        def end(self):
            self.ended = True

    observation = Observation()

    tracing.finish_embedding_observation(observation, {"input": 17}, 2)

    assert observation.values == {"usage_details": {"input": 17}, "metadata": {"text_count": 2, "success": True}}
    assert observation.ended


def test_finish_generation_observation_records_usage_without_answer_text():
    class Observation:
        def update(self, **kwargs):
            self.values = kwargs

        def end(self):
            self.ended = True

    observation = Observation()

    tracing.finish_generation_observation(observation, {"input": 17, "output": 4}, True, 28, 1)

    assert observation.values == {
        "usage_details": {"input": 17, "output": 4},
        "metadata": {"success": True, "answer_chars": 28, "citation_count": 1},
    }
    assert observation.ended


def test_finish_embedding_observation_marks_failed_provider_call():
    class Observation:
        def update(self, **kwargs):
            self.values = kwargs

        def end(self):
            self.ended = True

    observation = Observation()

    tracing.finish_embedding_observation(observation, {"input": 9}, 1, False, "APIError")

    assert observation.values["usage_details"] == {"input": 9}
    assert observation.values["metadata"] == {"text_count": 1, "success": False}
    assert observation.values["level"] == "ERROR"
    assert observation.values["status_message"] == "APIError"
    assert observation.ended
