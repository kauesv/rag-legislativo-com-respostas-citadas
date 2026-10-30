import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from app import providers
from app.providers import ProviderError


class FakeEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.3, 0.2, 0.1]


@pytest.fixture(autouse=True)
def clear_cached_langchain_clients(monkeypatch):
    monkeypatch.setattr(providers, "start_embedding_observation", lambda: None)
    monkeypatch.setattr(providers, "finish_embedding_observation", lambda *_args: None)
    monkeypatch.setattr(providers, "start_generation_observation", lambda: None)
    monkeypatch.setattr(providers, "finish_generation_observation", lambda *_args: None)
    for name in ("_embedding_model", "_chat_model"):
        factory = getattr(providers, name, None)
        if factory and hasattr(factory, "cache_clear"):
            factory.cache_clear()
    yield
    for name in ("_embedding_model", "_chat_model"):
        factory = getattr(providers, name, None)
        if factory and hasattr(factory, "cache_clear"):
            factory.cache_clear()


def configure_provider(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(providers, "settings", SimpleNamespace(
        provider_api_key="test-key",
        provider_base_url="https://provider.example/v1",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=3,
        chat_model="test-chat-model",
    ))
    monkeypatch.setattr(providers, "OpenAIEmbeddings", FakeEmbeddings, raising=False)


def test_embed_texts_uses_langchain_documents_and_preserves_dimensions(monkeypatch):
    configure_provider(monkeypatch)

    vectors = providers.embed_texts(["primeiro", "segundo"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert providers._embedding_model().kwargs == {
        "model": "text-embedding-3-small",
        "api_key": "test-key",
        "base_url": "https://provider.example/v1",
        "dimensions": 3,
    }


def test_embed_query_uses_langchain_query_interface(monkeypatch):
    configure_provider(monkeypatch)

    vector = providers.embed_query("pergunta")

    assert vector == [0.3, 0.2, 0.1]


def test_embed_texts_records_provider_token_usage_without_text(monkeypatch):
    from types import SimpleNamespace

    configure_provider(monkeypatch)
    recorded = []
    class Client:
        def create(self, **_kwargs):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=17))

    class TracedEmbeddings:
        def __init__(self, **_kwargs):
            self.client = Client()

        def embed_documents(self, texts):
            self.client.create(input=texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(providers, "OpenAIEmbeddings", TracedEmbeddings)
    monkeypatch.setattr(providers, "start_embedding_observation", lambda: object())
    monkeypatch.setattr(providers, "finish_embedding_observation", lambda observation, usage, count, *_status: recorded.append((usage, count)))

    vectors = providers.embed_texts(["texto confidencial"])

    assert vectors == [[0.1, 0.2, 0.3]]
    assert recorded == [({"input": 17}, 1)]


@pytest.mark.parametrize("vectors", [
    [[0.1, 0.2]],
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    [[0.1, 0.2, float("nan")]],
])
def test_embed_texts_rejects_malformed_vectors(monkeypatch, vectors):
    configure_provider(monkeypatch)
    monkeypatch.setattr(FakeEmbeddings, "embed_documents", lambda _self, _texts: vectors)

    with pytest.raises(ProviderError, match="quantidade ou dimensão"):
        providers.embed_texts(["documento"])


def test_embedding_without_api_key_fails_before_model_creation(monkeypatch):
    configure_provider(monkeypatch)
    providers.settings.provider_api_key = ""

    with pytest.raises(ProviderError, match="Configure OPENAI_API_KEY"):
        providers.embed_texts(["documento"])


def test_generate_answer_uses_langchain_prompt_and_structured_parser(monkeypatch):
    configure_provider(monkeypatch)
    seen_messages = []

    def respond(prompt):
        seen_messages.extend(prompt.to_messages())
        return AIMessage(content='{"answer":"Resposta baseada no trecho.","citations":["chunk-1"]}', usage_metadata={"input_tokens": 17, "output_tokens": 4, "total_tokens": 21})

    model_options = []

    def create_chat_model(**kwargs):
        model_options.append(kwargs)
        return RunnableLambda(respond)

    monkeypatch.setattr(providers, "ChatOpenAI", create_chat_model, raising=False)
    chunks = [{"chunk_id": "chunk-1", "document_id": "doc-1", "reference": "PL 1/2024", "title": "Projeto", "text": "Trecho oficial."}]

    answer = providers.generate_answer("O que diz o projeto?", chunks)

    assert answer == {
        "answer": "Resposta baseada no trecho.",
        "citations": ["chunk-1"],
        "usage": {"input": 17, "output": 4, "total": 21},
    }
    assert model_options == [{
        "model": "test-chat-model",
        "api_key": "test-key",
        "base_url": "https://provider.example/v1",
        "max_retries": 2,
    }]
    prompt_text = "\n".join(message.content for message in seen_messages)
    assert "O que diz o projeto?" in prompt_text
    assert "Trecho oficial." in prompt_text


@pytest.mark.parametrize("model_output", [
    "not json",
    '{"answer":"Resposta","citations":[123]}',
])
def test_generate_answer_converts_invalid_model_output_to_provider_error(monkeypatch, model_output):
    configure_provider(monkeypatch)
    monkeypatch.setattr(providers, "ChatOpenAI", lambda **_kwargs: RunnableLambda(lambda _prompt: AIMessage(content=model_output)), raising=False)
    chunks = [{"chunk_id": "chunk-1", "document_id": "doc-1", "reference": "PL 1/2024", "title": "Projeto", "text": "Trecho oficial."}]

    with pytest.raises(ProviderError, match="resposta"):
        providers.generate_answer("Pergunta", chunks)
