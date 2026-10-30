from contextlib import nullcontext

import pytest

from app.db import save_document


def test_save_document_executes_chunk_batch_with_cursor():
    class Cursor:
        def __init__(self):
            self.query = None
            self.rows = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def executemany(self, query, rows):
            self.query = query
            self.rows = list(rows)

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.statements = []

        def transaction(self):
            return nullcontext()

        def execute(self, query, params):
            self.statements.append((query, params))

        def cursor(self):
            return self.cursor_instance

    connection = Connection()
    document = {"document_id": "doc-1", "reference": "PL 1/2024", "title": "Projeto", "url": "https://example.com"}
    chunks = [{"chunk_id": "chunk-1", "text": "Trecho do documento."}]
    embeddings = [[0.1, 0.2]]

    save_document(connection, document, chunks, embeddings)

    assert len(connection.statements) == 2
    assert connection.cursor_instance.rows[0][:3] == ("chunk-1", "doc-1", "Trecho do documento.")
    assert connection.cursor_instance.rows[0][3].to_list() == pytest.approx([0.1, 0.2])
