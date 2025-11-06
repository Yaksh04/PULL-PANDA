"""
Pytest tests for rag_loader_traditional.py

Covers:
- download_repo_files: single file, nested dirs, root non-200, file download non-200
- build_index_for_repo: force_rebuild with empty download -> fallback, saving created index,
  loading existing index when present, embeddings constructed with expected model_name
- assemble_context: normal concatenation, exact-boundary inclusion, object without page_content,
  zero char_limit and too-small limits

All external interactions (requests, filesystem, FAISS, embeddings) are mocked.
Each test follows Arrange-Act-Assert (AAA) and has a clear descriptive name.

Each test prints its name when executed so pytest output shows which test case is running.
"""

import importlib
import sys
from types import SimpleNamespace
import builtins
import os
from pathlib import Path
import pytest

# Helper to import the target module fresh for each test
def _import_rag_module_fresh():
    """Import rag_loader_traditional as a fresh module (clears existing entry)."""
    if "rag_loader_traditional" in sys.modules:
        del sys.modules["rag_loader_traditional"]
    return importlib.import_module("rag_loader_traditional")


# ===== Tests for download_repo_files() =====
def test_download_repo_files_single_root_text_file_returns_content(monkeypatch):
    print("Running test_download_repo_files_single_root_text_file_returns_content")
    # Arrange
    # Fake requests.get responses for root listing and file download
    def fake_get(url, headers=None):
        class Resp:
            def __init__(self, status_code, payload=None, text=None):
                self.status_code = status_code
                self._payload = payload
                self.text = text
            def json(self):
                return self._payload

        if url.endswith("/contents/"):
            # Root listing with single .py file
            return Resp(200, payload=[{"type": "file", "name": "hello.py", "download_url": "https://cdn/hello.py", "path": "hello.py"}])
        if url == "https://cdn/hello.py":
            return Resp(200, text="print('hello')")
        raise AssertionError("Unexpected URL: " + url)

    monkeypatch.setattr("requests.get", fake_get)
    # Act
    rag = _import_rag_module_fresh()
    result = rag.download_repo_files("owner", "repo", "token")
    # Assert
    assert isinstance(result, list)
    assert result == ["print('hello')"]


def test_download_repo_files_traverses_nested_directories_and_skips_non_text(monkeypatch):
    print("Running test_download_repo_files_traverses_nested_directories_and_skips_non_text")
    # Arrange
    root_items = [
        {"type": "dir", "name": "src", "path": "src"},
        {"type": "file", "name": "binary.bin", "download_url": "https://cdn/binary.bin", "path": "binary.bin"}
    ]
    src_items = [
        {"type": "file", "name": "README.md", "download_url": "https://cdn/README.md", "path": "src/README.md"},
        {"type": "file", "name": "tool.exe", "download_url": "https://cdn/tool.exe", "path": "src/tool.exe"}
    ]

    def fake_get(url, headers=None):
        class Resp:
            def __init__(self, status_code, payload=None, text=None):
                self.status_code = status_code
                self._payload = payload
                self.text = text
            def json(self):
                return self._payload

        if url.endswith("/contents/"):
            return Resp(200, payload=root_items)
        if url.endswith("/contents/src"):
            return Resp(200, payload=src_items)
        if url == "https://cdn/README.md":
            return Resp(200, text="# Project")
        if url == "https://cdn/binary.bin":
            return Resp(200, text="\x00\x01")  # binary content, but extension unsupported so skipped
        if url == "https://cdn/tool.exe":
            return Resp(200, text="EXE")
        raise AssertionError("Unexpected URL: " + url)

    monkeypatch.setattr("requests.get", fake_get)
    # Act
    rag = _import_rag_module_fresh()
    result = rag.download_repo_files("owner", "repo", "token")
    # Assert
    assert isinstance(result, list)
    # Only README.md should be included (md is supported); binary.bin and tool.exe skipped
    assert result == ["# Project"]


def test_download_repo_files_root_non_200_returns_empty_and_prints_error(monkeypatch, capsys):
    print("Running test_download_repo_files_root_non_200_returns_empty_and_prints_error")
    # Arrange
    def fake_get(url, headers=None):
        class Resp:
            status_code = 500
            text = "Server error"
            def json(self):
                return {"message": "server error"}
        return Resp()

    monkeypatch.setattr("requests.get", fake_get)
    rag = _import_rag_module_fresh()
    # Act
    result = rag.download_repo_files("owner", "repo", "token")
    # Assert
    assert result == []
    captured = capsys.readouterr()
    assert "Error fetching" in captured.out


def test_download_repo_files_file_download_non_200_is_skipped(monkeypatch):
    print("Running test_download_repo_files_file_download_non_200_is_skipped")
    # Arrange
    def fake_get(url, headers=None):
        class Resp:
            def __init__(self, status_code, payload=None, text=None):
                self.status_code = status_code
                self._payload = payload
                self.text = text
            def json(self):
                return self._payload

        if url.endswith("/contents/"):
            return Resp(200, payload=[{"type": "file", "name": "a.py", "download_url": "https://cdn/a.py", "path": "a.py"}])
        if url == "https://cdn/a.py":
            return Resp(404, text="Not found")
        raise AssertionError("Unexpected URL: " + url)

    monkeypatch.setattr("requests.get", fake_get)
    # Act
    rag = _import_rag_module_fresh()
    result = rag.download_repo_files("owner", "repo", "token")
    # Assert
    assert result == []


# ===== Tests for build_index_for_repo() =====
def test_build_index_force_rebuild_creates_index_and_uses_fallback_when_no_files(monkeypatch):
    print("Running test_build_index_force_rebuild_creates_index_and_uses_fallback_when_no_files")
    # Arrange
    # Force Path.exists to report that index file does not exist
    monkeypatch.setattr(Path, "exists", lambda self: False)
    # Prevent actual directory creation
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    recorded = {}

    # Fake embeddings constructor to capture model_name
    class FakeEmb:
        def __init__(self, model_name=None):
            recorded["emb_model"] = model_name

    # Fake FAISS with from_texts and save_local
    class FakeVectorStore:
        def __init__(self):
            pass
        @classmethod
        def from_texts(cls, texts, embeddings):
            recorded["from_texts_texts"] = list(texts)
            return cls()
        def save_local(self, path):
            recorded["saved_to"] = str(path)

    # Inject fakes into sys.modules so import picks them up
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", SimpleNamespace(HuggingFaceEmbeddings=FakeEmb))
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", SimpleNamespace(FAISS=FakeVectorStore))
    # Also mock download_repo_files to return empty list so fallback used
    monkeypatch.setattr("rag_loader_traditional.download_repo_files", lambda owner, repo, token: [])
    # Act
    rag = _import_rag_module_fresh()
    vec = rag.build_index_for_repo("owner", "repo", "token", force_rebuild=True)
    # Assert
    assert "emb_model" in recorded
    assert recorded["emb_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert recorded["from_texts_texts"] == ["Initial dummy text"]
    assert "saved_to" in recorded
    assert isinstance(vec, FakeVectorStore)


def test_build_index_loads_existing_index_when_present_and_not_force_rebuild(monkeypatch):
    print("Running test_build_index_loads_existing_index_when_present_and_not_force_rebuild")
    # Arrange
    # Make Path.exists return True when checking index file path
    def fake_exists(self):
        # return True so the implementation will attempt to load an existing index
        return True
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)

    recorded = {}
    class FakeVectorLoad:
        @classmethod
        def load_local(cls, index_path, embeddings, allow_dangerous_deserialization=False):
            recorded["loaded_index_path"] = str(index_path)
            return cls()
        @classmethod
        def from_texts(cls, *a, **k):
            raise AssertionError("from_texts should not be called when index exists and not force_rebuild")

    class FakeEmb2:
        def __init__(self, model_name=None):
            recorded["emb_model2"] = model_name

    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", SimpleNamespace(HuggingFaceEmbeddings=FakeEmb2))
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", SimpleNamespace(FAISS=FakeVectorLoad))
    # Act
    rag = _import_rag_module_fresh()
    vec = rag.build_index_for_repo("owner", "repo", "token", force_rebuild=False)
    # Assert
    assert "loaded_index_path" in recorded
    assert isinstance(vec, FakeVectorLoad)
    # The implementation loads from the repository-specific directory; ensure rag_indexes is present
    assert "rag_indexes" in recorded["loaded_index_path"]
    # and the path includes the owner_repo portion
    assert recorded["loaded_index_path"].endswith("owner_repo")


def test_build_index_creates_index_with_texts_returned_by_download(monkeypatch):
    print("Running test_build_index_creates_index_with_texts_returned_by_download")
    # Arrange
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(os, "makedirs", lambda *a, **k: None)
    recorded = {}

    class FakeEmb:
        def __init__(self, model_name=None):
            recorded["emb_model"] = model_name

    class FakeVS:
        @classmethod
        def from_texts(cls, texts, embeddings):
            recorded["from_texts_texts"] = list(texts)
            return cls()
        def save_local(self, path):
            recorded["saved_to"] = str(path)

    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", SimpleNamespace(HuggingFaceEmbeddings=FakeEmb))
    monkeypatch.setitem(sys.modules, "langchain_community.vectorstores", SimpleNamespace(FAISS=FakeVS))
    # Provide actual file texts to be indexed by replacing download_repo_files after module import
    rag = _import_rag_module_fresh()
    monkeypatch.setattr(rag, "download_repo_files", lambda owner, repo, token: ["A", "B", "C"])
    # Act
    vec = rag.build_index_for_repo("ownerX", "repoY", "token-123", force_rebuild=True)
    # Assert
    assert recorded["from_texts_texts"] == ["A", "B", "C"]
    assert recorded["emb_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    # saved path should end with the folder name combining ownerX_repoY
    assert recorded["saved_to"].endswith(f"rag_indexes{os.sep}ownerX_repoY")


# ===== Tests for assemble_context() =====
def test_assemble_context_concatenates_page_content_until_limit_and_preserves_order():
    print("Running test_assemble_context_concatenates_page_content_until_limit_and_preserves_order")
    # Arrange
    from rag_loader_traditional import assemble_context
    class Doc:
        def __init__(self, text):
            self.page_content = text
    docs = [Doc("first"), Doc("second"), Doc("third")]
    # Act
    result = assemble_context(docs, char_limit=100)
    # Assert
    assert "first" in result and "second" in result and "third" in result
    # Parts separated by double newline as implemented
    assert result.count("\n\n") >= 2


def test_assemble_context_respects_exact_char_limit_includes_equal_size_item_and_stops_after():
    print("Running test_assemble_context_respects_exact_char_limit_includes_equal_size_item_and_stops_after")
    # Arrange
    from rag_loader_traditional import assemble_context
    class Doc:
        def __init__(self, text):
            self.page_content = text
    # Compose two docs where first length equals char limit for inclusion
    d1 = Doc("A" * 10)  # 10 chars
    d2 = Doc("B" * 5)
    limit = len(d1.page_content)
    # Act
    result = assemble_context([d1, d2], char_limit=limit)
    # Assert
    assert "A" * 10 in result
    # second doc should not be included because adding it would exceed limit
    assert "B" * 5 not in result


def test_assemble_context_handles_objects_without_page_content_by_using_str_representation():
    print("Running test_assemble_context_handles_objects_without_page_content_by_using_str_representation")
    # Arrange
    from rag_loader_traditional import assemble_context
    class Obj:
        def __str__(self):
            return "stringified-object"
    docs = [Obj()]
    # Act
    result = assemble_context(docs, char_limit=100)
    # Assert
    assert "stringified-object" in result


def test_assemble_context_with_zero_char_limit_returns_empty_string():
    print("Running test_assemble_context_with_zero_char_limit_returns_empty_string")
    # Arrange
    from rag_loader_traditional import assemble_context
    class Doc:
        def __init__(self, text):
            self.page_content = text
    docs = [Doc("anything")]
    # Act
    result = assemble_context(docs, char_limit=0)
    # Assert
    assert result == ""