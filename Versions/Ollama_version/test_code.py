

import importlib.util
import sys
import os
import json
import requests
import pytest

# Path to the target module file that tests will import under different module names.
# Use the local `code.py` in this folder so tests can load the same script under
# different module names while monkeypatching network calls.
MODULE_PATH = os.path.join(os.path.dirname(__file__), "code.py")



class FakeGetResp:
    def __init__(self, text):
        self.text = text


class FakePostResp:
    def __init__(self, lines):
        # lines: iterable of bytes
        self._lines = list(lines)

    def iter_lines(self, **kwargs):
        for l in self._lines:
            yield l


def _load_module(mod_name="ollama_code"):
    """Load the target module from MODULE_PATH freshly under mod_name."""
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    # register then execute so top-level code runs in that module namespace
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_missing_github_token_raises(monkeypatch):
    """No GITHUB_TOKEN in env => module import raises ValueError."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError):
        _load_module("code_missing_token")


def test_requests_get_raises_timeout_propagates(monkeypatch):
    """requests.get raises requests.exceptions.Timeout -> import should propagate."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    def raise_timeout(*a, **k):
        raise requests.exceptions.Timeout("simulated timeout")
    monkeypatch.setattr(requests, "get", raise_timeout)
    with pytest.raises(requests.exceptions.Timeout):
        _load_module("code_get_timeout")


def test_requests_get_headers_include_auth_and_accept(monkeypatch):
    """Ensure code sends both Authorization and Accept headers to GitHub API."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok-ABC")
    def fake_get(url, headers=None, timeout=None):
        assert headers is not None
        assert "Authorization" in headers and "tok-ABC" in headers["Authorization"]
        assert headers.get("Accept") == "application/vnd.github.v3.diff"
        return FakeGetResp("DIFF-CONTENT")
    monkeypatch.setattr(requests, "get", fake_get)
    # make post return empty stream so module completes without error
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp([]))
    mod = _load_module("code_check_headers")
    assert hasattr(mod, "REVIEW_TEXT")
    assert mod.REVIEW_TEXT.strip() == ""


def test_post_stream_multiple_response_lines_concatenate_preserve_order(monkeypatch):
    """Multiple per-line JSON responses should be concatenated in order."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff"))
    lines = [
        b'{"response":"First part. "}\n',
        b'{"response":"Second part."}\n'
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp(lines))
    mod = _load_module("code_multi_concat")
    assert "First part." in mod.REVIEW_TEXT
    assert "Second part." in mod.REVIEW_TEXT
    assert mod.REVIEW_TEXT.index("First part.") < mod.REVIEW_TEXT.index("Second part.")


def test_post_stream_unicode_response_decoding(monkeypatch):
    """Ensure unicode and emojis in response fields are decoded and concatenated."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff"))
    lines = [
        json.dumps({"response": "Handles unicode: ✓ "}).encode("utf-8"),
        json.dumps({"response": "Emoji: 🚀"}).encode("utf-8"),
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp(lines))
    mod = _load_module("code_unicode")
    assert "Handles unicode: ✓" in mod.REVIEW_TEXT
    assert "Emoji: 🚀" in mod.REVIEW_TEXT


def test_streaming_split_json_across_lines_ignored_boundary(monkeypatch):
    """Boundary: JSON object split across two iter_lines yields should be ignored (per-line parsing)."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff-split"))
    # split JSON that cannot be parsed per-line
    lines = [b'{"res', b'ponse":"split"}\n']
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp(lines))
    mod = _load_module("code_split_boundary")
    # Split JSON should not produce a response entry
    assert mod.REVIEW_TEXT.strip() == ""


def test_streaming_mixed_invalid_and_valid_lines_only_response_used(monkeypatch):
    """Mixed invalid JSON, JSON without 'response', and valid response JSON -> only response values included."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff-mix"))
    lines = [
        b'invalid-json\n',
        b'{"not_response":"x"}\n',
        b'{"response":"OK1"}\n',
        b'garbage\n',
        b'{"response":"OK2"}\n'
    ]
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp(lines))
    mod = _load_module("code_mixed_unique")
    assert "OK1" in mod.REVIEW_TEXT and "OK2" in mod.REVIEW_TEXT
    assert "not_response" not in mod.REVIEW_TEXT
    assert "invalid-json" not in mod.REVIEW_TEXT


def test_streaming_empty_iter_lines_results_empty_review_text(monkeypatch):
    """Empty stream (no lines) should produce empty REVIEW_TEXT."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff-none"))
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakePostResp([]))
    mod = _load_module("code_empty_stream_unique")
    assert getattr(mod, "REVIEW_TEXT", "").strip() == ""


def test_response_iter_lines_raising_error_propagates(monkeypatch):
    """If response.iter_lines raises, the exception should propagate during import/execution."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff-exc"))
    class BadResp:
        def iter_lines(self, **kwargs):
            raise RuntimeError("stream failure")
    monkeypatch.setattr(requests, "post", lambda *a, **k: BadResp())
    with pytest.raises(RuntimeError):
        _load_module("code_stream_error")


def test_requests_post_raises_timeout_propagates(monkeypatch):
    """requests.post raising Timeout should propagate during module import."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeGetResp("diff"))
    def raise_post_timeout(*a, **k):
        raise requests.exceptions.Timeout("post timeout")
    monkeypatch.setattr(requests, "post", raise_post_timeout)
    with pytest.raises(requests.exceptions.Timeout):
        _load_module("code_post_timeout")