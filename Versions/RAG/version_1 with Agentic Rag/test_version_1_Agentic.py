"""
Pytest suite for version_1_Agentic.py

Unique tests cover:
- get_pr_number_from_args numeric argument success and failure cases
- get_pr_number_from_args list-fetch errors and empty-list edge case
- fetch_pr_diff success and non-200 behavior
- post_review_comment returns JSON body
- get_full_file_content: success, file-missing, read-error
- setup_agent_tools: tool creation and retriever formatting behavior
- main(): error when no diff fetched (prints critical error)
- main(): successful run posts comment (prints posted url)

Each test imports the module fresh with minimal fake langchain packages injected
so tests do not require external dependencies or network calls.
"""
from importlib import util
import importlib
import sys
import os
from types import SimpleNamespace, ModuleType
from pathlib import Path
import io
import zipfile
import json
import pytest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "version_1_Agentic.py")

def _fake_rag_loader_agentic():
    fake_mod = ModuleType("rag_loader_agentic")

    def fake_build_index_for_repo(*a, **k):
        return SimpleNamespace(as_retriever=lambda **_: SimpleNamespace(get_relevant_documents=lambda q: []))

    def fake_assemble_context(docs, char_limit=4000):
        # return "assembled context"
        return "CTX:" + " ".join(getattr(d, "page_content", str(d)) for d in docs)


    fake_mod.build_index_for_repo = fake_build_index_for_repo
    fake_mod.assemble_context = fake_assemble_context
    fake_mod.REPO_DOWNLOAD_DIR = Path("repo_download")

    sys.modules["rag_loader_agentic"] = fake_mod


# ---- Helpers to import module with lightweight fakes ----
def _make_fake_langchain_env():
    """
    Insert minimal fake packages/classes into sys.modules so the target module can import.
    Provides:
      - langchain_core.prompts.ChatPromptTemplate & MessagesPlaceholder
      - langchain_groq.ChatGroq
      - langchain.agents.AgentExecutor & create_tool_calling_agent
      - langchain.tools.Tool
    """
    # prompts
    prompts_mod = ModuleType("langchain_core.prompts")
    class FakeChatPromptTemplate:
        @staticmethod
        def from_messages(msgs):
            return {"_from_messages": msgs}
    prompts_mod.ChatPromptTemplate = FakeChatPromptTemplate
    prompts_mod.MessagesPlaceholder = lambda **k: {"_messages_placeholder": k}
    sys.modules["langchain_core.prompts"] = prompts_mod

    # groq
    groq_mod = ModuleType("langchain_groq")
    class FakeChatGroq:
        def __init__(self, *a, **k):
            pass
    groq_mod.ChatGroq = FakeChatGroq
    sys.modules["langchain_groq"] = groq_mod

    # agents
    agents_mod = ModuleType("langchain.agents")
    def fake_create_tool_calling_agent(llm, tools, agent_prompt):
        return {"agent": True, "llm": llm, "tools": tools, "prompt": agent_prompt}
    class FakeAgentExecutor:
        def __init__(self, agent=None, tools=None, verbose=False, handle_parsing_errors=False):
            self._agent = agent
            self._tools = tools
        def invoke(self, kwargs):
            # default behavior; tests can monkeypatch an instance method as needed
            return {"output": "default review"}
    agents_mod.create_tool_calling_agent = fake_create_tool_calling_agent
    agents_mod.AgentExecutor = FakeAgentExecutor
    sys.modules["langchain.agents"] = agents_mod

    # tools
    tools_mod = ModuleType("langchain.tools")
    class FakeTool:
        def __init__(self, name, func, description=""):
            self.name = name
            self.func = func
            self.description = description
    tools_mod.Tool = FakeTool
    sys.modules["langchain.tools"] = tools_mod


def _load_module_fresh(name="version_1_agentic"):
    """Load target module fresh under the given name (after injecting fakes)."""
    # Ensure environment keys exist for import-time check
    import os as _os
    _os.environ["GITHUB_TOKEN"] = _os.environ.get("GITHUB_TOKEN", "ghp_test")
    _os.environ["API_KEY"] = _os.environ.get("API_KEY", "groq_test")

    # insert fake langchain packages
    _make_fake_langchain_env()
    _fake_rag_loader_agentic()

    # Ensure fresh import
    if name in sys.modules:
        del sys.modules[name]
    spec = util.spec_from_file_location(name, MODULE_PATH)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---- Tests ----

def test_get_pr_number_with_numeric_argument_returns_number_and_url(monkeypatch):
    """Numeric PR arg that exists -> returns (int, html_url)."""
    mod = _load_module_fresh("version_1_agentic")
    # fake requests.get for single-PR check
    class Resp:
        status_code = 200
        def json(self):
            return {"html_url": "https://github/pr/1"}

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: Resp())

    result = mod.get_pr_number_from_args("owner", "repo", "t", pr_arg="1")
    assert result == (1, "https://github/pr/1")


def test_get_pr_number_with_numeric_argument_not_found_raises_ValueError(monkeypatch):
    """Numeric PR arg that does not exist -> raises ValueError."""
    mod = _load_module_fresh("v1_a_num_nf")
    class BadResp:
        status_code = 404
        def json(self):
            return {"message": "Not Found"}
    monkeypatch.setattr("version_1_agentic.requests.get", lambda *a, **k: BadResp(), raising=False)
    with pytest.raises(ValueError):
        mod.get_pr_number_from_args("o", "r", "t", pr_arg="999")


def test_get_pr_number_fetch_list_api_error_raises_ValueError(monkeypatch):
    """When fetching open PRs the API returns non-200 -> ValueError containing API response."""
    mod = _load_module_fresh("v1_a_list_err")
    class ErrResp:
        status_code = 500
        def json(self):
            return {"error": "server"}
    monkeypatch.setattr("version_1_agentic.requests.get", lambda *a, **k: ErrResp(), raising=False)
    with pytest.raises(ValueError) as ei:
        mod.get_pr_number_from_args("o", "r", "t", pr_arg=None)
    assert "GitHub API Error" in str(ei.value)


def test_get_pr_number_no_open_prs_raises_ValueError(monkeypatch):
    """List PR API returns 200 but empty -> raises 'No open PRs' ValueError."""
    mod = _load_module_fresh("v1_a_no_prs")
    class OkEmpty:
        status_code = 200
        def json(self):
            return []
    monkeypatch.setattr("version_1_agentic.requests.get", lambda *a, **k: OkEmpty(), raising=False)
    with pytest.raises(ValueError) as ei:
        mod.get_pr_number_from_args("o", "r", "t", pr_arg=None)
    assert "No open PRs" in str(ei.value)


def test_fetch_pr_diff_success_returns_text(monkeypatch):
    """fetch_pr_diff returns response.text when status_code == 200."""
    mod = _load_module_fresh("v1_fetch_ok")
    class R:
        status_code = 200
        text = "diff-content"
    monkeypatch.setattr("version_1_agentic.requests.get", lambda *a, **k: R(), raising=False)
    txt = mod.fetch_pr_diff("o", "r", 5, "t")
    assert txt == "diff-content"


def test_fetch_pr_diff_non_200_returns_empty_string_and_prints(monkeypatch, capsys):
    """Non-200 fetch_pr_diff returns empty string and prints an error line."""
    mod = _load_module_fresh("v1_fetch_404")
    class R:
        status_code = 404
        text = "not found"
    monkeypatch.setattr("version_1_agentic.requests.get", lambda *a, **k: R(), raising=False)
    txt = mod.fetch_pr_diff("o", "r", 7, "t")
    captured = capsys.readouterr()
    assert txt == ""
    assert "Error fetching diff" in captured.out or "❌ Error fetching diff" in captured.out


def test_post_review_comment_returns_github_json(monkeypatch):
    """post_review_comment returns response.json() payload."""
    mod = _load_module_fresh("v1_post_ok")
    class R:
        def json(self):
            return {"id": 123, "html_url": "http://comment"}
    monkeypatch.setattr("version_1_agentic.requests.post", lambda *a, **k: R(), raising=False)
    out = mod.post_review_comment("o", "r", 1, "t", "body text")
    assert isinstance(out, dict)
    assert out["html_url"] == "http://comment"


def test_get_full_file_content_returns_truncated_content_when_file_exists(tmp_path, monkeypatch):
    """If file exists under extracted repo top-level, return header + truncated content."""
    # Prepare module with fake langchain env
    mod = _load_module_fresh("v1_full_ok")
    # create repo_download/<topdir>/file.txt
    monkeypatch.chdir(tmp_path)
    repo_download = Path("repo_download")
    repo_download.mkdir()
    top = repo_download / "topdir"
    top.mkdir()
    (top / "notes.txt").write_text("x" * 5000, encoding="utf-8")
    # call function with relative path 'notes.txt' (file is under topdir)
    res = mod.get_full_file_content("notes.txt")
    assert "CONTENT OF notes.txt" in res
    assert len(res) <= len("CONTENT OF notes.txt\n") + 4000 + 100  # header + truncated body approx


def test_get_full_file_content_missing_file_returns_error_message(tmp_path, monkeypatch):
    """When requested file is absent, returns descriptive error message."""
    mod = _load_module_fresh("v1_full_missing")
    monkeypatch.chdir(tmp_path)
    # create repo_download with single top dir but no target file
    repo_download = Path("repo_download")
    repo_download.mkdir()
    (repo_download / "top").mkdir()
    out = mod.get_full_file_content("nope.py")
    assert "ERROR: File not found" in out


def test_get_full_file_content_read_error_returns_error_message(tmp_path, monkeypatch):
    """If reading the file raises an OSError, return an error string with reason."""
    mod = _load_module_fresh("v1_full_readerr")
    monkeypatch.chdir(tmp_path)
    repo_download = Path("repo_download")
    repo_download.mkdir()
    top = repo_download / "top"
    top.mkdir()
    f = top / "bad.txt"
    f.write_text("content")
    # monkeypatch Path.read_text to raise for this specific file
    orig = Path.read_text
    def fake_read(self, encoding="utf-8", errors="ignore"):
        if self.name == "bad.txt":
            raise OSError("disk error")
        return orig(self, encoding=encoding, errors=errors)
    monkeypatch.setattr(Path, "read_text", fake_read)
    out = mod.get_full_file_content("bad.txt")
    assert "ERROR: Could not read file bad.txt" in out


def test_setup_agent_tools_creates_tools_and_retriever_uses_assemble_context(monkeypatch):
    """setup_agent_tools should return two tools; retriever tool uses assemble_context to format docs."""
    mod = _load_module_fresh("v1_setup_tools")
    # prepare fake vectorstore with as_retriever
    class FakeRetriever:
        def get_relevant_documents(self, q):
            return [SimpleNamespace(page_content="AAA"), SimpleNamespace(page_content="BBB")]
    class FakeVectorstore:
        def as_retriever(self, search_kwargs=None):
            return FakeRetriever()
    # monkeypatch module.assemble_context to a simple joiner, to be called by tool.func
    monkeypatch.setattr("version_1_agentic.assemble_context", lambda docs, char_limit=4000: "CTX:" + " ".join(getattr(d, "page_content", str(d)) for d in docs), raising=False)
    tools = mod.setup_agent_tools(FakeVectorstore())
    assert isinstance(tools, list) and len(tools) == 2
    rag_tool, file_tool = tools[0], tools[1]
    assert rag_tool.name == "project_context_search"
    # calling rag_tool.func should return assembled context string
    ctx = rag_tool.func("some query")
    assert ctx.startswith("CTX:")
    assert "AAA" in ctx and "BBB" in ctx
    assert file_tool.name == "full_file_reader"
    # file_tool.func should be callable and delegate to module.get_full_file_content
    assert callable(file_tool.func)


def test_main_when_no_diff_fetched_prints_critical_error(monkeypatch, capsys):
    """
    main() should catch ValueError raised due to no diff fetched and print a critical error line.
    We stub many dependencies to keep control.
    """
    mod = _load_module_fresh("v1_main_nodiff")
    # stub build_index_for_repo to return a dummy vectorstore
    monkeypatch.setattr("version_1_agentic.build_index_for_repo", lambda *a, **k: SimpleNamespace(), raising=False)
    # stub setup_agent_tools to return empty tools (unused because fetch_pr_diff returns "")
    monkeypatch.setattr("version_1_agentic.setup_agent_tools", lambda *a, **k: [], raising=False)
    # make get_pr_number_from_args return a pr
    monkeypatch.setattr("version_1_agentic.get_pr_number_from_args", lambda *a, **k: (1, "url"), raising=False)
    # make fetch_pr_diff return empty string to trigger the "No diff fetched." path
    monkeypatch.setattr("version_1_agentic.fetch_pr_diff", lambda *a, **k: "", raising=False)

    monkeypatch.setattr(sys, "argv", ["version_1_Agentic.py", "owner", "repo"])

    # run main()
    mod.main()
    out = capsys.readouterr().out
    assert "Critical Error" in out or "❌ Critical Error" in out


def test_main_successful_run_posts_comment_and_prints_url(monkeypatch, capsys):
    """Full happy-path main run: agent returns review text and post_review_comment returns html_url."""
    mod = _load_module_fresh("version_1_agentic")
    # stub build_index_for_repo and setup_agent_tools
    monkeypatch.setattr("version_1_agentic.build_index_for_repo", lambda *a, **k: SimpleNamespace(), raising=False)
    monkeypatch.setattr("version_1_agentic.setup_agent_tools", lambda *a, **k: [], raising=False)
    # stub pr lookup and diff fetch
    monkeypatch.setattr("version_1_agentic.get_pr_number_from_args", lambda *a, **k: (11, "http://pr"), raising=False)
    monkeypatch.setattr("version_1_agentic.fetch_pr_diff", lambda *a, **k: "diff text", raising=False)
    # create a fake AgentExecutor that returns a review
    class FakeExec:
        def __init__(self, *a, **k): pass
        def invoke(self, kwargs):
            return {"output": "GENERATED REVIEW"}
    monkeypatch.setattr("version_1_agentic.AgentExecutor", FakeExec, raising=False)
    # stub create_tool_calling_agent to return a fake agent descriptor
    monkeypatch.setattr("version_1_agentic.create_tool_calling_agent", lambda *a, **k: {"agent": True}, raising=False)
    # stub ChatGroq constructor
    monkeypatch.setattr("version_1_agentic.ChatGroq", lambda *a, **k: SimpleNamespace(), raising=False)
    # stub post_review_comment to return html_url
    monkeypatch.setattr("version_1_agentic.post_review_comment", lambda *a, **k: {"html_url": "http://comment"}, raising=False)

    monkeypatch.setattr(sys, "argv", ["version_1_Agentic.py", "owner", "repo"])

    # run main()
    mod.main()
    out = capsys.readouterr().out
    assert "Review Generated" in out or "FINAL PULL-PANDA REVIEW" in out
    assert "Review posted" in out or "✅ Review posted" in out
