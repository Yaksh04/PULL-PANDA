import os
import re
import subprocess
import requests
from dotenv import load_dotenv
from typing import Dict, List

# =====================================================
# 0. IMPORTS (Updated for latest LangChain ecosystem)
# =====================================================
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# 1. ENV & CONFIG
# =====================================================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("API_KEY")
owner = os.getenv("OWNER")
repo = os.getenv("REPO")
pr_number = os.getenv("PR_NUMBER")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not (GITHUB_TOKEN and owner and repo and pr_number):
    print("GitHub configuration incomplete. Some steps may be skipped.")

# Map extensions to languages
FILE_LANG_MAP = {
    "py": "python",
    "js": "javascript", "jsx": "javascript", "ts": "javascript", "tsx": "javascript",
    "java": "java",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "h": "cpp", "hpp": "cpp",
    "go": "go",
    "kt": "kotlin",
    "rs": "rust"
}

# Static analyzers for each language
ANALYZERS = {
    "python": [
        ("Pylint", ["pylint", "--exit-zero"]),
        ("Flake8", ["flake8", "--exit-zero"]),
        ("Bandit", ["bandit", "-r"]),
        ("Mypy", ["mypy", "--ignore-missing-imports"]),
    ],
    "javascript": [("ESLint", ["eslint", "--max-warnings=0"])],
    "cpp": [("Cppcheck", ["cppcheck", "--enable=all", "--quiet"])],
    "java": [("Checkstyle", ["checkstyle", "-c", "/google_checks.xml"])],
}


# =====================================================
# 2. GITHUB HELPERS (with auto comment posting)
# =====================================================
def fetch_pr_data(owner, repo, pr_number, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API Error: {response.json()}")
    return response.json()


def fetch_pr_diff(owner, repo, pr_number, token):
    pr_data = fetch_pr_data(owner, repo, pr_number, token)
    diff_url = pr_data["diff_url"]
    diff = requests.get(diff_url, headers={"Authorization": f"token {token}"}).text
    return diff, pr_data["title"]


def post_review_comment(owner, repo, pr_number, token, review_body):
    """
    Posts the AI-generated review as a comment on the GitHub Pull Request.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "AI-PR-Reviewer"
    }
    payload = {"body": review_body}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        data = response.json()
        print(f"Review posted: {data.get('html_url', 'URL not found')}")
        return data
    else:
        print(f"Failed to post review. GitHub response: {response.status_code}")
        print(response.text)
        return None


# =====================================================
# 3. STATIC ANALYSIS
# =====================================================
def get_changed_files_and_languages(diff_text: str) -> Dict[str, List[str]]:
    file_paths = re.findall(r'\+\+\+ b/(.*)', diff_text)
    changed = {}
    for path in file_paths:
        ext = path.split('.')[-1].lower()
        lang = FILE_LANG_MAP.get(ext)
        if lang:
            changed.setdefault(lang, []).append(path)
    return changed


def run_static_analysis(diff_text: str) -> str:
    changed = get_changed_files_and_languages(diff_text)
    if not changed:
        return "No supported language files detected."

    results = []
    for lang, files in changed.items():
        results.append(f"=== Static Analysis for {lang.upper()} ===")
        analyzers = ANALYZERS.get(lang, [])
        if not analyzers:
            results.append(f"No analyzer configured for {lang}")
            continue

        for name, base_cmd in analyzers:
            existing_files = [f for f in files if os.path.exists(f)]
            if not existing_files:
                results.append(f"| {name}: Skipped (files not found locally).")
                continue

            cmd = base_cmd + existing_files
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                out = proc.stdout.strip()
                err = proc.stderr.strip()
                if out or err:
                    results.append(f"| {name} Output:\n```\n{out or err}\n```")
                else:
                    results.append(f"| {name}: No issues found.")
            except FileNotFoundError:
                results.append(f"| {name}: Tool not installed.")
            except subprocess.TimeoutExpired:
                results.append(f"| {name}: Timed out.")
            except Exception as e:
                results.append(f"| {name}: Error - {e}")

    return "\n\n".join(results)


# =====================================================
# 4. RAG INDEXING + RETRIEVAL
# =====================================================
def index_repository(repo_path: str = ".", persist_dir: str = "./repo_index") -> None:
    print("Building vector index for repository...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".py", ".js", ".cpp", ".java", ".md")):
                try:
                    path = os.path.join(root, file)
                    loader = TextLoader(path, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Skipped {file}: {e}")

    if not documents:
        print("No documents found to index.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)

    print(f"Repository indexed and saved at: {persist_dir}")


def load_vector_index(persist_dir: str = "./repo_index") -> Chroma:
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Vector index not found at {persist_dir}. Please run index_repository() first.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def query_repo_context(query: str, k: int = 4, persist_dir: str = "./repo_index", max_unique_chunks: int = 3) -> str:
    vectordb = load_vector_index(persist_dir)
    docs = vectordb.similarity_search(query, k=k)
    seen = set()
    unique_texts = []
    for d in docs:
        content = getattr(d, "page_content", None) or str(d)
        normalized = content.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_texts.append(normalized)
        if len(unique_texts) >= max_unique_chunks:
            break
    return "\n\n".join(unique_texts)


# =====================================================
# 5. GROQ + STRUCTURED REVIEW FORMAT
# =====================================================
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
except Exception:
    print("llama-3.3-70b-versatile not available, switching to llama-3.1-8b-instant.")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

parser = StrOutputParser()

structured_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior software engineer reviewing a GitHub Pull Request. "
     "Write the output ONLY in this format:\n\n"
     "Code Review: <PR Title>\n\n"
     "Issues\n"
     "<bullet or numbered list of key problems found in code>\n\n"
     "Suggestions\n"
     "<bullet or numbered list of improvements or refactorings>\n\n"
     "Verdict\n"
     "<final evaluation with summary and optionally improved code snippets>\n\n"
     "Be precise, detailed, and professional."),
    ("human",
     "Repository context:\n{context}\n\n"
     "Pull Request Diff:\n{diff}\n\n"
     "Static Analysis Results:\n{static}\n\n"
     "PR Title: {pr_title}\n\n"
     "Generate the structured review now.")
])

review_chain = structured_prompt | llm | parser


def safe_truncate(text: str, max_len: int = 8000) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... (truncated)"


# =====================================================
# 6. MAIN EXECUTION (Agentic RAG Workflow)
# =====================================================
if __name__ == "__main__":
    try:
        if not all([GITHUB_TOKEN, owner, repo, pr_number]):
            raise Exception("Missing GitHub config in .env")

        print("Fetching PR diff and title from GitHub...")
        diff, pr_title = fetch_pr_diff(owner, repo, pr_number, GITHUB_TOKEN)
        print(f"PR Title: {pr_title}\n")

        print("Running targeted static analysis...")
        static_results = run_static_analysis(diff)
        print("Static analysis complete.\n")

        print("Checking repository index...")
        if not os.path.exists("./repo_index"):
            index_repository(".", "./repo_index")
        else:
            print("Existing index found.\n")

        print("Retrieving repository context (RAG)...")
        repo_context = query_repo_context("code structure and utilities", k=8, persist_dir="./repo_index", max_unique_chunks=3)
        print("Context retrieved.\n")

        print("Generating AI structured PR review...")
        review = review_chain.invoke({
            "context": safe_truncate(repo_context, 3000),
            "diff": safe_truncate(diff, 3000),
            "static": safe_truncate(static_results, 2000),
            "pr_title": pr_title
        })

        print("\n==============================")
        print("AI STRUCTURED CODE REVIEW")
        print("==============================\n")
        print(review)
        print("\n==============================\n")

        print("Posting structured review to GitHub...")
        comment_body = f"{review}\n\n(Generated via AI Code Reviewer with RAG Context)"
        comment = post_review_comment(owner, repo, pr_number, GITHUB_TOKEN, comment_body)

        if comment:
            print("Successfully commented on the PR.")
        else:
            print("Review generated but failed to post to GitHub.")

    except Exception as e:
        print("Error:", str(e))
