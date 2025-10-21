# version_1.py (Updated for LangChain >= 0.2)
import os
import sys
import requests
from dotenv import load_dotenv

# ✅ New imports for LangChain v0.2+
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ✅ Use new community packages for FAISS & embeddings (if used in rag_loader)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import local RAG loader (should also use new imports inside)
from rag_loader import build_index_for_repo, assemble_context


# ------------------------------
# 1. Load API Keys
# ------------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("API_KEY")

if not GITHUB_TOKEN or not GROQ_API_KEY:
    raise ValueError("❌ Missing API keys in .env file")


# ------------------------------
# 2. Fetch Latest PR Number
# ------------------------------
def get_latest_pr(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&sort=created&direction=desc"
    headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API Error: {response.json()}")

    prs = response.json()
    if not prs:
        raise Exception("⚠️ No open PRs found in this repository.")

    latest_pr = prs[0]
    return latest_pr["number"], latest_pr["html_url"]


# ------------------------------
# 3. Fetch PR Diff
# ------------------------------
def fetch_pr_diff(owner, repo, pr_number, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff"   # ✅ Return unified diff
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("❌ Error fetching diff:", response.status_code, response.text)
        return ""

    diff = response.text
    print("📏 Diff length:", len(diff))
    if len(diff) < 50:
        print("⚠️ Warning: Diff seems too small, check if PR actually has changes.")

    with open("latest_pr.diff", "w", encoding="utf-8") as f:
        f.write(diff)

    return diff


# ------------------------------
# 4. Post Review Comment
# ------------------------------
def post_review_comment(owner, repo, pr_number, token, review_body):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    payload = {"body": review_body}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


# ------------------------------
# 5. Groq AI Reviewer
# ------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=GROQ_API_KEY
)
parser = StrOutputParser()

# --- MODIFIED PROMPT ---
system_instruction = (
    "You are PULL-PANDA, a senior software engineer specialized in code review. "
    "Your primary goal is to review the code changes contained ONLY in the provided PR diff. "
    "Use the 'Project Context' only to ensure the changes align with existing project logic, "
    "naming conventions, and style. If the PR diff is in a different language than the "
    "context, ignore the context completely and focus only on the quality, correctness, "
    "and style of the new code in the diff. "
    "Structure your response clearly with specific findings (Issues, Suggestions) and a final verdict."
)

review_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "Project Context:\n{context}\n\nPR Diff:\n\n{diff}\n\n"
              "Please provide a detailed, actionable code review.")
])
# --- END MODIFIED PROMPT ---


# Compose chain
review_chain = review_prompt | llm | parser


# ------------------------------
# 6. Main Logic
# ------------------------------
if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            print("Usage: python version_1.py <owner> <repo>")
            sys.exit(1)

        owner, repo = sys.argv[1], sys.argv[2]

        # 1. Build or load RAG index for repo
        print("📦 Building/loading RAG index for repo (this may take a minute)...")
        rag = build_index_for_repo(owner, repo, GITHUB_TOKEN, force_rebuild=False)

        # 2. Get latest PR from GitHub
        pr_number, pr_url = get_latest_pr(owner, repo, GITHUB_TOKEN)
        print(f"🔸 Found latest PR: {pr_url}")

        # 3. Fetch PR diff
        diff_text = fetch_pr_diff(owner, repo, pr_number, GITHUB_TOKEN)
        if not diff_text:
            raise Exception("No diff fetched.")

        # 4. Create retriever and get context
        retriever = rag.as_retriever(search_kwargs={"k": 6})
        
        # --- FIX from last interaction applied here ---
        # Retrieve relevant Document objects
        retrieved_documents = retriever.get_relevant_documents(diff_text)
        print("Retrieved", len(retrieved_documents), "context chunks.")

        # Assemble context (This function expects Document objects from LangChain)
        context_text = assemble_context(retrieved_documents, char_limit=3000)
        # --- END FIX ---


        # 5. Generate AI review
        prompt_vars = {"context": context_text, "diff": diff_text[:80_000]}
        review_text = review_chain.invoke(prompt_vars)
        print("=== 🧠 AI REVIEW ===")
        print(review_text)
        print("====================")

        # 6. Post review comment on PR
        comment = post_review_comment(owner, repo, pr_number, GITHUB_TOKEN, review_text)
        if "html_url" in comment:
            print(f"✅ Review posted: {comment['html_url']}")
        else:
            print(f"⚠️ Failed to post review: {comment}")

    except Exception as e:
        print("❌ Error:", e)
