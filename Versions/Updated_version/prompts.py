# prompts.py
# Prompt registry (all prompts centrally defined)

from langchain.prompts import ChatPromptTemplate

# --- NEW PROMPT TEMPLATE CORE ---
PROMPT_CORE = """
Analyze the PR diff and the static analysis results provided below. 
Produce a professional, structured GitHub PR review, focusing on correctness, security, maintainability, and readability. 
Make all suggestions actionable, using specific file/line references where possible.

PR Diff (truncated):\n{diff}\n
Static Analysis Results:\n{static}\n
"""
# -------------------------------

def get_prompts():
    """Return all available prompts in a dictionary"""
    prompts = {}

    # 1. Zero-shot
    prompts["Zero-shot"] = ChatPromptTemplate.from_messages([
        ("system", "You are a software engineer reviewing a PR."),
        ("human", "Review the following GitHub Pull Request diff and static analysis results. Point out bugs, mistakes, and give suggestions.\n" + PROMPT_CORE)
    ])

    # 2. Few-shot
    prompts["Few-shot"] = ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer who writes concise, example-driven reviews. Use the static analysis results as primary evidence."),
        ("human",
         "Here are examples of good short PR reviews (follow their style):\n\n"
         "Example 1:\n- Bug: `fetchData` may throw on null response.\n- Suggestion: Add null-check and early return, add unit test.\n\n"
         "Example 2:\n- Code style: inconsistent naming `myVar` vs `my_var`.\n- Suggestion: follow project lint rules and rename variables.\n\n"
         "Now review the following diff and analysis in the same style:\n" + PROMPT_CORE)
    ])

    # 3. Chain-of-Thought
    prompts["Chain-of-Thought"] = ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer. Use step-by-step analysis then produce a concise final review."),
        ("human",
         "Steps:\n"
         "1) Summarize what changed, citing static analysis findings.\n"
         "2) Identify functional/logic bugs.\n"
         "3) Identify style/maintainability issues (especially from static analysis).\n"
         "4) Suggest prioritized fixes.\n\n"
         "Provide a short final review after the analysis.\n" + PROMPT_CORE +
         "\nIMPORTANT: Provide only the final review in the 'Final Review' section at the end.")
    ])

    # 4. Tree-of-Thought
    prompts["Tree-of-Thought"] = ChatPromptTemplate.from_messages([
        ("system", "You are an experienced reviewer. Explore the diff through multiple focused branches, then consolidate."),
        ("human",
         "Create 4 branches of analysis and then consolidate:\n"
         "- Branch A: Functional correctness (bugs, edge cases).\n"
         "- Branch B: Code quality (readability, naming, duplication) - **Focus heavily on Static Analysis findings**.\n"
         "- Branch C: Performance & security concerns.\n"
         "- Branch D: Tests, docs, and CI considerations.\n\n"
         "For each branch, list findings (short bullets). Then produce a consolidated structured review.\n" + PROMPT_CORE)
    ])

    # 5. Self-Consistency
    prompts["Self-Consistency"] = ChatPromptTemplate.from_messages([
        ("system", "You are an expert reviewer. Generate multiple candidate reviews and select the best, leveraging the static analysis results."),
        ("human",
         "Task:\n"
         "1) Generate 3 concise reviews (label them Review A, B, C).\n"
         "2) Compare them for clarity, correctness, and actionability, specifically assessing how well they handle the static analysis findings.\n"
         "3) Return the best review and a 1-line reason why you picked it.\n" + PROMPT_CORE)
    ])

    # 6. Reflection
    prompts["Reflection"] = ChatPromptTemplate.from_messages([
        ("system", "You are a senior engineer. Produce a review, critique it, then refine it."),
        ("human",
         "Steps:\n"
         "1) Write an initial concise review, referencing the static analysis where appropriate.\n"
         "2) Critique that review for any missing issues or unclear suggestions.\n"
         "3) Produce a final refined review incorporating the critique.\n" + PROMPT_CORE)
    ])

    # 7. Meta
    prompts["Meta"] = ChatPromptTemplate.from_messages([
        ("system", "You are an expert senior software engineer performing a professional GitHub Pull Request review. You MUST cite static analysis results where relevant."),
        ("human",
         "Analyze the PR diff and static analysis across these dimensions:\n"
         "- Summary: short overview (1-2 lines).\n"
         "- Critical Bugs: list issues that must be fixed before merge. Include high-severity static analysis findings.\n"
         "- Important Improvements: performance, security, correctness.\n"
         "- Code Quality & Maintainability: naming, complexity, duplication. Address all stylistic issues found by static analysis.\n"
         "- Tests & CI: missing tests or flakiness.\n"
         "- Positive notes: what was done well.\n\n"
         "Make all suggestions actionable (code snippets or exact lines when possible). Keep it concise and prioritize items by severity.\n" + PROMPT_CORE)
    ])

    return prompts