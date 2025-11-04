# selector_runner.py
# Iterative prompt selector runner

from selector import IterativePromptSelector
from selector import process_pr_with_selector

def run_selector(pr_numbers, load_previous=True):
    selector = IterativePromptSelector()
    if load_previous:
        selector.load_state()

    results = []
    for pr in pr_numbers:
        try:
            res = process_pr_with_selector(selector, pr)
            results.append(res)
        except Exception as e:
            print(f"Failed to process PR #{pr}: {e}")
            continue

    print("\nFINAL ITERATIVE SELECTOR REPORT")
    for r in results:
        print(f"PR #{r['pr_number']}: {r['chosen_prompt']} -> Score: {r['score']}")

    selector.save_state()
    return results, selector
