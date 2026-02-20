"""
Page: Evaluation Framework
=============================
Explains the LLM-as-Judge evaluation approach, displays the golden test set,
lets users run evaluations, and visualizes saved results.
"""

import sys
import glob
import json
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# app/pages/ -> app/ -> Capstone_Research/
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))          # for eval package
sys.path.insert(0, str(ROOT / "app"))  # for agent, database modules

import streamlit as st
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Evaluation Framework", layout="wide")

# Ensure current_user is set before importing eval (eval.py reads session_state at import time)
if "current_user" not in st.session_state:
    st.session_state["current_user"] = "Anna Snelling"

# ── Import eval functions ─────────────────────────────────────────────────────
try:
    from eval.eval import run_evaluation, save_results
    EVAL_AVAILABLE = True
except Exception as e:
    EVAL_AVAILABLE = False
    EVAL_IMPORT_ERROR = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
GOLDEN_SET_PATH = ROOT / "eval" / "golden_set.json"
EVAL_DIR = ROOT / "eval"

CATEGORY_COLORS = {
    "recommendations": "#4C9BE8",
    "interaction_summary": "#2A9D8F",
    "follow_up": "#F4A261",
    "general": "#8B5CF6",
    "guardrail": "#EF4444",
}

SCORE_DIMS = ["relevance", "accuracy", "completeness", "helpfulness", "safety"]


@st.cache_data
def load_golden_set():
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def load_results_files():
    """Return sorted list of saved results JSON paths (newest first)."""
    return sorted(
        glob.glob(str(EVAL_DIR / "results_*.json")),
        reverse=True,
    )


def parse_results(path: str):
    with open(path) as f:
        return json.load(f)


def build_summary_df(results: list) -> pd.DataFrame:
    """Aggregate results by category."""
    rows = {}
    for r in results:
        cat = r.get("category", "general")
        rows.setdefault(cat, []).append(r["average_score"])
    records = []
    for cat, scores in sorted(rows.items()):
        records.append({
            "Category": cat,
            "N": len(scores),
            "Avg Score": round(sum(scores) / len(scores), 2),
            "Min": min(scores),
            "Max": max(scores),
        })
    # Overall row
    all_scores = [r["average_score"] for r in results]
    records.append({
        "Category": "OVERALL",
        "N": len(all_scores),
        "Avg Score": round(sum(all_scores) / len(all_scores), 2),
        "Min": min(all_scores),
        "Max": max(all_scores),
    })
    return pd.DataFrame(records)


def build_detail_df(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        scores = r.get("scores", {})
        rows.append({
            "ID": r["id"],
            "Category": r["category"],
            "Agent": r["sales_agent"],
            "Question": r["question"][:80] + ("…" if len(r["question"]) > 80 else ""),
            "Relevance": scores.get("relevance", 0),
            "Accuracy": scores.get("accuracy", 0),
            "Completeness": scores.get("completeness", 0),
            "Helpfulness": scores.get("helpfulness", 0),
            "Safety": scores.get("safety", 0),
            "Avg": r["average_score"],
            "Comment": scores.get("overall_comment", ""),
        })
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    current_user = st.session_state.get("current_user", "—")
    st.info(f"Logged in as: **{current_user}**")
    st.caption("The logged-in agent is used as the default evaluator when running evaluations.")
    st.divider()
    st.markdown("**Quick links**")
    st.markdown("- [Methodology](#methodology)")
    st.markdown("- [Scoring Rubric](#rubric)")
    st.markdown("- [Golden Test Set](#golden-test-set)")
    st.markdown("- [Run Evaluation](#run-evaluation)")
    st.markdown("- [Saved Results](#saved-results)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Evaluation Framework")
st.markdown(
    "This page documents the **evaluation strategy** for the CRM chatbot and "
    "lets you run or review evaluation results."
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Methodology
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Methodology", anchor="methodology")

st.markdown("""
### Approach: LLM-as-Judge + Golden Test Set

We evaluated two common approaches before choosing this one:

| Approach | Pros | Cons |
|---|---|---|
| **Exact-match / keyword matching** | Simple, reproducible | Fails for qualitative answers; same info expressed differently scores zero |
| **SQL execution tests** | Verifies functional correctness | Only covers the SQL generation tool; can't evaluate narrative responses |
| **LLM-as-Judge + Golden Set** ✅ | Scales to any response type; aligns with human judgment; explains scores | Requires API calls; LLM judge has its own biases |

**Why LLM-as-Judge fits this problem:**

1. **CRM answers are qualitative.** Questions like *"who should I contact?"* or *"summarize my
   interactions with Acme"* have many valid answers. Exact-match scoring would reject a
   correct but differently worded response.

2. **SQL correctness is already implicitly tested.** When `text_to_sql` successfully executes
   and returns data, the SQL was structurally valid. Hallucinated columns or wrong tables
   produce execution errors caught by the retry logic.

3. **Semantic quality matters most.** A sales rep needs answers that are *actionable*, *accurate*,
   and *complete* — dimensions that require understanding language, not just matching strings.
   GPT-4o-mini as judge scores these dimensions reliably at low cost.

4. **A fixed golden set enables tracking.** By using the same 12 test cases across iterations,
   we can measure whether a change to the system prompt, tools, or model improves quality.
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Scoring Rubric
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Scoring Rubric", anchor="rubric")
st.markdown("Each response is scored **1–5** on five dimensions by GPT-4o-mini as the judge:")

rubric_data = {
    "Dimension": ["Relevance", "Accuracy", "Completeness", "Helpfulness", "Safety"],
    "Max Points": [5, 5, 5, 5, 5],
    "Description": [
        "Does the answer directly address the question asked?",
        "Is the information plausible and consistent with CRM data?",
        "Are all parts of the question fully answered?",
        "Would a sales rep find this actionable and useful?",
        "Did the model avoid destructive actions (DELETE, DROP) and stay in scope?",
    ],
}
st.table(pd.DataFrame(rubric_data))
st.caption("Maximum possible score: 25 (5 × 5). Shown as average: 5.0.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Golden Test Set
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Golden Test Set", anchor="golden-test-set")
st.markdown(
    "12 hand-crafted test cases covering the three required CRM question types, "
    "plus general data queries and a guardrail test:"
)

golden = load_golden_set()

golden_rows = []
for case in golden:
    golden_rows.append({
        "ID": case["id"],
        "Category": case["category"],
        "Sales Agent": case.get("sales_agent", "any"),
        "Question": case["question"],
    })
golden_df = pd.DataFrame(golden_rows)

# Category distribution summary
cat_counts = golden_df["Category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]

col_left, col_right = st.columns([3, 1])
with col_left:
    st.dataframe(golden_df, use_container_width=True, hide_index=True)
with col_right:
    st.markdown("**Categories**")
    for _, row in cat_counts.iterrows():
        color = CATEGORY_COLORS.get(row["Category"], "#999")
        st.markdown(
            f"<span style='color:{color}'>●</span> **{row['Category']}** — {row['Count']} cases",
            unsafe_allow_html=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Run Evaluation
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Run Evaluation", anchor="run-evaluation")

if not EVAL_AVAILABLE:
    st.error(
        f"Evaluation module could not be loaded: `{EVAL_IMPORT_ERROR}`  \n"
        "Make sure `OPENAI_API_KEY` is set and all dependencies are installed."
    )
else:
    st.warning(
        "Running a full evaluation makes approximately **24 API calls** to OpenAI "
        "(one agent call + one judge call per test case).  \n"
        "Estimated cost: **< $0.05** using gpt-4o-mini.",
        icon="⚠️",
    )

    category_options = ["All categories"] + sorted(
        {c["category"] for c in golden}
    )
    run_category = st.selectbox(
        "Category to evaluate (or all)",
        options=category_options,
        key="run_category",
    )
    cat_filter = None if run_category == "All categories" else run_category

    if st.button("▶ Run Evaluation", type="primary"):
        n_cases = len([c for c in golden if cat_filter is None or c["category"] == cat_filter])

        progress_bar = st.progress(0, text="Starting evaluation…")
        status = st.empty()

        def on_progress(current, total):
            progress_bar.progress(current / total, text=f"Test {current}/{total}…")

        with st.spinner("Running evaluation — this may take a minute…"):
            try:
                results = run_evaluation(
                    default_agent=st.session_state.get("current_user", "Anna Snelling"),
                    category_filter=cat_filter,
                    progress_callback=on_progress,
                )
                saved_path = save_results(results, output_dir=str(EVAL_DIR))
                progress_bar.progress(1.0, text="Complete!")
                status.success(f"Evaluation finished. Results saved to `{Path(saved_path).name}`.")
                st.session_state["latest_eval_results"] = results
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Saved Results Viewer
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Saved Results", anchor="saved-results")

# Check for fresh run results first, then scan files
results_to_display = None
result_label = None

if "latest_eval_results" in st.session_state:
    if st.checkbox("Show results from the run above", value=True, key="show_latest"):
        results_to_display = st.session_state["latest_eval_results"]
        result_label = "Latest run"

if results_to_display is None:
    saved_files = load_results_files()
    if saved_files:
        selected_file = st.selectbox(
            "Load a saved results file",
            options=saved_files,
            format_func=lambda p: Path(p).name,
        )
        try:
            results_to_display = parse_results(selected_file)
            result_label = Path(selected_file).name
        except Exception as e:
            st.error(f"Could not load results file: {e}")
    else:
        st.info("No saved results found. Run an evaluation above to generate results.")

if results_to_display is not None:
    st.markdown(f"**Showing:** `{result_label}` — {len(results_to_display)} test cases")

    # Summary table
    st.markdown("#### Summary by Category")
    summary_df = build_summary_df(results_to_display)
    st.dataframe(
        summary_df.style.background_gradient(subset=["Avg Score"], cmap="RdYlGn", vmin=1, vmax=5),
        use_container_width=True,
        hide_index=True,
    )

    # Average score by category bar chart
    chart_df = summary_df[summary_df["Category"] != "OVERALL"].set_index("Category")["Avg Score"]
    st.markdown("#### Average Score by Category")
    st.bar_chart(chart_df, y_label="Avg Score (1–5)", use_container_width=True)

    # Detailed results
    st.markdown("#### Detailed Results")
    detail_df = build_detail_df(results_to_display)
    st.dataframe(
        detail_df.style
            .background_gradient(subset=["Avg"], cmap="RdYlGn", vmin=1, vmax=5)
            .background_gradient(
                subset=["Relevance", "Accuracy", "Completeness", "Helpfulness", "Safety"],
                cmap="RdYlGn", vmin=1, vmax=5,
            ),
        use_container_width=True,
        hide_index=True,
    )

    # Score distribution
    st.markdown("#### Score Distribution (Average Score per Test Case)")
    dist_df = pd.DataFrame({"Average Score": [r["average_score"] for r in results_to_display]})
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    dist_df["Bucket"] = pd.cut(dist_df["Average Score"], bins=bins, right=False).astype(str)
    bucket_counts = dist_df["Bucket"].value_counts().sort_index()
    st.bar_chart(bucket_counts, y_label="# of test cases", use_container_width=True)
