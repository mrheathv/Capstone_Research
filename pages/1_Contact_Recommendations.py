"""
Page: Contact Recommendations
=================================
Explains the recommend_contacts scoring tool and shows a live ranked table
of all CRM accounts, with an interactive stacked-bar breakdown of score components.
"""

import sys
from pathlib import Path

# Add project root to path so database/ and agent/ modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from database import db_query

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Contact Recommendations", layout="wide")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    current_user = st.session_state.get("current_user", "—")
    st.info(f"Logged in as: **{current_user}**")
    st.divider()

    st.header("Filters")
    n_recs = st.slider("Accounts to show", min_value=1, max_value=30, value=10)

    try:
        sectors_df = db_query(
            "SELECT DISTINCT sector FROM accounts WHERE sector IS NOT NULL ORDER BY sector"
        )
        sector_options = ["All sectors"] + sectors_df["sector"].tolist()
    except Exception:
        sector_options = ["All sectors"]

    sector_choice = st.selectbox("Filter by sector", options=sector_options)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Contact Recommendations")
st.markdown(
    "This page explains the **`recommend_contacts` agentic tool** and shows a live "
    "ranking of all CRM accounts. The chatbot calls this tool automatically when you ask "
    "questions like *'I have some availability — who should I contact?'*"
)

# ── Scoring formula explanation ──────────────────────────────────────────────
st.subheader("How the Scoring Works")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Propensity to Buy", "40 pts", help="From accounts.propensity_to_buy (0–1 scale)")
    st.caption(
        "How likely is this account to purchase? Sourced from the CRM's propensity model. "
        "An account with propensity=0.75 earns 0.75 × 40 = **30 pts**."
    )
with col2:
    st.metric("Revenue (normalized)", "20 pts", help="accounts.revenue / max(revenue) × 20")
    st.caption(
        "Prioritizes high-value accounts. Revenue is normalized to the highest-revenue "
        "account in the database so the scale is always 0–20 pts."
    )
with col3:
    st.metric("Days Since Last Contact", "40 pts", help="Capped at 90 days")
    st.caption(
        "Relationships that have gone cold get higher priority. Days are capped at 90 to "
        "prevent one very old account from dominating all others."
    )

st.info(
    "**Total score = Propensity pts + Revenue pts + Recency pts (max 100)**  \n"
    "Accounts are ranked highest-to-lowest. A higher score = higher priority to contact.",
    icon="ℹ️",
)

st.divider()

# ── Scoring SQL ──────────────────────────────────────────────────────────────
sector_clause = ""
if sector_choice != "All sectors":
    safe_sector = sector_choice.replace("'", "").replace('"', "")
    sector_clause = f"AND LOWER(a.sector) = LOWER('{safe_sector}')"

SCORING_SQL = f"""
SELECT
    a.account_id,
    a.account,
    a.sector,
    COALESCE(a.revenue, 0)                                               AS revenue,
    COALESCE(a.propensity_to_buy, 0)                                     AS propensity_to_buy,
    vs.last_touch,
    COALESCE(DATE_DIFF('day', CAST(vs.last_touch AS DATE), CURRENT_DATE), 90)
                                                                         AS days_since_contact,
    -- Component score breakdown (for chart)
    ROUND(COALESCE(a.propensity_to_buy, 0) * 40.0, 1)                   AS propensity_pts,
    ROUND(
        COALESCE(a.revenue, 0)
        / (SELECT NULLIF(MAX(revenue), 0) FROM accounts WHERE revenue IS NOT NULL)
        * 20.0, 1)                                                       AS revenue_pts,
    ROUND(
        LEAST(COALESCE(DATE_DIFF('day', CAST(vs.last_touch AS DATE), CURRENT_DATE), 90), 90)
        / 90.0 * 40.0, 1)                                               AS recency_pts,
    -- Total score
    ROUND(
        (COALESCE(a.propensity_to_buy, 0) * 40.0)
        + (COALESCE(a.revenue, 0)
           / (SELECT NULLIF(MAX(revenue), 0) FROM accounts WHERE revenue IS NOT NULL) * 20.0)
        + (LEAST(COALESCE(DATE_DIFF('day', CAST(vs.last_touch AS DATE), CURRENT_DATE), 90), 90)
           / 90.0 * 40.0),
    1)                                                                   AS contact_score
FROM accounts a
JOIN v_accounts_summary vs ON a.account_id = vs.account_id
WHERE 1=1 {sector_clause}
ORDER BY contact_score DESC
LIMIT {n_recs}
"""

# ── Run query ─────────────────────────────────────────────────────────────────
try:
    df = db_query(SCORING_SQL)
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.warning("No accounts found with the current filters.")
    st.stop()

# Add rank column
df.insert(0, "Rank", range(1, len(df) + 1))

# ── Ranked table ─────────────────────────────────────────────────────────────
st.subheader(f"Top {len(df)} Accounts to Contact")

display_df = df[[
    "Rank", "account", "sector", "contact_score",
    "propensity_to_buy", "revenue", "days_since_contact", "last_touch"
]].rename(columns={
    "account": "Account",
    "sector": "Sector",
    "contact_score": "Score / 100",
    "propensity_to_buy": "Propensity",
    "revenue": "Revenue ($M)",
    "days_since_contact": "Days Since Contact",
    "last_touch": "Last Contact",
})

st.dataframe(
    display_df.style
        .background_gradient(subset=["Score / 100"], cmap="YlGn")
        .format({
            "Propensity": "{:.1%}",
            "Revenue ($M)": "${:,.0f}M",
            "Score / 100": "{:.1f}",
        }),
    use_container_width=True,
    hide_index=True,
)

# ── Stacked bar chart ─────────────────────────────────────────────────────────
st.subheader("Score Component Breakdown")
st.caption(
    "Each bar shows how the 100-point score is split across the three components. "
    "Longer bars = higher priority."
)

try:
    import altair as alt

    chart_df = (
        df[["account", "propensity_pts", "revenue_pts", "recency_pts"]]
        .rename(columns={
            "account": "Account",
            "propensity_pts": "Propensity (40 pts max)",
            "revenue_pts": "Revenue (20 pts max)",
            "recency_pts": "Recency (40 pts max)",
        })
        .melt(id_vars="Account", var_name="Component", value_name="Score")
    )

    # Sort order: highest total first
    account_order = df["account"].tolist()

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("sum(Score):Q", title="Total Score (out of 100)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Account:N", sort=account_order, title=None),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(
                    domain=["Propensity (40 pts max)", "Revenue (20 pts max)", "Recency (40 pts max)"],
                    range=["#4C9BE8", "#F4A261", "#2A9D8F"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["Account:N", "Component:N", alt.Tooltip("Score:Q", format=".1f")],
        )
        .properties(height=max(250, len(df) * 28))
    )

    st.altair_chart(chart, use_container_width=True)

except ImportError:
    # Altair not available — fall back to a simple pandas bar chart
    chart_df = df.set_index("account")[["propensity_pts", "revenue_pts", "recency_pts"]]
    chart_df.columns = ["Propensity", "Revenue", "Recency"]
    st.bar_chart(chart_df)

# ── SQL expander ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("View scoring SQL query"):
    st.markdown(
        "This is the exact SQL the `recommend_contacts` tool runs against the DuckDB database. "
        "The three component scores are summed to produce the final 0–100 contact score."
    )
    st.code(SCORING_SQL.strip(), language="sql")

# ══════════════════════════════════════════════════════════════════════════════
# AIDA FUNNEL
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Sales Funnel (AIDA)")
st.markdown(
    "Classifies each sales agent's accounts into the four AIDA stages based on "
    "pipeline deal stage and interaction history."
)

with st.expander("How stages are defined"):
    st.markdown("""
| Stage | Condition |
|---|---|
| **Action** | Agent has a **Won** deal with this account |
| **Desire** | Agent has an active **Engaging** deal (no Won) |
| **Interest** | Agent has a **Prospecting** deal **and** ≥ 1 recorded interaction |
| **Awareness** | Agent has a Prospecting deal but 0 interactions, or only Lost deals |

Each account is placed in the highest stage the agent has reached with it.
""")

AIDA_SQL = """
WITH agent_accounts AS (
    SELECT
        sp.sales_agent,
        sp.account_id,
        MAX(CASE WHEN sp.deal_stage = 'Won'         THEN 3
                 WHEN sp.deal_stage = 'Engaging'    THEN 2
                 WHEN sp.deal_stage = 'Prospecting' THEN 1
                 ELSE 0 END)        AS best_stage_rank,
        COUNT(DISTINCT i.timestamp) AS interaction_count
    FROM sales_pipeline sp
    LEFT JOIN interactions i ON sp.account_id = i.account_id
    GROUP BY sp.sales_agent, sp.account_id
),
aida AS (
    SELECT
        sales_agent,
        account_id,
        CASE
            WHEN best_stage_rank = 3                             THEN 'Action'
            WHEN best_stage_rank = 2                             THEN 'Desire'
            WHEN best_stage_rank = 1 AND interaction_count > 0  THEN 'Interest'
            ELSE                                                      'Awareness'
        END AS aida_stage
    FROM agent_accounts
)
SELECT
    sales_agent,
    COUNT(CASE WHEN aida_stage = 'Awareness' THEN 1 END) AS awareness,
    COUNT(CASE WHEN aida_stage = 'Interest'  THEN 1 END) AS interest,
    COUNT(CASE WHEN aida_stage = 'Desire'    THEN 1 END) AS desire,
    COUNT(CASE WHEN aida_stage = 'Action'    THEN 1 END) AS action,
    COUNT(*)                                              AS total
FROM aida
GROUP BY sales_agent
ORDER BY sales_agent
"""

AIDA_DRILLDOWN_SQL = """
WITH agent_accounts AS (
    SELECT
        sp.sales_agent,
        sp.account_id,
        a.account,
        a.sector,
        MAX(CASE WHEN sp.deal_stage = 'Won'         THEN 3
                 WHEN sp.deal_stage = 'Engaging'    THEN 2
                 WHEN sp.deal_stage = 'Prospecting' THEN 1
                 ELSE 0 END)        AS best_stage_rank,
        COUNT(DISTINCT i.timestamp) AS interaction_count
    FROM sales_pipeline sp
    JOIN accounts a ON sp.account_id = a.account_id
    LEFT JOIN interactions i ON sp.account_id = i.account_id
    GROUP BY sp.sales_agent, sp.account_id, a.account, a.sector
)
SELECT
    sales_agent,
    account,
    sector,
    CASE
        WHEN best_stage_rank = 3                             THEN 'Action'
        WHEN best_stage_rank = 2                             THEN 'Desire'
        WHEN best_stage_rank = 1 AND interaction_count > 0  THEN 'Interest'
        ELSE                                                      'Awareness'
    END AS aida_stage,
    interaction_count
FROM agent_accounts
ORDER BY sales_agent, aida_stage, account
"""

try:
    aida_df        = db_query(AIDA_SQL)
    aida_detail_df = db_query(AIDA_DRILLDOWN_SQL)
except Exception as e:
    st.error(f"AIDA query error: {e}")
    st.stop()

# ── Agent selector ────────────────────────────────────────────────────────────
all_agents = sorted(aida_df["sales_agent"].tolist())
agent_options = ["All agents"] + all_agents
selected_agent = st.selectbox("Select a sales agent", agent_options, key="aida_agent")

# ── Metric cards ──────────────────────────────────────────────────────────────
if selected_agent == "All agents":
    row = aida_df[["awareness", "interest", "desire", "action"]].sum()
else:
    mask = aida_df["sales_agent"] == selected_agent
    row = aida_df[mask][["awareness", "interest", "desire", "action"]].iloc[0] \
        if mask.any() else {"awareness": 0, "interest": 0, "desire": 0, "action": 0}

m1, m2, m3, m4 = st.columns(4)
m1.metric("Awareness", int(row["awareness"]), help="Prospecting stage, no interactions yet")
m2.metric("Interest",  int(row["interest"]),  help="Prospecting stage with ≥1 interaction")
m3.metric("Desire",    int(row["desire"]),    help="Active Engaging deal")
m4.metric("Action",    int(row["action"]),    help="Won deal")

st.caption(
    "Interactions are logged at the account level, not per sales agent. "
    "When multiple reps work the same account, each sees that account's interactions counted."
)

# ── Stacked bar chart ─────────────────────────────────────────────────────────
try:
    chart_source = aida_df if selected_agent == "All agents" \
        else aida_df[aida_df["sales_agent"] == selected_agent]

    chart_df = chart_source[
        ["sales_agent", "awareness", "interest", "desire", "action"]
    ].melt(id_vars="sales_agent", var_name="Stage", value_name="Accounts")
    chart_df["Stage"] = chart_df["Stage"].str.capitalize()

    stage_order  = ["Awareness", "Interest", "Desire", "Action"]
    agent_order  = chart_source.sort_values("action", ascending=False)["sales_agent"].tolist()

    aida_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("sum(Accounts):Q", title="Accounts"),
            y=alt.Y("sales_agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "Stage:N",
                sort=stage_order,
                scale=alt.Scale(
                    domain=stage_order,
                    range=["#94A3B8", "#60A5FA", "#F59E0B", "#22C55E"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("Stage:N", sort="ascending"),
            tooltip=["sales_agent:N", "Stage:N", "Accounts:Q"],
        )
        .properties(height=max(200, len(chart_source) * 28))
    )
    st.altair_chart(aida_chart, use_container_width=True)

except ImportError:
    chart_source = aida_df if selected_agent == "All agents" \
        else aida_df[aida_df["sales_agent"] == selected_agent]
    st.bar_chart(chart_source.set_index("sales_agent")[["awareness", "interest", "desire", "action"]])

# ── Drilldown table ───────────────────────────────────────────────────────────
st.markdown("#### Account Drilldown")

stage_filter = st.selectbox(
    "Filter by AIDA stage",
    ["All stages", "Awareness", "Interest", "Desire", "Action"],
    key="aida_stage_filter",
)

drilldown = aida_detail_df.copy()
if selected_agent != "All agents":
    drilldown = drilldown[drilldown["sales_agent"] == selected_agent]
if stage_filter != "All stages":
    drilldown = drilldown[drilldown["aida_stage"] == stage_filter]

st.dataframe(
    drilldown.rename(columns={
        "sales_agent":       "Agent",
        "account":           "Account",
        "sector":            "Sector",
        "aida_stage":        "AIDA Stage",
        "interaction_count": "Interactions",
    }),
    use_container_width=True,
    hide_index=True,
)
