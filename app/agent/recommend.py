import streamlit as st
from typing import Dict, Any
from database import db_query


def recommend_contacts_handler(args: Dict[str, Any]) -> str:
    """
    Agentic tool that scores and ranks CRM accounts to recommend who to contact.

    Scoring formula (100-point scale):
      - Propensity to buy   (40 pts): likelihood the account will purchase
      - Revenue             (20 pts): normalized to max revenue in database
      - Days since contact  (40 pts): capped at 90 days — longer gap = higher priority

    Args:
        args: Dict with optional keys:
            - n (int): number of recommendations to return (default: 3)
            - sector (str): filter to a specific industry sector (optional)

    Returns:
        Formatted string with ranked recommendations and reasoning per account.
    """
    n = int(args.get("n", 3))
    sector_filter = args.get("sector", "").strip()

    sector_clause = ""
    if sector_filter:
        # safe because tool input comes from LLM, not raw user text, but we
        # still sanitize by stripping quotes and limiting to alphanumeric/spaces
        safe_sector = sector_filter.replace("'", "").replace('"', "")
        sector_clause = f"AND LOWER(a.sector) = LOWER('{safe_sector}')"

    sql = f"""
        SELECT
            a.account_id,
            a.account,
            a.sector,
            COALESCE(a.revenue, 0)            AS revenue,
            COALESCE(a.propensity_to_buy, 0)  AS propensity_to_buy,
            a.employees,
            vs.last_touch,
            COALESCE(
                DATE_DIFF('day', CAST(vs.last_touch AS DATE), CURRENT_DATE),
                90
            ) AS days_since_contact,
            vs.has_open_work,
            ROUND(
                (COALESCE(a.propensity_to_buy, 0) * 40.0)
                + (COALESCE(a.revenue, 0)
                   / (SELECT NULLIF(MAX(revenue), 0) FROM accounts WHERE revenue IS NOT NULL)
                   * 20.0)
                + (
                    LEAST(
                        COALESCE(DATE_DIFF('day', CAST(vs.last_touch AS DATE), CURRENT_DATE), 90),
                        90
                    ) / 90.0 * 40.0
                  ),
            1) AS contact_score
        FROM accounts a
        JOIN v_accounts_summary vs ON a.account_id = vs.account_id
        WHERE 1=1
        {sector_clause}
        ORDER BY contact_score DESC
        LIMIT {n}
    """

    try:
        df = db_query(sql)
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

    if df.empty:
        msg = "No accounts found"
        if sector_filter:
            msg += f" in sector '{sector_filter}'"
        return msg + "."

    lines = [f"**Top {len(df)} Contact Recommendations**\n"]

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        account   = row.get("account", "Unknown")
        sector    = row.get("sector", "N/A")
        revenue   = row.get("revenue", 0)
        propensity = row.get("propensity_to_buy", 0)
        days      = int(row.get("days_since_contact", 0))
        last_touch = row.get("last_touch", None)
        score     = row.get("contact_score", 0)
        has_work  = row.get("has_open_work", False)

        # Human-readable reason
        reasons = []
        if propensity >= 0.65:
            reasons.append(f"high propensity to buy ({propensity:.0%})")
        elif propensity >= 0.45:
            reasons.append(f"moderate propensity to buy ({propensity:.0%})")
        if revenue >= 5000:
            reasons.append(f"high-value account (${revenue:,.0f}M revenue)")
        elif revenue >= 1000:
            reasons.append(f"mid-market account (${revenue:,.0f}M revenue)")
        if days >= 90:
            reasons.append(f"relationship has gone cold ({days} days since last contact)")
        elif days >= 30:
            reasons.append(f"not contacted recently ({days} days ago)")
        if has_work:
            reasons.append("has an active deal in pipeline")
        if last_touch is None:
            reasons.append("never contacted — high-potential new outreach")

        reason_str = "; ".join(reasons) if reasons else "strong overall profile"

        last_str = str(last_touch)[:10] if last_touch is not None else "never"

        lines.append(
            f"**{rank}. {account}** ({sector}) — Score: {score}/100\n"
            f"   _Why contact:_ {reason_str}.\n"
            f"   Last contact: {last_str}"
        )

    return "\n\n".join(lines)
