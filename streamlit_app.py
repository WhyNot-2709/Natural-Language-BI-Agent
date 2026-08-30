import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Natural-Language BI Agent", page_icon="📊")
st.title("📊 Natural-Language BI Agent")
st.caption("Ask a business question in plain English")

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None
if "last_report" not in st.session_state:
    st.session_state.last_report = None


def render_report(report: dict) -> None:
    narrative = report["narrative_summary"].replace("`", "")
    rows = report.get("raw_data_preview") or []

    with st.container(border=True):
        st.subheader("📊 Answer")
        if len(rows) == 1 and len(rows[0]) == 1:
            label, value = next(iter(rows[0].items()))
            display_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            st.metric(label=label.replace("_", " ").title(), value=display_value)
            st.caption(narrative)
        else:
            st.write(narrative)

        if report.get("chart_spec"):
            st.vega_lite_chart(report["chart_spec"], use_container_width=True)

    with st.expander("🔍 SQL used"):
        st.code(report["sql_used"], language="sql")

    if rows:
        st.dataframe(rows, use_container_width=True)

    if report.get("confidence_note"):
        st.info(report["confidence_note"])


def render_approval_request(thread_id: str, approval_request: dict) -> None:
    with st.container(border=True):
        st.warning("⚠️ This query needs approval before it runs.")
        st.write(f"**Reason:** {approval_request['reason']}")
        st.code(approval_request["sql"], language="sql")

        validation = approval_request["validation"]
        col1, col2 = st.columns(2)
        col1.metric("Estimated rows", validation.get("estimated_row_count"))
        col2.metric("Restricted column?", "Yes" if validation.get("touches_restricted_column") else "No")

        reviewer_note = st.text_input("Reviewer note (optional)", key="reviewer_note")
        col_approve, col_reject = st.columns(2)
        if col_approve.button("✅ Approve", type="primary", use_container_width=True):
            _resume(thread_id, approved=True, reviewer_note=reviewer_note)
        if col_reject.button("❌ Reject", use_container_width=True):
            _resume(thread_id, approved=False, reviewer_note=reviewer_note)


def _resume(thread_id: str, approved: bool, reviewer_note: str) -> None:
    response = requests.post(
        f"{API_BASE_URL}/queries/{thread_id}/resume",
        json={"approved": approved, "reviewer_note": reviewer_note or None},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()

    st.session_state.pending_approval = None
    if result["status"] == "pending_approval":
        st.session_state.pending_approval = (thread_id, result["approval_request"])
    else:
        st.session_state.last_report = result["report"]
    st.rerun()


# --- main interaction ---

if st.session_state.pending_approval:
    thread_id, approval_request = st.session_state.pending_approval
    render_approval_request(thread_id, approval_request)
else:
    with st.form("question_form"):
        question = st.text_input("Ask a question about the business", placeholder="e.g. What are our top 5 best-selling products by revenue?")
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    if submitted and question:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_BASE_URL}/queries", json={"question": question}, timeout=300)
            response.raise_for_status()
            result = response.json()

        if result["status"] == "pending_approval":
            st.session_state.pending_approval = (result["thread_id"], result["approval_request"])
            st.rerun()
        else:
            st.session_state.last_report = result["report"]

    if st.session_state.last_report:
        render_report(st.session_state.last_report)   