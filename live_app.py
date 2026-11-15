"""Local operations console for the authenticated payment API."""
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FraudGraph · Payment Operations", page_icon="🛡️", layout="wide")
st.title("FraudGraph · Payment Operations")
st.caption("Live card-payment review • tokenized identities • human decisions • durable audit trail")
default_token = os.environ.get("FRAUD_DASHBOARD_TOKEN", "")
config = Path(".local-live.json")
if not default_token and config.exists():
    default_token = json.loads(config.read_text(encoding="utf-8"))["keys"]["reviewer"]["token"]
base = st.sidebar.text_input("API endpoint", os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000"))
token = st.sidebar.text_input("API credential", default_token, type="password")
st.sidebar.caption("API roles enforce access. Keep this console on localhost or a protected internal network.")
if not token:
    st.info("Start the API with python scripts/start_live.py api, then supply an authorized credential.")
    st.stop()


def call(method, endpoint, **kwargs):
    try:
        with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=8) as client:
            response = client.request(method, endpoint, **kwargs)
        if response.status_code != 200:
            st.warning(f"API {response.status_code}: {response.json().get('detail', 'Request failed')}")
            return None
        return response.json()
    except (httpx.HTTPError, ValueError):
        st.warning("API unavailable. Check the service and endpoint; no action was assumed successful.")
        return None


operations, reviews, explorer, quality, audit = st.tabs([
    "Live operations", "Review queue", "Payment explorer", "Model & drift", "Audit history"])


@st.fragment(run_every="3s")
def live_panel():
    summary = call("GET", "/v1/stats")
    if not summary:
        return
    a, b, c, d = st.columns(4)
    a.metric("Payments scored", f"{summary['payments']:,}")
    b.metric("Open reviews", summary["cases"].get("open", 0))
    c.metric("Investigating", summary["cases"].get("investigating", 0))
    d.metric("Recent scoring p95", f"{summary['scoring_p95_ms']:.1f} ms" if summary["scoring_p95_ms"] is not None else "—")
    st.caption(f"Active: {summary['model']} · version {summary['model_version'][:12]} · updated {datetime.now(UTC):%H:%M:%S} UTC")
    recent = call("GET", "/v1/payments?limit=100")
    if recent and recent["items"]:
        frame = pd.DataFrame([{"payment_id": r["payment_id"], "received": pd.to_datetime(r["received_at"], unit="s", utc=True),
                               "amount_USD": r["payment"]["amount_minor"]/100, "score": r["risk_score"],
                               "decision": r["decision"], "card": r["payment"]["card_id"],
                               "merchant": r["payment"]["merchant_id"]} for r in recent["items"]])
        st.plotly_chart(px.scatter(frame, x="received", y="score", color="decision", size="amount_USD",
                                  hover_data=["payment_id", "card", "merchant"],
                                  color_discrete_map={"review": "#ef4444", "pass": "#14b8a6"}),
                        width="stretch", key="live_scatter")
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info("Waiting for payments. Run: python scripts/start_live.py replay --limit 200 --rate 2")
    if summary["drift_status"] == "review":
        st.warning("Recent features differ from training. Investigate replay speed, traffic mix, and model applicability.")
    st.caption("Pass/review are advisory outputs. This application does not authorize, decline, or move money.")


with operations:
    live_panel()

with reviews:
    status = st.selectbox("Case status", ["open", "investigating", "closed_fraud", "closed_legitimate"])
    st.button("Refresh queue")
    result = call("GET", f"/v1/cases?status={status}")
    cases = result["items"] if result else []
    if cases:
        st.dataframe(pd.DataFrame([{"case": c["case_id"], "payment": c["payment_id"], "score": c["score"],
                                   "reasons": ", ".join(c["reasons"]), "version": c["version"]} for c in cases]),
                     hide_index=True, width="stretch")
        selected = st.selectbox("Case to investigate", [c["case_id"] for c in cases])
        case = next(c for c in cases if c["case_id"] == selected)
        st.write("Payment", case["payment"])
        st.write("Observed evidence / policy reasons", case["reasons"])
        st.caption("These signals and policy triggers are not causal explanations of GraphSAGE.")
        allowed = {"open": ["investigating", "closed_fraud", "closed_legitimate"],
                   "investigating": ["open", "closed_fraud", "closed_legitimate"],
                   "closed_fraud": ["investigating"], "closed_legitimate": ["investigating"]}
        with st.form("review_decision"):
            next_status = st.selectbox("New status", allowed[case["status"]])
            note = st.text_area("Investigation note", max_chars=1000)
            submit = st.form_submit_button("Save analyst decision")
        if submit:
            if len(note.strip()) < 3:
                st.error("Enter an investigation note.")
            else:
                updated = call("PATCH", f"/v1/cases/{selected}",
                               json={"expected_version": case["version"], "status": next_status, "note": note.strip()})
                if updated:
                    st.success(f"Saved case version {updated['version']}. The audit trail records this change.")
    else:
        st.info("No cases in this status.")

with explorer:
    payment_id = st.text_input("Payment ID", placeholder="replay_00000001")
    if st.button("Load payment") and payment_id:
        result = call("GET", f"/v1/payments/{payment_id}")
        st.session_state["inspected_payment"] = result
        st.session_state["inspected_graph"] = call("GET", f"/v1/payments/{payment_id}/graph") if result else None
    inspected = st.session_state.get("inspected_payment")
    if inspected:
        left, right = st.columns(2)
        left.metric("Baseline score", f"{inspected['scores']['relational']:.4f}")
        right.metric("GraphSAGE shadow score", f"{inspected['scores']['graphsage']:.4f}")
        st.write("Decision", inspected["decision"])
        st.write("Evidence", inspected["evidence"])
        with st.expander("Recorded scoring context"):
            st.json(inspected)
        data = st.session_state.get("inspected_graph")
        if data:
            graph = nx.Graph()
            for item in data["items"]:
                p = item["payment"]
                for field in ["merchant_id", "device_id", "ip_token"]:
                    graph.add_edge(p["card_id"], p[field])
            pos = nx.spring_layout(graph, seed=42)
            xs, ys = [], []
            for a, b in graph.edges:
                xs.extend([pos[a][0], pos[b][0], None])
                ys.extend([pos[a][1], pos[b][1], None])
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#94a3b8", "width": 1},
                                      hoverinfo="skip", showlegend=False))
            nodes = list(graph)
            colors = {"card": "#38bdf8", "merchant": "#a78bfa", "device": "#fbbf24", "ip": "#34d399"}
            fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], text=nodes,
                                    mode="markers", marker={"size": 12, "color": [
                                        colors.get(n.split("_")[0], "#888") for n in nodes]},
                                    hovertemplate="%{text}<extra></extra>", showlegend=False))
            fig.update_layout(height=500, xaxis={"visible": False}, yaxis={"visible": False},
                              title="Prior shared card, merchant, device, and IP-token connections")
            st.plotly_chart(fig, width="stretch")
        manual_note = st.text_input("Reason to open a manual case")
        if st.button("Open case for this payment"):
            if len(manual_note.strip()) >= 3:
                created = call("POST", f"/v1/payments/{inspected['payment_id']}/case", json={"note": manual_note.strip()})
                if created:
                    st.success(f"Case available: {created['case_id']}")
            else:
                st.error("Enter a reason of at least 3 characters.")

with quality:
    if st.button("Refresh model and drift"):
        st.session_state["model_info"] = call("GET", "/v1/model")
        st.session_state["drift_info"] = call("GET", "/v1/stats")
    model = st.session_state.get("model_info")
    if model:
        st.caption("Held-out synthetic data only. These scores do not establish real-world accuracy.")
        st.dataframe(pd.DataFrame([{"model": name, "test_AP": result["test"]["average_precision"],
                                   "test_recall": result["test"]["recall_at_validation_threshold"],
                                   "test_FPR": result["test"]["observed_fpr"]}
                                  for name, result in model["held_out_metrics"].items()]), hide_index=True)
        st.code(f"Model fingerprint: {model['version']}\nDataset SHA-256: {model['dataset_sha256']}")
    drift = st.session_state.get("drift_info")
    if drift:
        st.write("Recent distribution status", drift["drift_status"])
        if drift["drift"]:
            st.plotly_chart(px.bar(pd.DataFrame(drift["drift"]), x="psi", y="feature", orientation="h"), width="stretch")
        st.caption("PSI > 0.25 is a monitoring heuristic, not a statistical test or automatic retraining trigger.")

with audit:
    st.caption("Analyst authorization required. Actor identity comes from the API credential.")
    if st.button("Load recent audit entries"):
        result = call("GET", "/v1/audit?limit=200")
        if result:
            st.dataframe(pd.json_normalize(result["items"]), width="stretch", hide_index=True)
    if st.button("Verify audit hash chain"):
        result = call("GET", "/v1/audit/verify")
        if result:
            st.json(result)
