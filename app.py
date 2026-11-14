"""Run: streamlit run app.py"""
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

st.set_page_config(page_title="FraudGraph | Investigation", page_icon="🔎", layout="wide")
st.title("FraudGraph · Investigation workspace")
st.caption("Chronological model comparison and account connections. Scores prioritize review; they are not proof of fraud.")
run = Path(st.sidebar.text_input("Run directory", "runs/demo"))
if not (run / "metrics.json").is_file():
    st.info("Train a run first: python -m fraudgraph.cli train data/demo.csv --output runs/demo")
    st.stop()
try:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(run / "predictions.csv", dtype={"source": str, "target": str, "transaction_id": str},
                     keep_default_na=False)
    df.timestamp = pd.to_datetime(df.timestamp, utc=True)
except (OSError, ValueError) as exc:
    st.error(f"Cannot open run: {exc}")
    st.stop()
model = st.sidebar.selectbox("Scoring model", list(metrics["models"]),
                             index=list(metrics["models"]).index(manifest["champion"]))
score_column = f"score_{model}"
threshold = manifest["thresholds"][model]
test = df[(df.split == "test") & (df.label >= 0)]
summary = metrics["models"][model]["test"]
cards = st.columns(4)
cards[0].metric("Transactions", f"{len(df):,}")
cards[1].metric("Test average precision", f"{summary['average_precision']:.3f}" if summary.get("average_precision") is not None else "N/A")
cards[2].metric("Test recall", f"{summary['recall_at_validation_threshold']:.1%}" if summary.get("recall_at_validation_threshold") is not None else "N/A")
cards[3].metric("Observed test FPR", f"{summary['observed_fpr']:.1%}" if summary.get("observed_fpr") is not None else "N/A")
compare, investigate, features, methodology = st.tabs(["Model comparison", "Investigate", "Features", "Methodology"])
with compare:
    rows = []
    for key, result in metrics["models"].items():
        rows.append({"model": key, **{k: result["test"].get(k) for k in
                     ["average_precision", "roc_auc", "recall_at_validation_threshold", "observed_fpr"]},
                     "unseen_AP": result["test_any_unseen_account"].get("average_precision"),
                     "inference_seconds": result["cpu_full_history_inference_seconds"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig = go.Figure()
    if test.label.nunique() == 2:
        for key in metrics["models"]:
            p, r, _ = precision_recall_curve(test.label, test[f"score_{key}"])
            fig.add_trace(go.Scatter(x=r, y=p, name=key, mode="lines"))
        fig.add_hline(y=test.label.mean(), line_dash="dot", annotation_text="Test prevalence")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Held-out precision–recall curves")
        st.plotly_chart(fig, width="stretch")
    st.json(metrics["models"][model]["test_any_unseen_account"], expanded=False)
    scenario_metrics = metrics["models"][model].get("test_scenarios", {})
    if scenario_metrics:
        st.subheader("Synthetic scenario checks")
        st.dataframe(pd.DataFrame([{
            "scenario": name, "rows": result["n"],
            "recall": result.get("recall_at_validation_threshold"),
            "false_positive_rate": result.get("observed_fpr")}
            for name, result in scenario_metrics.items()]), hide_index=True, width="stretch")
        st.caption("Scenario names are evaluation-only ground truth and are excluded from model inputs.")
with investigate:
    st.caption(f"Validation-selected alert threshold: {threshold:.6f}. Graph shows activity at or before the selected transaction.")
    split = st.selectbox("Period", ["test", "validation", "train", "all"])
    view = df if split == "all" else df[df.split == split]
    alerts_only = st.checkbox("Only threshold alerts", value=False)
    if alerts_only:
        view = view[view[score_column] >= threshold]
    view = view.sort_values(score_column, ascending=False)
    st.dataframe(view[["transaction_id", "timestamp", "source", "target", "amount", "label", score_column]].head(500),
                 hide_index=True, width="stretch")
    st.download_button("Download filtered scores", view.to_csv(index=False), "review_queue.csv", "text/csv")
    if not view.empty:
        selected = st.selectbox("Investigate transaction (top 500)", view.transaction_id.head(500))
        row = df[df.transaction_id == selected].iloc[0]
        context = df[(df.timestamp <= row.timestamp)
                     & (df.source.isin([row.source, row.target]) | df.target.isin([row.source, row.target]))].tail(250)
        graph = nx.DiGraph()
        for tx in context.itertuples():
            if graph.has_edge(tx.source, tx.target):
                graph[tx.source][tx.target]["count"] += 1
            else:
                graph.add_edge(tx.source, tx.target, count=1)
        pos = nx.spring_layout(graph, seed=42)
        ex, ey = [], []
        for s, t in graph.edges:
            ex.extend([pos[s][0], pos[t][0], None])
            ey.extend([pos[s][1], pos[t][1], None])
        figure = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line={"width": 1, "color": "#94a3b8"},
                                      hoverinfo="skip", showlegend=False))
        nodes = list(graph.nodes)
        figure.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
                         text=[f"Account: {n}<br>Visible degree: {graph.degree(n)}" for n in nodes],
                         hovertemplate="%{text}<extra></extra>", marker={"size": 14,
                         "color": ["#f97316" if n in [row.source, row.target] else "#38bdf8" for n in nodes]},
                         showlegend=False))
        figure.update_layout(title="Account neighborhood · latest 250 incident transfers",
                             xaxis={"visible": False}, yaxis={"visible": False}, height=550)
        st.plotly_chart(figure, width="stretch")
        st.caption("Lines show connectivity; direction and transfer counts are in the table below. Orange nodes are the selected endpoints.")
        st.dataframe(pd.DataFrame([{"source": s, "target": t, "transfers": v["count"]}
                                   for s, t, v in graph.edges(data=True)]), hide_index=True)
with features:
    importance = pd.read_csv(run / "feature_importance.csv")
    st.plotly_chart(px.bar(importance.sort_values("ap_drop_mean"), x="ap_drop_mean", y="feature",
                           orientation="h", error_x="ap_drop_std", title="Validation permutation importance · graph-feature baseline"),
                    width="stretch")
    st.caption("Importance describes the baseline globally; it does not explain an individual GraphSAGE prediction.")
    curves = json.loads((run / "training_curves.json").read_text(encoding="utf-8"))
    curve_rows = [{"model": key, **epoch} for key, trace in curves.items() for epoch in trace]
    st.plotly_chart(px.line(pd.DataFrame(curve_rows), x="epoch", y="validation_ap", color="model"), width="stretch")
with methodology:
    st.markdown("""
    - **Target:** classify individual transactions, using account relationships as context.
    - **Split:** first 60% of distinct timestamps train, next 20% validate, final 20% test.
    - **Causality:** features use strictly earlier activity; graph messages travel past → present.
    - **Unknown labels:** retained as graph context, excluded from supervised loss and metrics.
    - **Selection:** baseline search and neural early stopping use validation average precision.
    - **Threshold:** selected on validation at the requested maximum FPR. Test FPR can differ.
    - **Unseen accounts:** at least one endpoint absent from training; also reported for both endpoints.
    - **Uncertainty:** row bootstrap intervals assume independent samples and can be optimistic for linked transactions.
    - **Scores:** ranking scores, not calibrated probabilities. Isolation Forest detects unusual behavior, not necessarily fraud.
    - **Scale:** bounded incoming edges, full-batch neural learning; benchmark memory before large datasets.
    """)
    st.json(manifest, expanded=False)
