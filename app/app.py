import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Churn Retention Assistant", page_icon="📉", layout="wide")


@st.cache_data
def test_probabilities():
    return pl.model.predict_proba(pl.X_test)[:, 1].tolist()


@st.cache_data
def heldout_metrics():
    return pl.get_test_metrics()


# ---------- Sidebar: customer selection ----------
st.sidebar.title("Select customer")
only_churners = st.sidebar.checkbox("Show only customers who left", value=False)
probabilities = test_probabilities()
risk_filter = st.sidebar.selectbox("Filter by model risk estimate", ["All customers", "High", "Medium", "Low"])
eligible = list(range(len(pl.X_test)))
if only_churners:
    eligible = [i for i in eligible if int(pl.y_test.iloc[i]) == 1]
if risk_filter != "All customers":
    eligible = [i for i in eligible if pl.risk_label(probabilities[i]) == risk_filter]

if pl.X_test_customer_ids is not None:
    id_query = st.sidebar.text_input("Find a customer by ID", placeholder="e.g. 0001-ABC")
    if id_query.strip():
        query_lower = id_query.strip().lower()
        eligible = [i for i in eligible if query_lower in str(pl.X_test_customer_ids.iloc[i]).lower()]
    if not eligible:
        st.sidebar.warning("No test customers match these filters.")
        st.stop()
    customer = st.sidebar.selectbox(
        "Customer", eligible,
        format_func=lambda i: f"{pl.X_test_customer_ids.iloc[i]} · customer {i + 1}",
    )
else:
    st.sidebar.caption("Customer IDs are unavailable; showing customer numbers instead.")
    if not eligible:
        st.sidebar.warning("No test customers match these filters.")
        st.stop()
    customer = st.sidebar.selectbox("Customer number", eligible)
customer = int(customer)

actual = "Left" if int(pl.y_test.iloc[customer]) == 1 else "Stayed"
st.sidebar.markdown(f"**What happened:** {actual}")

# Reset chat/recommendation when the customer changes
if st.session_state.get("current_customer") != customer:
    st.session_state["current_customer"] = customer
    st.session_state["messages"] = []
    st.session_state.pop("rec_result", None)
    st.session_state.pop("rag_result", None)
    query = pl.build_query_from_shap(customer)
    st.session_state["rag_result"] = {"query": query, "policies": pl.retrieve_policy(query)}

st.title("📉 Customer Churn Prediction & Retention Assistant")
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Customer overview", "2. Why this risk score?", "3. Matching guidance", "4. Suggested actions & chat"
])

# ---------- Tab 1: Prediction ----------
with tab1:
    prob = pl.get_prediction(customer)
    label = pl.risk_label(prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated chance of leaving", f"{prob:.1%}", help="The model's estimated probability that this customer will leave.")
    c2.metric("Estimated risk", label, help="Low, medium, or high based on the model's probability bands.")
    c3.metric("Model prediction", "Likely to leave" if prob >= pl.best_threshold else "Likely to stay")
    st.caption(f"The model uses a {pl.best_threshold:.0%} chance as its cutoff for predicting that someone will leave.")
    st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Customer features")
    st.dataframe(pl.X_test.iloc[[customer]].T.rename(columns={customer: "Customer value"}), use_container_width=True)

    with st.expander("How well does the model do overall?"):
        st.caption(
            f"These scores compare model predictions with known outcomes for {len(pl.X_test):,} past customers "
            "that were kept separate from model training. They describe overall performance, not this one customer."
        )
        metrics = heldout_metrics()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Correct churn alerts", f"{metrics['precision']:.1%}", help="Of customers flagged as likely to leave, how many actually left.")
        m2.metric("Churners found", f"{metrics['recall']:.1%}", help="Of all customers who left, how many the model flagged.")
        m3.metric("Overall churn balance", f"{metrics['f1']:.1%}", help="A combined score balancing correct alerts and churners found.")
        m4.metric("Predictions correct", f"{metrics['accuracy']:.1%}", help="The share of all customer predictions that were correct.")
        st.markdown("**What happened vs. what the model predicted**")
        st.caption("Rows show what customers actually did; columns show the model's prediction.")
        st.dataframe(pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actually stayed", "Actually left"],
            columns=["Predicted would stay", "Predicted would leave"],
        ), use_container_width=True)

# ---------- Tab 2: SHAP explanation ----------
with tab2:
    st.subheader("Why did the model give this risk score?")
    st.caption(
        "Positive SHAP values push the model prediction toward churn; negative values push it toward staying. "
        "These are model influences, not proof of why a customer will leave."
    )

    table = pl.get_shap_table(customer)
    positive = table[table["shap_value"] > 0].nlargest(10, "shap_value")
    negative = table[table["shap_value"] < 0].nsmallest(10, "shap_value")
    top = pd.concat([positive.head(5), negative.head(5)]).sort_values("shap_value")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Push toward churn")
        if positive.empty:
            st.write("No features push this prediction toward churn.")
        else:
            st.dataframe(positive[["feature", "value", "shap_value"]].rename(
                columns={"feature": "Feature", "value": "Customer value", "shap_value": "Impact"}
            ), hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Push toward staying")
        if negative.empty:
            st.write("No features push this prediction toward staying.")
        else:
            st.dataframe(negative[["feature", "value", "shap_value"]].rename(
                columns={"feature": "Feature", "value": "Customer value", "shap_value": "Impact"}
            ), hide_index=True, use_container_width=True)

    if not top.empty:
        labels = [f"{f} = {v}" for f, v in zip(top["feature"], top["value"])]
        fig = go.Figure(go.Bar(
            x=top["shap_value"], y=labels, orientation="h",
            marker_color=["#f06b67" if v > 0 else "#52b76b" for v in top["shap_value"]],
            customdata=top["feature"],
            hovertemplate="%{customdata}<br>Impact: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=max(300, 36 * len(top)),
            margin=dict(l=12, r=16, t=12, b=42),
            xaxis_title="SHAP impact on churn prediction",
            yaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0"),
            xaxis=dict(zeroline=True, zerolinecolor="#9aa0aa", gridcolor="rgba(150,160,175,0.18)"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3: RAG retrieval, independently of Gemini ----------
with tab3:
    st.subheader("Guidance matched to this customer")
    st.caption("Search the policy index without calling Gemini.")
    if st.button("Refresh policy retrieval"):
        with st.spinner("Building customer query and searching the policy index..."):
            query = pl.build_query_from_shap(customer)
            st.session_state["rag_result"] = {"query": query, "policies": pl.retrieve_policy(query)}

    rag = st.session_state.get("rag_result")
    if rag:
        st.markdown("**Search query built from churn-driving factors**")
        st.info(rag["query"])
        if rag["policies"]:
            for policy in rag["policies"]:
                with st.expander(f"{policy['policy_id']} · {policy['category']} · score {policy['score']:.3f}", expanded=True):
                    st.write(policy["full_text"])
        else:
            st.warning("No policy passed the retrieval confidence threshold.")
    else:
        st.info("Policy retrieval is prepared automatically for the selected customer.")

# ---------- Tab 4: Recommendation + chat ----------
with tab4:
    st.subheader("Suggested next steps")
    rag = st.session_state.get("rag_result", {"policies": []})
    local = pl.get_local_recommendation(customer, rag.get("policies", []))
    st.markdown(f"**Assessment:** {local['summary']}")
    for i, action in enumerate(local["actions"]):
        source = local["action_sources"][i] if i < len(local["action_sources"]) else "General guidance"
        action_col, feedback_col = st.columns([5, 1])
        with action_col:
            st.markdown(f"- {action}")
            st.caption(f"Policy support: {source}")
        with feedback_col:
            st.checkbox("Helpful", key=f"helpful_{customer}_{i}")
    st.caption(local["coverage"])

    st.divider()
    st.subheader("AI recommendation and chat")
    st.caption("The suggestions above work without Gemini. Generate the AI version for a conversational answer.")

    if st.button("Generate recommendation", type="primary"):
        with st.spinner("Retrieving policy and asking the AI..."):
            st.session_state["rec_result"] = pl.generate_retention_recommendation(customer)

    rec = st.session_state.get("rec_result")
    if rec:
        resp = rec["response"]
        if resp.get("fallback_used"):
            st.warning("Gemini is unavailable right now. The local policy-based suggestions above are still available.")
            with st.expander("Gemini error details"):
                st.code(resp.get("llm_error", "No error details available."))

        st.markdown(f"**Summary:** {resp.get('summary', '')}")
        st.markdown("**Recommended actions:**")
        policy_sources = ", ".join(p["policy_id"] for p in rec["retrieved_policies"]) or "No confident policy match"
        for i, a in enumerate(resp.get("recommended_actions", [])):
            st.markdown(f"- {a}")
            st.caption(f"Grounded in retrieved policy context: {policy_sources}")
            st.checkbox("Helpful", key=f"ai_helpful_{customer}_{i}")
        st.markdown(f"**Policy coverage:** {resp.get('policy_coverage_note', '')}")
        st.caption(f"Confidence: {resp.get('confidence_note', '')}  |  Model: {rec['model_used'] or 'n/a'}")

        st.caption(f"RAG query: {rec['query_used']}")
        if rec["retrieved_policies"]:
            st.markdown("**Policies used for this recommendation**")
            for p in rec["retrieved_policies"]:
                with st.expander(f"{p['policy_id']} · {p['category']} · score {p['score']:.3f}"):
                    st.write(p["full_text"])
        else:
            st.warning("No policy passed the retrieval threshold; recommendation uses the no-match fallback.")

        st.divider()
        st.subheader("Ask about this customer")
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        question = st.chat_input("e.g. Why might this customer churn, and what should we offer?")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = pl.answer_followup(customer, question, rec)
                st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        st.info("Click **Generate recommendation** to get the AI suggestion and enable chat.")
