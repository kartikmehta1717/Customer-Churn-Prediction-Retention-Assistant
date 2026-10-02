"""
pipeline.py - Modules 2-5 logic (model, SHAP, RAG, LLM) as reusable functions.
Imported by app.py. Artifacts are loaded once when this module is first imported.
"""
import os
import re
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

# ---------- Load artifacts ----------
model = joblib.load(ROOT / "models" / "churn_model.pkl")
shap_values = joblib.load(ROOT / "models" / "shap_values.pkl")
X_test = pd.read_csv(ROOT / "data" / "processed" / "X_test.csv")
y_test = pd.read_csv(ROOT / "data" / "processed" / "y_test.csv").squeeze()

# Rebuild the test-row to customer-ID mapping using the same split as training.
# IDs are display-only and are never passed into the churn model.
X_test_customer_ids = None
try:
    source = pd.read_csv(ROOT / "data" / "raw" / "telco.csv")
    source_target = source["Churn Label"].map({"Yes": 1, "No": 0})
    _, test_rows = train_test_split(
        np.arange(len(source)), test_size=0.2, random_state=42, stratify=source_target
    )
    candidate_ids = source.iloc[test_rows]["Customer ID"].reset_index(drop=True)
    candidate_target = source_target.iloc[test_rows].reset_index(drop=True)
    if len(candidate_ids) == len(X_test) and np.array_equal(candidate_target.values, y_test.values):
        X_test_customer_ids = candidate_ids
except (FileNotFoundError, KeyError, ValueError):
    pass

index = faiss.read_index(str(ROOT / "models" / "policy_index.faiss"))
with open(ROOT / "models" / "policies_metadata.pkl", "rb") as f:
    policies = pickle.load(f)
min_score = joblib.load(ROOT / "models" / "rag_min_score.pkl")
best_threshold = float(joblib.load(ROOT / "models" / "best_threshold.pkl"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Gemini ----------
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


def call_gemini_with_retry(prompt, models=None, max_retries=3, base_wait=10, json_mode=False):
    """Retry with backoff + model fallback. Returns (text, model_used)."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    models = models or GEMINI_MODELS
    last_error = None
    for model_name in models:
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json" if json_mode else None,
                )
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                return response.text, model_name
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(base_wait * (attempt + 1))
    raise last_error


# ---------- Prediction ----------
def get_prediction(customer_index):
    features = X_test.iloc[[customer_index]]
    return float(model.predict_proba(features)[0][1])


def risk_label(prob):
    if prob >= 0.6:
        return "High"
    if prob >= 0.4:
        return "Medium"
    return "Low"


def get_test_metrics():
    """Metrics on the held-out test set at the app's configured threshold."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def get_local_recommendation(customer_index, retrieved_policies=None):
    """Provide a useful policy-grounded suggestion without requiring Gemini."""
    probability = get_prediction(customer_index)
    retrieved_policies = retrieve_policy(build_query_from_shap(customer_index)) if retrieved_policies is None else retrieved_policies
    if probability < 0.4:
        return {
            "summary": f"The model estimates {probability:.1%} churn risk ({risk_label(probability).lower()} risk). A light-touch approach is appropriate.",
            "actions": [
                "Monitor the account and avoid an unnecessary discount or aggressive retention offer.",
                "If contacting the customer, use a brief check-in to ask whether anything about the service needs attention.",
            ],
            "action_sources": ["Risk-based guidance"] * 2,
            "coverage": "Light-touch guidance based on the model's low estimated risk.",
        }

    policy_ids = [p["policy_id"] for p in retrieved_policies]
    if not policy_ids:
        return {
            "summary": f"The model estimates {probability:.1%} churn risk ({risk_label(probability).lower()} risk), but no policy matched confidently.",
            "actions": [
                "Review the customer's strongest churn-driving factors with them before proposing a change.",
                "Ask an open question about the main service concern, then route it to the relevant support team.",
            ],
            "action_sources": ["No confident policy match"] * 2,
            "coverage": "No policy passed the retrieval threshold; these are general discovery steps, not a specific offer.",
        }

    suggestions = {
        "P00_FALLBACK": ["Use a low-pressure check-in to understand the customer's situation; do not make a retention offer when the reason is unknown or non-actionable."],
        "P02": ["Arrange a follow-up from a senior support representative and acknowledge the customer's experience.", "Review the relevant interaction and confirm what will change to prevent a repeat issue."],
        "P03": ["Review the bill and usage with the customer, then check whether a lower-cost plan or more predictable billing option fits their needs."],
        "P04": ["Check for service interruptions affecting the customer and give them a clear diagnostic or repair follow-up."],
        "P05": ["Route the unresolved technical issue to an experienced support specialist and confirm the fix with the customer."],
        "P06": ["Compare the customer's speed needs with their current service and investigate any performance issue before discussing a plan change."],
        "P07": ["Offer a short guided walkthrough of the relevant self-service tools and make sure the customer knows how to reach a person if they get stuck."],
        "P01A": ["Ask which device feature matters to the customer, then compare it with available options before suggesting a change."],
        "P01B": ["Clarify the competitor's offer and compare its terms with the customer's current plan before discussing a retention option."],
        "P01C": ["Clarify the data allowance the customer needs and compare available plans using their actual usage."],
        "P01D": ["Clarify the speed issue and compare the customer's current connection with available service options."],
    }
    actions = []
    action_sources = []
    for policy_id in policy_ids:
        for action in suggestions.get(policy_id, []):
            if action not in actions:
                actions.append(action)
                action_sources.append(policy_id)
    return {
        "summary": f"The model estimates {probability:.1%} churn risk ({risk_label(probability).lower()} risk). These suggestions follow the retrieved policy guidance.",
        "actions": list(dict.fromkeys(actions))[:4],
        "action_sources": action_sources[:4],
        "coverage": "Based on: " + ", ".join(policy_ids),
    }


# ---------- SHAP ----------
def get_shap_table(customer_index):
    """All features with value + SHAP contribution for one customer."""
    return pd.DataFrame({
        "feature": X_test.columns,
        "value": X_test.iloc[customer_index].values,
        "shap_value": shap_values[customer_index],
    })


def get_top_shap_reasons(customer_index, top_n=5):
    return get_shap_table(customer_index).sort_values("shap_value", ascending=False).head(top_n)


def build_query_from_shap(customer_index, top_n=6):
    contributions = get_shap_table(customer_index)
    top_factors = contributions[contributions["shap_value"] > 0].sort_values(
        "shap_value", ascending=False
    ).head(top_n)

    query_parts = []
    contract_flags = {}

    for _, row in top_factors.iterrows():
        feature_name = row["feature"]
        value = row["value"]

        if feature_name.startswith("Contract_"):
            category = feature_name.split("_", 1)[1]
            contract_flags[category] = value
            continue

        if "_" in feature_name:
            base_feature, category = feature_name.rsplit("_", 1)
            if value == True:  # noqa: E712
                query_parts.append(f"{base_feature} is {category}")
            else:
                query_parts.append(f"{base_feature} is not {category}")
        else:
            query_parts.append(f"{feature_name} is {value}")

    if contract_flags:
        if not contract_flags.get("One Year", False) and not contract_flags.get("Two Year", False):
            query_parts.insert(0, "Contract is Month-to-Month")
        elif contract_flags.get("One Year", False):
            query_parts.insert(0, "Contract is One Year")
        elif contract_flags.get("Two Year", False):
            query_parts.insert(0, "Contract is Two Year")

    return "Customer profile: " + ", ".join(query_parts)


# ---------- RAG ----------
def retrieve_policy(query, top_k=3, threshold=None):
    threshold = min_score if threshold is None else threshold
    query_embedding = embed_model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score >= threshold:
            results.append({
                "policy_id": policies[idx]["policy_id"],
                "category": policies[idx]["category"],
                "score": float(score),
                "full_text": policies[idx]["full_text"],
            })
    return results


# ---------- LLM prompt ----------
def build_llm_prompt(customer_index, retrieved_policies, churn_probability):
    shap_reasons = get_top_shap_reasons(customer_index, top_n=5)
    shap_context = "\n".join(
        f"- {row['feature']}: {row['value']} (impact score: {row['shap_value']:.3f})"
        for _, row in shap_reasons.iterrows()
    )

    if len(retrieved_policies) == 0:
        policy_context = "NO RELEVANT POLICY FOUND (score below confidence threshold)."
        has_match = False
    else:
        policy_context = "\n\n".join(
            f"[{p['policy_id']}] {p['category']} (relevance score: {p['score']:.3f})\n{p['full_text']}"
            for p in retrieved_policies
        )
        has_match = True

    instruction = f"""Using ONLY the retrieved policy information above (if any), analyze this customer and respond in STRICT JSON format with exactly these fields:

{{
  "has_confident_match": {str(has_match).lower()},
  "summary": "One or two sentence plain-language summary. Your risk language (e.g. 'high risk', 'moderate risk', 'low risk') MUST match the actual churn probability given above ({churn_probability:.1%}) - do not call a customer high-risk if the probability is low, even if the listed factors individually look concerning.",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "policy_coverage_note": "A short honest note on whether the retrieved policy fully addresses this customer's risk factors, or only partially. If no policy matched, explain that a general check-in is recommended instead.",
  "confidence_note": "Brief note on how confident this recommendation is, based on the retrieval score"
}}

Rules:
- Do not invent actions, discounts, or numbers not present in the retrieved policy text.
- If no policy matched, recommended_actions should contain only general, low-risk steps (e.g., a check-in call), not fabricated offers.
- If the overall churn probability is low (below 40%), recommended_actions can be lighter-touch (e.g. monitoring, a soft check-in) rather than aggressive retention offers.
- Return ONLY valid JSON, no other text before or after it.
"""

    return f"""You are a customer retention assistant. A machine learning model has predicted this customer's churn probability as {churn_probability:.1%}.

CUSTOMER'S TOP CHURN-RISK FACTORS (from SHAP, i.e. factors that push toward churn, regardless of the final probability):
{shap_context}

RETRIEVED RETENTION POLICY:
{policy_context}

INSTRUCTIONS: {instruction}
"""


# ---------- Full pipeline ----------
def generate_retention_recommendation(customer_index, max_retries=3):
    churn_probability = get_prediction(customer_index)
    query = build_query_from_shap(customer_index)
    retrieved = retrieve_policy(query)
    prompt = build_llm_prompt(customer_index, retrieved, churn_probability)

    model_used, raw_text = None, None
    try:
        raw_text, model_used = call_gemini_with_retry(prompt, max_retries=max_retries, json_mode=True)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {
            "has_confident_match": len(retrieved) > 0,
            "summary": "Could not parse structured response.",
            "recommended_actions": [],
            "policy_coverage_note": "N/A",
            "confidence_note": "N/A",
            "raw_text": raw_text,
        }
    except Exception as e:
        local = get_local_recommendation(customer_index, retrieved)
        parsed = {
            "has_confident_match": len(retrieved) > 0,
            "summary": local["summary"],
            "recommended_actions": local["actions"],
            "policy_coverage_note": local["coverage"],
            "confidence_note": "Local policy-based suggestions shown; Gemini could not be reached.",
            "llm_error": str(e)[:300],
            "fallback_used": True,
        }

    return {
        "customer_index": int(customer_index),
        "churn_probability": churn_probability,
        "query_used": query,
        "retrieved_policies": retrieved,
        "model_used": model_used,
        "response": parsed,
    }


def answer_followup(customer_index, question, rec_result):
    """Chat: answer a question about this customer, grounded in SHAP + retrieved policy."""
    shap_reasons = get_top_shap_reasons(customer_index, top_n=5)
    shap_context = "\n".join(
        f"- {r['feature']}: {r['value']} (impact {r['shap_value']:.3f})"
        for _, r in shap_reasons.iterrows()
    )
    policy_context = "\n\n".join(
        f"[{p['policy_id']}] {p['full_text']}" for p in rec_result["retrieved_policies"]
    ) or "NO RELEVANT POLICY FOUND."

    prompt = f"""You are a customer retention assistant. Answer the question briefly (max 5 lines).
Use ONLY the information below. Do not invent offers, discounts or numbers. If the answer is not in the information, say so.

Predicted churn probability: {rec_result['churn_probability']:.1%}
Top churn-risk factors:
{shap_context}

Retrieved policy:
{policy_context}

Question: {question}
"""
    try:
        text, _ = call_gemini_with_retry(prompt, max_retries=2)
        return text
    except Exception as e:
        return f"AI service is unavailable right now ({str(e)[:80]}). Please try again in a minute."
