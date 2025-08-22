import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import random
from io import BytesIO
import os

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="AI-Powered Advisor Redistribution", page_icon="🤖", layout="wide")

FEEDBACK_FILE = "feedback.csv"

# -------------------------------
# Feedback Functions
# -------------------------------
def save_feedback(account_id, old_advisor, suggested_advisor, final_advisor, reasoning):
    """Save feedback (overrides or ratings) to CSV."""
    feedback_entry = pd.DataFrame([{
        "Account ID": account_id,
        "Old Advisor": old_advisor,
        "Suggested Advisor": suggested_advisor,
        "Final Advisor": final_advisor,
        "Reasoning": reasoning
    }])
    
    if os.path.exists(FEEDBACK_FILE):
        feedback_entry.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        feedback_entry.to_csv(FEEDBACK_FILE, index=False)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file_path="advisors_extended.xlsx"):
    df = pd.read_excel(file_path)

    # Merge feedback corrections if available
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        df = df.merge(feedback_df[["Account ID", "Final Advisor"]], on="Account ID", how="left")
        df["Advisor Name"] = df["Final Advisor"].fillna(df["Advisor Name"])
    return df

# -------------------------------
# Train Model
# -------------------------------
def train_model(df):
    df_train = df.copy()
    encoders = {}

    for col in ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                "Account Type", "Household Composition"]:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df_train["Advisor_enc"] = le_target.fit_transform(df_train["Advisor Name"])

    feature_cols = ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                    "Experience (Years)", "Performance Score", "Client Load (Capacity)",
                    "Assets", "Account Type", "Household Composition"]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(df_train[feature_cols], df_train["Advisor_enc"])

    return model, encoders, le_target, feature_cols

# -------------------------------
# Redistribution Logic
# -------------------------------
def redistribute_accounts_ai(df, retiring_advisor, model, encoders, le_target, feature_cols, weights, balance_factor=0.5):
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()
    advisor_loads = remaining_data["Advisor Name"].value_counts().to_dict()
    advisor_assets = remaining_data.groupby("Advisor Name")["Assets"].sum().to_dict()

    recommendations = []

    for _, account in retiring_accounts.iterrows():
        scores = {}
        explanations = {}
        summaries = {}

        total_weight = sum(weights.values())
        acc_features = account.copy()
        for col, le in encoders.items():
            acc_features[col] = le.transform([str(account[col])])[0]

        X_acc = [acc_features[feature_cols].values]
        pred_probs = model.predict_proba(X_acc)[0]

        for adv in remaining_advisors:
            adv_data = df[df["Advisor Name"] == adv].iloc[0]
            score = 0
            reason_parts = []
            narrative_parts = []

            for category in ["Location", "Specialty", "Languages Spoken",
                             "Licenses & Certifications", "Account Type", "Household Composition"]:
                if pd.notna(account.get(category)) and pd.notna(adv_data.get(category)):
                    if str(account[category]).strip().lower() == str(adv_data[category]).strip().lower():
                        score += weights[category]
                        reason_parts.append(f"✅ {category} matched")
                        narrative_parts.append(f"same {category.lower()}")

            for num_col in ["Experience (Years)", "Performance Score", "Account Size"]:
                if pd.notna(account.get(num_col)) and pd.notna(adv_data.get(num_col)):
                    diff = abs(float(account[num_col]) - float(adv_data[num_col]))
                    similarity = 1 / (1 + diff)
                    contribution = weights[num_col] * similarity
                    score += contribution
                    reason_parts.append(f"{num_col} similarity contributed {contribution:.2f}")
                    narrative_parts.append(f"similar {num_col.lower()}")

            load_penalty = advisor_loads.get(adv, 0)
            penalty = balance_factor * load_penalty
            score -= penalty
            if penalty > 0:
                reason_parts.append(f"⚖️ Load penalty {penalty:.1f}")
                narrative_parts.append("balanced workload")

            adv_enc = le_target.transform([adv])[0]
            confidence = 100 * pred_probs[adv_enc]
            score += confidence
            reason_parts.append(f"🤖 AI confidence +{confidence:.1f}")
            narrative_parts.append("AI confidence")

            scores[adv] = max(0, score)
            explanations[adv] = "; ".join(reason_parts)
            summaries[adv] = f"Assigned to {adv} due to " + ", ".join(narrative_parts[:3]) + ("..." if len(narrative_parts) > 3 else "")

        total_score = sum(scores.values())
        if total_score > 0:
            for adv in scores:
                scores[adv] = (scores[adv] / total_score) * 100

        max_score = max(scores.values())
        best_matches = [adv for adv, sc in scores.items() if sc == max_score]
        target = random.choice(best_matches)

        advisor_loads[target] = advisor_loads.get(target, 0) + 1
        advisor_assets[target] = advisor_assets.get(target, 0) + account.get("Assets", 0)

        recommendations.append({
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            "Assets": account.get("Assets", 0),
            "Match Score (%)": round(scores[target], 1),
            "Summary": summaries[target],
            "Detailed Explanation": explanations[target]
        })

    return pd.DataFrame(recommendations)

# -------------------------------
# Streamlit UI
# -------------------------------
df = load_data()
model, encoders, le_target, feature_cols = train_model(df)

st.title("AI-Powered Advisor Redistribution Tool")

st.metric("Total Advisors", df["Advisor Name"].nunique())
st.metric("Total Accounts", len(df))

with st.expander("View All Advisors & Accounts"):
    st.dataframe(df, use_container_width=True)

st.subheader("⚖️ Adjust Attribute Weights (0 to 1)")
weights = {}
for param in ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
              "Experience (Years)", "Performance Score", "Client Load (Capacity)",
              "Account Size", "Account Type", "Household Composition"]:
    weights[param] = st.slider(param, 0.0, 1.0, 0.5, 0.1)

retiring_advisor = st.selectbox("Select Advisor to Retire", df["Advisor Name"].unique().tolist())

# Store recommendations in session
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if st.button("Generate Recommendations"):
    st.session_state.recommendations = redistribute_accounts_ai(
        df, retiring_advisor, model, encoders, le_target, feature_cols, weights
    )
    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")

# -------------------------------
# Show recommendations + override + feedback
# -------------------------------
if st.session_state.recommendations is not None:
    recommendations_df = st.session_state.recommendations

    st.dataframe(recommendations_df.drop(columns=["Summary", "Detailed Explanation"]), use_container_width=True)

    with st.expander("See Explanations for Each Account"):
        for _, row in recommendations_df.iterrows():
            st.markdown(f"**Account {row['Account ID']} → {row['New Advisor']}**")
            st.write("📝 Summary:", row["Summary"])
            st.write("📊 Detailed:", row["Detailed Explanation"])

            # Feedback buttons
            rating = st.radio(f"Rate recommendation for Account {row['Account ID']}", ["👍", "👎"], key=f"rating_{row['Account ID']}")
            if st.button(f"Submit Rating {row['Account ID']}"):
                save_feedback(
                    account_id=row["Account ID"],
                    old_advisor=row["Old Advisor"],
                    suggested_advisor=row["New Advisor"],
                    final_advisor=row["New Advisor"] if rating == "👍" else None,
                    reasoning=f"User rated {rating}"
                )
                st.success(f"✅ Feedback recorded for Account {row['Account ID']}")

            st.divider()

    # Workload balance charts
    before_counts = df["Advisor Name"].value_counts()
    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()
    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)
    st.subheader("Before & After Workload Distribution")
    st.bar_chart(workload_df)

    # Assets balance
    before_assets = df.groupby("Advisor Name")["Assets"].sum()
    after_assets = pd.concat([
        df[df["Advisor Name"] != retiring_advisor][["Advisor Name", "Assets"]],
        recommendations_df[["New Advisor", "Assets"]].rename(columns={"New Advisor": "Advisor Name"})
    ]).groupby("Advisor Name")["Assets"].sum()
    assets_df = pd.DataFrame({"Before Assets": before_assets, "After Assets": after_assets}).fillna(0)
    st.subheader("Before & After Asset Distribution")
    st.bar_chart(assets_df)

    # Download option
    output = BytesIO()
    recommendations_df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Redistribution Plan (Excel)",
        data=output,
        file_name=f"redistribution_{retiring_advisor}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
