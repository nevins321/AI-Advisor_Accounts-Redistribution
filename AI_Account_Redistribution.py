import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import random
from io import BytesIO
from datetime import datetime
import os

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="AI-Powered Advisor Redistribution", page_icon="🤖", layout="wide")

FEEDBACK_FILE = "feedback.csv"
DATA_FILE = "advisors_extended.xlsx"
LOG_FILE = "redistribution_log.xlsx"
OVERRIDE_LOG = "override_log.csv"

# -------------------------------
# Helper Functions
# -------------------------------
def save_feedback(account_id, old_advisor, suggested_advisor, final_advisor, reasoning):
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

def log_override(account_id, old_advisor, new_advisor, reasoning):
    entry = f"{datetime.now()},{account_id},{old_advisor},{new_advisor},{reasoning}\n"
    with open(OVERRIDE_LOG, "a") as f:
        f.write(entry)

# -------------------------------
# Load/Save Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)

    # Drop duplicate Final Advisor column if it already exists
    if "Final Advisor" in df.columns:
        df = df.drop(columns=["Final Advisor"])

    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)

        if "Final Advisor" in feedback_df.columns and not feedback_df["Final Advisor"].isnull().all():
            df = df.merge(
                feedback_df[["Account ID", "Final Advisor"]],
                on="Account ID",
                how="left"
            )

            # Apply final advisor override if available
            df["Advisor Name"] = df.apply(
                lambda row: row["Final Advisor"]
                if pd.notna(row.get("Final Advisor"))
                else row["Advisor Name"],
                axis=1
            )

            # Drop helper column so it doesn’t stick around
            df = df.drop(columns=["Final Advisor"])
    return df


def save_data(df):
    df.to_excel(DATA_FILE, index=False, engine="openpyxl")

def log_redistribution(changes):
    log_entry = pd.DataFrame(changes)
    if os.path.exists(LOG_FILE):
        existing = pd.read_excel(LOG_FILE)
        updated_log = pd.concat([existing, log_entry], ignore_index=True)
    else:
        updated_log = log_entry
    updated_log.to_excel(LOG_FILE, index=False, engine="openpyxl")

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

def retrain_model_from_feedback():
    """Retrains the AI model using the updated feedback and overrides."""
    global df, model, encoders, le_target, feature_cols
    df = load_data()
    model, encoders, le_target, feature_cols = train_model(df)

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
    changes_log = []

    for _, account in retiring_accounts.iterrows():
        scores = {}
        explanations = {}
        summaries = {}

        acc_features = account.copy()
        for col, le in encoders.items():
            acc_features[col] = le.transform([str(account[col])])[0]

        X_acc = acc_features[feature_cols].to_numpy().reshape(1, -1)
        pred_probs = model.predict_proba(X_acc)[0]

        for adv in remaining_advisors:
            adv_data = df[df["Advisor Name"] == adv].iloc[0]
            score = 0
            reason_parts = []
            narrative_parts = []

            # Categorical matching
            for category in ["Location", "Specialty", "Languages Spoken",
                             "Licenses & Certifications", "Account Type", "Household Composition"]:
                if pd.notna(account.get(category)) and pd.notna(adv_data.get(category)):
                    if str(account[category]).strip().lower() == str(adv_data[category]).strip().lower():
                        score += weights.get(category, 0)
                        reason_parts.append(f"✅ {category} matched")
                        narrative_parts.append(f"same {category.lower()}")

            # Numerical similarity
            for num_col in ["Experience (Years)", "Performance Score", "Assets"]:
                if pd.notna(account.get(num_col)) and pd.notna(adv_data.get(num_col)):
                    diff = abs(float(account[num_col]) - float(adv_data[num_col]))
                    similarity = 1 / (1 + diff)
                    contribution = weights.get(num_col, 0) * similarity
                    score += contribution
                    reason_parts.append(f"{num_col} similarity +{contribution:.2f}")
                    narrative_parts.append(f"similar {num_col.lower()}")

            # Load penalty
            load_penalty = advisor_loads.get(adv, 0)
            penalty = balance_factor * load_penalty
            score -= penalty
            if penalty > 0:
                reason_parts.append(f"⚖️ Load penalty {penalty:.1f}")
                narrative_parts.append("balanced workload")

            # AI confidence
            adv_enc = le_target.transform([adv])[0]
            confidence = 100 * pred_probs[adv_enc]
            score += confidence
            reason_parts.append(f"🤖 AI confidence +{confidence:.1f}")
            narrative_parts.append("AI confidence")

            scores[adv] = max(0, score)
            explanations[adv] = "; ".join(reason_parts)
            summaries[adv] = f"Assigned to {adv} due to " + ", ".join(narrative_parts[:3]) + ("..." if len(narrative_parts) > 3 else "")

        # Normalize to percentage
        total_score = sum(scores.values())
        if total_score > 0:
            for adv in scores:
                scores[adv] = (scores[adv] / total_score) * 100

        max_score = max(scores.values())
        best_matches = [adv for adv, sc in scores.items() if sc == max_score]
        target = random.choice(best_matches)

        advisor_loads[target] = advisor_loads.get(target, 0) + 1
        advisor_assets[target] = advisor_assets.get(target, 0) + account.get("Assets", 0)

        # Apply assignment to main df
        df.loc[account.name, "Advisor Name"] = target

        recommendations.append({
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            "Assets": account.get("Assets", 0),
            "Match Score (%)": round(scores[target], 1),
            "Summary": summaries[target],
            "Detailed Explanation": explanations[target]
        })

        changes_log.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            "Reason": "AI Redistribution"
        })

    # Save updated assignments and log
    save_data(df)
    log_redistribution(changes_log)

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
              "Assets", "Account Type", "Household Composition"]:
    weights[param] = st.slider(param, 0.0, 1.0, 0.5, 0.1)

retiring_advisor = st.selectbox("Select Advisor to Retire", df["Advisor Name"].unique().tolist())

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# -------------------------------
# Generate AI Recommendations
# -------------------------------
if st.button("Generate Recommendations"):
    st.session_state.recommendations = redistribute_accounts_ai(
        df, retiring_advisor, model, encoders, le_target, feature_cols, weights
    )
    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")

# -------------------------------
# Display Recommendations + Feedback + Override
# -------------------------------
if st.session_state.recommendations is not None:
    recommendations_df = st.session_state.recommendations

    # Main table
    st.dataframe(recommendations_df.drop(columns=["Summary", "Detailed Explanation"]), use_container_width=True)

    # Detailed explanations & feedback
    with st.expander("See Explanations for Each Account"):
        for _, row in recommendations_df.iterrows():
            st.markdown(f"**Account {row['Account ID']} → {row['New Advisor']}**")
            st.write("📝 Summary:", row["Summary"])
            st.write("📊 Detailed:", row["Detailed Explanation"])

            # Feedback rating
            rating = st.radio(f"Rate recommendation for Account {row['Account ID']}", ["👍", "👎"], key=f"rating_{row['Account ID']}")
            if st.button(f"Submit Rating {row['Account ID']}", key=f"submit_{row['Account ID']}"):
                reasoning_text = f"User rated {rating}"
                save_feedback(
                    account_id=row["Account ID"],
                    old_advisor=row["Old Advisor"],
                    suggested_advisor=row["New Advisor"],
                    final_advisor=row["New Advisor"] if rating == "👍" else None,
                    reasoning=reasoning_text
                )
                st.success(f"✅ Feedback recorded for Account {row['Account ID']}")

                # Retrain model immediately using updated feedback
                retrain_model_from_feedback()

            st.divider()

    # Manual override
    st.subheader("🔄 Manual Override Recommendation")
    account_to_override = st.selectbox(
        "Select Account ID to Override",
        recommendations_df["Account ID"].tolist()
    )

    current_assignment = recommendations_df.loc[
        recommendations_df["Account ID"] == account_to_override, "New Advisor"
    ].values[0]
    st.write(f"Current assignment: **{current_assignment}**")

    new_advisor = st.selectbox(
        "Select New Advisor",
        [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    )
    reasoning = st.text_area("Reason for override")

    if st.button("Submit Override"):
        if len(reasoning) < 10:
            st.error("❌ Please provide a more detailed reasoning.")
        else:
            # Update recommendations and main dataframe
            st.session_state.recommendations.loc[
                st.session_state.recommendations["Account ID"] == account_to_override, "New Advisor"
            ] = new_advisor
            df.loc[df["Account ID"] == account_to_override, "Advisor Name"] = new_advisor

            # Log override
            log_override(account_to_override, current_assignment, new_advisor, reasoning)
            st.success(f"✅ Override complete: Account {account_to_override} reassigned to {new_advisor}")

            # Retrain model immediately using updated override
            retrain_model_from_feedback()

    # -------------------------------
    # Workload & Assets charts
    # -------------------------------
    before_counts = df["Advisor Name"].value_counts()
    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()
    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)
    st.subheader("Before & After Workload Distribution")
    st.bar_chart(workload_df)

    before_assets = df.groupby("Advisor Name")["Assets"].sum()
    after_assets = pd.concat([
        df[df["Advisor Name"] != retiring_advisor][["Advisor Name", "Assets"]],
        recommendations_df[["New Advisor", "Assets"]].rename(columns={"New Advisor": "Advisor Name"})
    ]).groupby("Advisor Name")["Assets"].sum()
    assets_df = pd.DataFrame({"Before Assets": before_assets, "After Assets": after_assets}).fillna(0)
    st.subheader("Before & After Asset Distribution")
    st.bar_chart(assets_df)

    # -------------------------------
    # Download Options
    # -------------------------------
    output = BytesIO()
    recommendations_df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Redistribution Plan (Excel)",
        data=output,
        file_name=f"redistribution_{retiring_advisor}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if os.path.exists(LOG_FILE):
        output_log = BytesIO()
        pd.read_excel(LOG_FILE).to_excel(output_log, index=False, engine="openpyxl")
        st.download_button(
            label="📥 Download Redistribution Log",
            data=output_log,
            file_name="redistribution_log.xlsx",
            mime="application/vnd.openxmlformats-spreadsheetml.sheet"
        )

    if os.path.exists(OVERRIDE_LOG):
        override_df = pd.read_csv(
            OVERRIDE_LOG,
            names=["Timestamp", "Account ID", "Old Advisor", "New Advisor", "Reason"]
        )
        output_override = BytesIO()
        override_df.to_excel(output_override, index=False, engine="openpyxl")
        st.download_button(
            label="📥 Download Override Log",
            data=output_override,
            file_name="override_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        #streamlit run AI_Account_Redistribution.py
        #