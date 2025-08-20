import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import random
from io import BytesIO

# --- Load Excel file ---
@st.cache_data
def load_data(file_path="advisors_extended.xlsx"):
    # We start by loading all advisor and account data from Excel
    return pd.read_excel(file_path)

# --- Train XGBoost Model on historical data ---
def train_model(df):
    # Make a copy of the dataframe so we don’t overwrite the original
    df_train = df.copy()
    encoders = {}

    # Encode all categorical features into numeric values (needed for ML)
    for col in ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                "Account Type", "Household Composition"]:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le

    # Encode the target column (Advisor names → numeric labels)
    le_target = LabelEncoder()
    df_train["Advisor_enc"] = le_target.fit_transform(df_train["Advisor Name"])

    # Define the set of features we use for training
    feature_cols = ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                    "Experience (Years)", "Performance Score", "Client Load (Capacity)",
                    "Assets", "Account Type", "Household Composition"]

    # Train an XGBoost classification model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(df_train[feature_cols], df_train["Advisor_enc"])

    return model, encoders, le_target, feature_cols


# --- Redistribution Logic ---
def redistribute_accounts_ai(df, retiring_advisor, model, encoders, le_target, feature_cols, weights, balance_factor=0.5):
    # Get all advisors except the one retiring
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]

    # Get all accounts currently under the retiring advisor
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()

    # Data for remaining advisors: current loads and assets
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()
    advisor_loads = remaining_data["Advisor Name"].value_counts().to_dict()
    advisor_assets = remaining_data.groupby("Advisor Name")["Assets"].sum().to_dict()

    # Store final redistribution recommendations
    recommendations = []

    # --- Loop through each account of the retiring advisor ---
    for _, account in retiring_accounts.iterrows():
        scores = {}         # stores match scores for each advisor
        explanations = {}   # detailed reasoning for each advisor
        summaries = {}      # short narrative summary

        # Total weight is the sum of all sliders chosen by the user
        total_weight = sum(weights.values())

        # Encode the retiring account’s attributes so we can use them with the ML model
        acc_features = account.copy()
        for col, le in encoders.items():
            acc_features[col] = le.transform([str(account[col])])[0]

        # Convert into proper shape for prediction
        X_acc = [acc_features[feature_cols].values]

        # AI model predicts probability distribution over advisors
        pred_probs = model.predict_proba(X_acc)[0]

        # --- Now evaluate each possible advisor for this account ---
        for adv in remaining_advisors:
            adv_data = df[df["Advisor Name"] == adv].iloc[0]
            score = 0
            reason_parts = []    # detailed reasons for debugging
            narrative_parts = [] # short reasons for human explanation

            # --- 1. Categorical matches (like location, specialty, etc.) ---
            for category in ["Location", "Specialty", "Languages Spoken",
                             "Licenses & Certifications", "Account Type", "Household Composition"]:
                if pd.notna(account.get(category)) and pd.notna(adv_data.get(category)):
                    if str(account[category]).strip().lower() == str(adv_data[category]).strip().lower():
                        score += weights[category]  # add weight if it matches
                        reason_parts.append(f"✅ {category} matched")
                        narrative_parts.append(f"same {category.lower()}")

            # --- 2. Numerical similarity (experience, performance, account size) ---
            for num_col in ["Experience (Years)", "Performance Score", "Account Size"]:
                if pd.notna(account.get(num_col)) and pd.notna(adv_data.get(num_col)):
                    # Smaller difference = higher similarity
                    diff = abs(float(account[num_col]) - float(adv_data[num_col]))
                    similarity = 1 / (1 + diff)  # bounded between 0 and 1
                    contribution = weights[num_col] * similarity
                    score += contribution
                    reason_parts.append(f"{num_col} similarity contributed {contribution:.2f}")
                    narrative_parts.append(f"similar {num_col.lower()}")

            # --- 3. Apply load penalty (to balance workload) ---
            load_penalty = advisor_loads.get(adv, 0)
            penalty = balance_factor * load_penalty
            score -= penalty
            if penalty > 0:
                reason_parts.append(f"⚖️ Load penalty of {penalty:.1f} applied")
                narrative_parts.append("balanced workload")

            # --- 4. Add AI model confidence ---
            adv_enc = le_target.transform([adv])[0]
            confidence = 100 * pred_probs[adv_enc]
            score += confidence
            reason_parts.append(f"🤖 AI model confidence +{confidence:.1f}")
            narrative_parts.append("high AI confidence")

            # Ensure score is not negative
            scores[adv] = max(0, score)
            explanations[adv] = "; ".join(reason_parts)
            summaries[adv] = f"Assigned to {adv} due to " + ", ".join(narrative_parts[:3]) + ("..." if len(narrative_parts) > 3 else "")

        # --- Normalize all scores so they sum to 100 (probability style) ---
        total_score = sum(scores.values())
        if total_score > 0:
            for adv in scores:
                scores[adv] = (scores[adv] / total_score) * 100
        else:
            for adv in scores:
                scores[adv] = 0

        # Pick the advisor with the highest score (break ties randomly)
        max_score = max(scores.values())
        best_matches = [adv for adv, sc in scores.items() if sc == max_score]
        target = random.choice(best_matches)

        # Update load and assets for chosen advisor (so future accounts are balanced)
        advisor_loads[target] = advisor_loads.get(target, 0) + 1
        advisor_assets[target] = advisor_assets.get(target, 0) + account.get("Assets", 0)

        # Save recommendation
        recommendations.append({
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            "Assets": account.get("Assets", 0),
            "Match Score (%)": round(scores[target], 1),   # normalized score of chosen advisor
            "Summary": summaries[target],
            "Detailed Explanation": explanations[target]
        })

    # Return full recommendation dataframe
    return pd.DataFrame(recommendations)


# --- Streamlit User Interface ---
st.set_page_config(page_title="AI-Powered Advisor Redistribution", page_icon="🤖", layout="wide")
st.title("AI-Powered Advisor Redistribution Tool")

# Load the dataset
df = load_data()

# Show quick stats
st.metric("Total Advisors", df["Advisor Name"].nunique())
st.metric("Total Accounts", len(df))

# Allow user to expand and see the full dataset
with st.expander("View All Advisors & Accounts"):
    st.dataframe(df, use_container_width=True)

# Let the user choose custom weights for each attribute
st.subheader("⚖️ Adjust Attribute Weights (0 to 1)")
weights = {}
for param in ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
              "Experience (Years)", "Performance Score", "Client Load (Capacity)",
              "Account Size", "Account Type", "Household Composition"]:
    # Slider between 0 and 1, default = 0.5
    weights[param] = st.slider(param, 0.0, 1.0, 0.5, 0.1)

# Train the ML model on the dataset
model, encoders, le_target, feature_cols = train_model(df)

# Pick which advisor is retiring
retiring_advisor = st.selectbox("Select Advisor to Retire", df["Advisor Name"].unique().tolist())

# When user clicks button, run redistribution
if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts_ai(df, retiring_advisor, model, encoders, le_target, feature_cols, weights)

    # Show summary table
    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")
    st.dataframe(recommendations_df.drop(columns=["Summary", "Detailed Explanation"]), use_container_width=True)

    # Show detailed reasoning for each recommendation
    with st.expander("See Explanations for Each Account"):
        for _, row in recommendations_df.iterrows():
            st.markdown(f"**Account {row['Account ID']} → {row['New Advisor']}**")
            st.write("📝 Summary:", row["Summary"])
            st.write("📊 Detailed:", row["Detailed Explanation"])
            st.divider()

    # Compare before & after workloads
    before_counts = df["Advisor Name"].value_counts()
    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()
    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)

    st.subheader("Before & After Workload Distribution")
    st.bar_chart(workload_df)

    # Compare before & after assets
    before_assets = df.groupby("Advisor Name")["Assets"].sum()
    after_assets = pd.concat([
        df[df["Advisor Name"] != retiring_advisor][["Advisor Name", "Assets"]],
        recommendations_df[["New Advisor", "Assets"]].rename(columns={"New Advisor": "Advisor Name"})
    ]).groupby("Advisor Name")["Assets"].sum()
    assets_df = pd.DataFrame({"Before Assets": before_assets, "After Assets": after_assets}).fillna(0)

    st.subheader("Before & After Asset Distribution")
    st.bar_chart(assets_df)

    # Allow download of results as Excel
    output = BytesIO()
    recommendations_df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Redistribution Plan (Excel)",
        data=output,
        file_name=f"redistribution_{retiring_advisor}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
