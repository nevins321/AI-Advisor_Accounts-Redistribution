import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import random
from io import BytesIO

# --- Load Excel automatically from local file ---
@st.cache_data
def load_data(file_path="advisors_extended.xlsx"):
    return pd.read_excel(file_path)

# --- Fixed Weights for 10 Attributes ---
FIXED_WEIGHTS = {
    "Location": 15,
    "Specialty": 15,
    "Languages Spoken": 20,
    "Licenses & Certifications": 10,
    "Experience (Years)": 10,
    "Performance Score": 10,
    "Client Load (Capacity)": 5,
    "Account Size": 5,
    "Account Type": 5,
    "Household Composition": 5,
}

# --- Train XGBoost Model ---
def train_model(df):
    df_train = df.copy()
    encoders = {}

    # Encode categorical columns
    for col in ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                "Account Type", "Household Composition"]:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        encoders[col] = le

    # Encode target (Advisor)
    le_target = LabelEncoder()
    df_train["Advisor_enc"] = le_target.fit_transform(df_train["Advisor Name"])

    feature_cols = ["Location", "Specialty", "Languages Spoken", "Licenses & Certifications",
                    "Experience (Years)", "Performance Score", "Client Load (Capacity)",
                    "Assets", "Account Type", "Household Composition"]

    X = df_train[feature_cols]
    y = df_train["Advisor_enc"]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    return model, encoders, le_target, feature_cols

# --- Redistribution logic ---
def redistribute_accounts_ai(df, retiring_advisor, model, encoders, le_target, feature_cols, balance_factor=0.5):
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()

    # Precompute advisor stats
    advisor_loads = remaining_data["Advisor Name"].value_counts().to_dict()
    advisor_assets = remaining_data.groupby("Advisor Name")["Assets"].sum().to_dict()

    recommendations = []

    for _, account in retiring_accounts.iterrows():
        scores = {}
        total_weight = sum(FIXED_WEIGHTS.values())

        # Prepare account features for AI model
        acc_features = account.copy()
        for col, le in encoders.items():
            acc_features[col] = le.transform([str(account[col])])[0]
        X_acc = [acc_features[feature_cols].values]

        pred_probs = model.predict_proba(X_acc)[0]

        for adv in remaining_advisors:
            adv_data = df[df["Advisor Name"] == adv].iloc[0]
            score = 0

            # --- Weighted categorical matching ---
            for category in ["Location", "Specialty", "Languages Spoken",
                             "Licenses & Certifications", "Account Type", "Household Composition"]:
                if pd.notna(account.get(category)) and pd.notna(adv_data.get(category)):
                    if str(account[category]).strip().lower() == str(adv_data[category]).strip().lower():
                        score += FIXED_WEIGHTS[category]

            # --- Weighted numerical similarity ---
            for num_col in ["Experience (Years)", "Performance Score", "Account Size"]:
                if pd.notna(account.get(num_col)) and pd.notna(adv_data.get(num_col)):
                    diff = abs(float(account[num_col]) - float(adv_data[num_col]))
                    similarity = 1 / (1 + diff)
                    score += FIXED_WEIGHTS[num_col] * similarity

            # --- Capacity balancing ---
            load_penalty = advisor_loads.get(adv, 0)
            score -= balance_factor * load_penalty

            # --- AI model confidence ---
            adv_enc = le_target.transform([adv])[0]
            score += 100 * pred_probs[adv_enc]

            scores[adv] = score / (total_weight + 100) * 100

        # Pick best advisor
        max_score = max(scores.values())
        best_matches = [adv for adv, sc in scores.items() if sc == max_score]
        target = random.choice(best_matches)

        # Update stats
        advisor_loads[target] = advisor_loads.get(target, 0) + 1
        advisor_assets[target] = advisor_assets.get(target, 0) + account.get("Assets", 0)

        recommendations.append({
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            "Assets": account.get("Assets", 0),
            "Match Score (%)": round(max_score, 1)
        })

    return pd.DataFrame(recommendations)

# --- Streamlit UI ---
st.set_page_config(page_title="AI-Powered Advisor Redistribution", page_icon="🤖", layout="wide")
st.title("AI-Powered Advisor Redistribution Tool")

df = load_data()

st.metric("Total Advisors", df["Advisor Name"].nunique())
st.metric("Total Accounts", len(df))

with st.expander("View All Advisors & Accounts"):
    st.dataframe(df, use_container_width=True)

# Train model
model, encoders, le_target, feature_cols = train_model(df)

retiring_advisor = st.selectbox("Select Advisor to Retire", df["Advisor Name"].unique().tolist())

if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts_ai(df, retiring_advisor, model, encoders, le_target, feature_cols)

    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")
    st.dataframe(recommendations_df, use_container_width=True)

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

    # Export Excel
    output = BytesIO()
    recommendations_df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Redistribution Plan (Excel)",
        data=output,
        file_name=f"redistribution_{retiring_advisor}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
