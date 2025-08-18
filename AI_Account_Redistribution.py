import pandas as pd
import streamlit as st
import random
from io import BytesIO

# --- Load Excel automatically from local file ---
@st.cache_data
def load_data(file_path="advisors.xlsx"):
    """
    Reads advisor account data from local Excel file automatically.
    """
    return pd.read_excel(file_path)

# --- Redistribution logic with weighted categories & balancing ---
def redistribute_accounts_ai(df, retiring_advisor, weights, balance_factor=0.5):
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()

    # Precompute advisor stats for balancing
    advisor_loads = remaining_data["Advisor Name"].value_counts().to_dict()
    advisor_assets = remaining_data.groupby("Advisor Name")["Assets"].sum().to_dict()

    recommendations = []

    for _, account in retiring_accounts.iterrows():
        scores = {}
        total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1

        for adv in remaining_advisors:
            adv_data = df[df["Advisor Name"] == adv].iloc[0]
            score = 0

            # --- Categorical Matching ---
            for category, weight in weights.items():
                if category in df.columns and pd.notna(account.get(category)) and pd.notna(adv_data.get(category)):
                    if isinstance(account[category], str) and account[category] == adv_data[category]:
                        score += weight

            # --- Numerical Matching --- (Experience, Performance, Assets)
            for num_col in ["Experience (Years)", "Performance Score", "Assets"]:
                if num_col in df.columns and pd.notna(account.get(num_col)) and pd.notna(adv_data.get(num_col)):
                    diff = abs(float(account[num_col]) - float(adv_data[num_col]))
                    similarity = 1 / (1 + diff)  # closer values → higher score
                    score += weights.get(num_col, 0) * similarity

            # --- Capacity Balancing ---
            load_penalty = advisor_loads.get(adv, 0)
            score -= balance_factor * load_penalty  # discourage overloading heavy advisors

            scores[adv] = score / total_weight * 100  # normalize to %

        # Pick advisor with highest score
        max_score = max(scores.values())
        best_matches = [adv for adv, sc in scores.items() if sc == max_score]
        target = random.choice(best_matches)

        # Update loads
        advisor_loads[target] = advisor_loads.get(target, 0) + 1
        advisor_assets[target] = advisor_assets.get(target, 0) + account.get("Assets", 0)

        recommendations.append({
            "Account ID": account["Account ID"],
            "Old Advisor": retiring_advisor,
            "New Advisor": target,
            **{cat: account[cat] for cat in weights.keys() if cat in account},
            "Assets": account.get("Assets", 0),
            "Match Score (%)": round(max_score, 1)
        })

        remaining_data = pd.concat(
            [remaining_data, pd.DataFrame([account]).assign(**{"Advisor Name": target})]
        )

    return pd.DataFrame(recommendations)

# --- Streamlit UI ---
st.set_page_config(page_title="AI-Powered Advisor Redistribution", page_icon="🤖", layout="wide")

st.title("AI-Powered Advisor Redistribution Tool")
st.markdown(
    "This tool automatically loads `advisors.xlsx` from the local folder and "
    "uses weighted categories + balancing to recommend account reassignment when an advisor retires."
)

# Load Excel automatically
try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'advisors.xlsx' not found in the local folder.")
    st.stop()

# Quick stats
col1, col2 = st.columns(2)
col1.metric("Total Advisors", df["Advisor Name"].nunique())
col2.metric("Total Accounts", len(df))

with st.expander("View All Advisors & Accounts"):
    st.dataframe(df, use_container_width=True)

# Select retiring advisor
advisors = df["Advisor Name"].unique().tolist()
retiring_advisor = st.selectbox("Select Advisor to Retire", advisors)

# Automatically detect categorical columns (exclude advisor name)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in ["Advisor Name"]]

# Add numerical attributes explicitly
numerical_cols = [col for col in ["Experience (Years)", "Performance Score", "Assets"] if col in df.columns]

# Set category weights dynamically
st.subheader("Adjust Category Weights")
weights = {}
for cat in categorical_cols + numerical_cols:
    weights[cat] = st.slider(f"{cat} Weight", 0.0, 1.0, 0.5, 0.05)

# Generate recommendations
if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts_ai(df, retiring_advisor, weights)

    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")

    with st.expander("📌 View Redistribution Plan"):
        st.dataframe(recommendations_df, use_container_width=True)

    # Workload distribution before and after
    before_counts = df["Advisor Name"].value_counts()
    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()
    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)

    st.subheader("Before & After Workload Distribution")
    st.bar_chart(workload_df)

    # Assets distribution before and after
    before_assets = df.groupby("Advisor Name")["Assets"].sum()
    after_assets = pd.concat([
        df[df["Advisor Name"] != retiring_advisor][["Advisor Name", "Assets"]],
        recommendations_df[["New Advisor", "Assets"]].rename(columns={"New Advisor": "Advisor Name"})
    ]).groupby("Advisor Name")["Assets"].sum()
    assets_df = pd.DataFrame({"Before Assets": before_assets, "After Assets": after_assets}).fillna(0)

    st.subheader("Before & After Asset Distribution")
    st.bar_chart(assets_df)

    # Pie chart for account distribution
    st.subheader("After Redistribution - Account Share")
    st.pyplot(workload_df["After"].plot.pie(autopct='%1.1f%%', figsize=(6,6)).get_figure())

    # Export recommendations
    output = BytesIO()
    recommendations_df.to_excel(output, index=False, engine="openpyxl")
    st.download_button(
        label="📥 Download Redistribution Plan (Excel)",
        data=output,
        file_name=f"redistribution_{retiring_advisor}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
