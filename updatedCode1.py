import pandas as pd
import streamlit as st

# --- Pseudo-database: hardcoded data ---
def load_data():
    data = {
        "Account ID": [1, 2, 3, 4, 5, 6],
        "Advisor Name": ["Alice", "Bob", "Alice", "Charlie", "Bob", "Diana"],
        "Location": ["NY", "NY", "LA", "LA", "NY", "SF"],
        "Specialty": ["Tech", "Finance", "Tech", "Retail", "Finance", "Tech"],
        "Assets": [100000, 150000, 120000, 130000, 90000, 110000],
    }
    return pd.DataFrame(data)

# --- AI-based Redistribution logic ---
def redistribute_accounts(df, retiring_advisor):
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()

    recommendations = []

    for _, account in retiring_accounts.iterrows():
        scores = {}

        for adv in remaining_advisors:
            adv_accounts = remaining_data[remaining_data["Advisor Name"] == adv]

            # Location similarity: 1 if matches, else 0
            location_score = 1 if adv_accounts["Location"].iloc[0] == account["Location"] else 0

            # Specialty similarity: 1 if matches, else 0
            specialty_score = 1 if adv_accounts["Specialty"].iloc[0] == account["Specialty"] else 0

            # Workload penalty: inverse of number of accounts + 1 (to avoid div by zero)
            workload = len(adv_accounts)
            workload_score = 1 / (workload + 1)

            # Weighted total score (tune weights if you want)
            total_score = 2 * location_score + 1.5 * specialty_score + 1 * workload_score

            scores[adv] = total_score

        # Pick advisor with highest total score
        target = max(scores, key=scores.get)
        reason = f"AI score {scores[target]:.2f}, assigned to {target}."

        recommendations.append({
            "Account ID": account["Account ID"],
            "Assets": account["Assets"],
            "Location": account["Location"],
            "Specialty": account["Specialty"],
            "New Advisor": target,
            "Reason": reason
        })

        # Add reassigned account to remaining_data to update workload
        remaining_data = pd.concat([remaining_data, pd.DataFrame([account]).assign(**{"Advisor Name": target})])

    return pd.DataFrame(recommendations)

# --- Streamlit UI ---

st.title("Advisor Retirement Redistribution Tool")

# Load pseudo database automatically (no file upload)
df = load_data()

advisors = df["Advisor Name"].unique().tolist()
retiring_advisor = st.selectbox("Select Advisor to Retire", advisors)

if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts(df, retiring_advisor)

    st.subheader("Redistribution Plan")
    st.dataframe(recommendations_df)

    st.subheader("Before & After Workload Distribution")

    before_counts = df["Advisor Name"].value_counts()

    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()

    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)

    st.bar_chart(workload_df)
