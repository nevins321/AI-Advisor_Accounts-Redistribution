import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# --- Load Excel from local file ---
@st.cache_data
def load_data(file_path):
    """
    Reads advisor account data from Excel file stored locally.
    Expected columns: Account ID, Advisor Name, Location, Specialty, Assets
    """
    return pd.read_excel(file_path)

# --- Train AI model ---
def train_model(df):
    df_train = df.copy()
    le_location = LabelEncoder()
    le_specialty = LabelEncoder()
    le_advisor = LabelEncoder()

    df_train["Location_enc"] = le_location.fit_transform(df_train["Location"])
    df_train["Specialty_enc"] = le_specialty.fit_transform(df_train["Specialty"])
    df_train["Advisor_enc"] = le_advisor.fit_transform(df_train["Advisor Name"])

    X = df_train[["Location_enc", "Specialty_enc", "Assets"]]
    y = df_train["Advisor_enc"]

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    return model, le_location, le_specialty, le_advisor

# --- AI Redistribution ---
def redistribute_accounts_ai(df, retiring_advisor, model, le_location, le_specialty, le_advisor):
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()

    recommendations = []

    for _, account in retiring_accounts.iterrows():
        loc_enc = le_location.transform([account["Location"]])[0]
        spec_enc = le_specialty.transform([account["Specialty"]])[0]
        features = [[loc_enc, spec_enc, account["Assets"]]]

        pred_enc = model.predict(features)[0]
        target = le_advisor.inverse_transform([pred_enc])[0]

        if target == retiring_advisor:
            target = remaining_advisors[0]

        reason = f"Predicted by AI model as best fit: {target}."

        recommendations.append({
            "Account ID": account["Account ID"],
            "Assets": account["Assets"],
            "Location": account["Location"],
            "Specialty": account["Specialty"],
            "New Advisor": target,
            "Reason": reason
        })

        remaining_data = pd.concat(
            [remaining_data, pd.DataFrame([account]).assign(**{"Advisor Name": target})]
        )

    return pd.DataFrame(recommendations)

# --- Streamlit UI ---
st.title("AI-Powered Advisor Redistribution Tool")

# Load the data directly from a local file instead of file uploader
excel_path = "advisors.xlsx"  # <-- put your Excel file here
df = load_data(excel_path)

st.subheader("All Advisors & Accounts")
st.dataframe(df)

# Train AI model
model, le_location, le_specialty, le_advisor = train_model(df)

# Select retiring advisor
advisors = df["Advisor Name"].unique().tolist()
retiring_advisor = st.selectbox("Select Advisor to Retire", advisors)

if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts_ai(df, retiring_advisor, model, le_location, le_specialty, le_advisor)

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