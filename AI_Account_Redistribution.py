import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# Streamlit page config
st.set_page_config(
    page_title="AI-Powered Advisor Redistribution",
    page_icon="🤖",
    layout="wide"
)

# Load Excel from local file
@st.cache_data
def load_data(file_path):
    """
    Reads advisor account data from Excel file stored locally.
    Expected columns: Account ID, Advisor Name, Location, Specialty, Assets
    """
    return pd.read_excel(file_path)

# Train AI model
def train_model(df):
    # making a copy of the dataframe to avoid modifying the original
    df_train = df.copy()
    
    # initializing label encoders
    le_location = LabelEncoder()
    le_specialty = LabelEncoder()
    le_advisor = LabelEncoder()

    # encoding categorical variables
    df_train["Location_enc"] = le_location.fit_transform(df_train["Location"])
    df_train["Specialty_enc"] = le_specialty.fit_transform(df_train["Specialty"])
    df_train["Advisor_enc"] = le_advisor.fit_transform(df_train["Advisor Name"])

    # features and target variable
    X = df_train[["Location_enc", "Specialty_enc", "Assets"]]
    y = df_train["Advisor_enc"]

    # creates a classifier to train the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)

    return model, le_location, le_specialty, le_advisor

# AI Redistribution
def redistribute_accounts_ai(df, retiring_advisor, model, le_location, le_specialty, le_advisor):
    # list of advisors excluding the retiring one
    remaining_advisors = [a for a in df["Advisor Name"].unique() if a != retiring_advisor]
    
    # separating accounts of retiring advisor and others
    retiring_accounts = df[df["Advisor Name"] == retiring_advisor].copy()
    
    # data of remaining advisors
    remaining_data = df[df["Advisor Name"] != retiring_advisor].copy()

    recommendations = []

    # loops through each account of the retiring advisor and encodes features for prediction numerically
    for _, account in retiring_accounts.iterrows():
        loc_enc = le_location.transform([account["Location"]])[0]
        spec_enc = le_specialty.transform([account["Specialty"]])[0]
        
        # makes a set for the features for the model
        features = [[loc_enc, spec_enc, account["Assets"]]]

        # predicts the best fit advisor for the account/ converts back to original advisor name from encoded numerical value
        pred_enc = model.predict(features)[0]
        target = le_advisor.inverse_transform([pred_enc])[0]

        # if the predicted advisor is the retiring one, assign to the first remaining advisor
        if target == retiring_advisor:
            target = remaining_advisors[0]

        # reason for recommendation
        reason = f"Predicted by AI model as best fit: {target}."

        # appends the recommendation to the list
        recommendations.append({
            "Account ID": account["Account ID"],
            "Assets": account["Assets"],
            "Location": account["Location"],
            "Specialty": account["Specialty"],
            "New Advisor": target,
            "Reason": reason
        })

        # updates the remaining data to include the reassigned account
        remaining_data = pd.concat(
            [remaining_data, pd.DataFrame([account]).assign(**{"Advisor Name": target})]
        )

    return pd.DataFrame(recommendations)

# Streamlit UI
st.title("AI-Powered Advisor Redistribution Tool")
st.markdown("This tool uses an AI model to recommend how to reassign accounts when an advisor retires.")

# Load the data directly from local file
excel_path = "advisors.xlsx"
df = load_data(excel_path)

# Quick stats
col1, col2 = st.columns(2)
col1.metric("Total Advisors", df["Advisor Name"].nunique())
col2.metric("Total Accounts", len(df))

# Display the dataframe
with st.expander("View All Advisors & Accounts"):
    st.dataframe(df, use_container_width=True)

# Train AI model
model, le_location, le_specialty, le_advisor = train_model(df)

# Select retiring advisor
advisors = df["Advisor Name"].unique().tolist()
retiring_advisor = st.selectbox("Select Advisor to Retire", advisors)

# If button to generate recommendations is clicked
if st.button("Generate Recommendations"):
    recommendations_df = redistribute_accounts_ai(df, retiring_advisor, model, le_location, le_specialty, le_advisor)

    # Confirmation message
    st.success(f"✅ Redistribution complete for retiring advisor: {retiring_advisor}")

    # Display recommendations as a table
    with st.expander("📌 View Redistribution Plan"):
        st.dataframe(recommendations_df, use_container_width=True)

    # Display workload distribution before and after
    before_counts = df["Advisor Name"].value_counts()
    after_counts = pd.concat([
        df[df["Advisor Name"] != retiring_advisor]["Advisor Name"],
        recommendations_df["New Advisor"]
    ]).value_counts()
    workload_df = pd.DataFrame({"Before": before_counts, "After": after_counts}).fillna(0)

    st.subheader("Before & After Workload Distribution")
    st.bar_chart(workload_df)



# Terminal setup instructions

# python3 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip
# pip install streamlit pandas numpy scikit-learn xgboost openpyxl
# python3 -m streamlit run AI_Account_Redistribution.py