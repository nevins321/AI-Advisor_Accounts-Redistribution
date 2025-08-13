import pandas as pd
from datetime import datetime
import random

# Sample data
data = {
    "Account ID": list(range(1, 13)),
    "Advisor Name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
    "Location": ["NY", "NY", "LA", "LA", "SF", "SF", "NY", "LA", "SF", "NY", "LA", "SF"],
    "Specialty": ["Tech", "Finance", "Retail", "Tech", "Finance", "Retail", "Tech", "Finance", "Retail", "Tech", "Finance", "Retail"],
    "Assets": [100000, 150000, 120000, 130000, 90000, 110000, 95000, 140000, 125000, 135000, 98000, 112000],
    "Account Type": ["Individual", "Corporate", "Individual", "Corporate", "Individual", "Corporate", "Individual", "Corporate", "Individual", "Corporate", "Individual", "Corporate"],
    "Client Age": [34, 45, 29, 50, 38, 41, 36, 47, 31, 52, 39, 44],
    "Client Industry": ["Tech", "Finance", "Retail", "Tech", "Finance", "Retail", "Tech", "Finance", "Retail", "Tech", "Finance", "Retail"],
    "Risk Profile": ["Medium", "High", "Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium", "High", "Low"],
    "Tenure (years)": [5, 10, 2, 8, 6, 7, 4, 9, 3, 11, 5, 6],
    "Last Meeting Date": [datetime(2025, 1, random.randint(1, 28)) for _ in range(12)],
    "Portfolio Score": [random.randint(60, 100) for _ in range(12)]
}

df = pd.DataFrame(data)

# Save to Excel
df.to_excel("sample_advisor_data.xlsx", index=False)
print("Sample Excel file created: sample_advisor_data.xlsx")
