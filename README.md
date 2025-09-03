# Customer_Churn_Analysis #
 # importing libraries#
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load dataset (upload the dataset into pandas library and read the file)
df = pd.read_csv("customer_churn.csv")

# 2. Quick overview
print("Dataset Shape:", df.shape)
print(df.head())

# 3. Data Cleaning
df.dropna(inplace=True)  # remove missing values
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})  # convert target to numeric

# 4. Exploratory Data Analysis
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="Contract", y="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()

# 5. Feature Selection
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Convert TotalCharges to numeric
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
print("✅ Model Evaluation Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Save cleaned dataset for Power BI
df.to_csv("cleaned_churn.csv", index=False)
print("Cleaned dataset saved as 'cleaned_churn.csv'")
