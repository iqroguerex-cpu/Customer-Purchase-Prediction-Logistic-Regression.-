import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Page config
st.set_page_config(page_title="Purchase Prediction App", layout="wide")

# Title
st.title("🛍️ Customer Purchase Prediction")
st.markdown("Predict whether a customer will purchase based on **Age** and **Salary**")

# Load dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# Show dataset
with st.expander("📂 View Dataset"):
    st.dataframe(dataset)

# Features & target
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Sidebar input
st.sidebar.header("🔧 Input Customer Details")

age = st.sidebar.slider("Age", int(dataset.Age.min()), int(dataset.Age.max()), 30)
salary = st.sidebar.slider(
    "Estimated Salary",
    int(dataset.EstimatedSalary.min()),
    int(dataset.EstimatedSalary.max()),
    50000
)

# Prediction
input_data = sc.transform([[age, salary]])
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# Output
st.subheader("🔮 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("✅ Likely to PURCHASE")
    else:
        st.error("❌ Not likely to purchase")

with col2:
    st.metric("Purchase Probability", f"{probability:.2f}")

# Model performance
y_pred = model.predict(X_test)

st.subheader("📊 Model Performance")

col3, col4 = st.columns(2)

with col3:
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.2f}")

with col4:
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix")
    st.write(cm)

# Visualization
st.subheader("📉 Decision Boundary (Training Data)")

X_set, y_set = sc.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 5, X_set[:, 0].max() + 5, 0.5),
    np.arange(X_set[:, 1].min() - 5000, X_set[:, 1].max() + 5000, 500)
)

Z = model.predict(
    sc.transform(np.array([X1.ravel(), X2.ravel()]).T)
).reshape(X1.shape)

fig, ax = plt.subplots()

ax.contourf(X1, X2, Z, alpha=0.3)

for i, j in enumerate(np.unique(y_set)):
    ax.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        label=("Not Purchased" if j == 0 else "Purchased")
    )

ax.set_xlabel("Age")
ax.set_ylabel("Estimated Salary")
ax.set_title("Decision Boundary")
ax.legend()

st.pyplot(fig)
