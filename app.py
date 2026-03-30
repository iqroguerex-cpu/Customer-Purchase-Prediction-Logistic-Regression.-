import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Page Configuration
st.set_page_config(page_title="Logistic Regression Dashboard", layout="wide")

st.title("📊 Logistic Regression - Social Network Ads")
st.markdown("""
This dashboard visualizes how a Logistic Regression model classifies users based on **Age** and **Estimated Salary**.
""")

# --- Data Loading ---
@st.cache_data
def load_data():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    return dataset

df = load_data()

# --- Sidebar: Model Parameters & Prediction ---
st.sidebar.header("Model Settings")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.25)
random_state = st.sidebar.number_input("Random State", value=0)

st.sidebar.header("Predict New User")
input_age = st.sidebar.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
input_salary = st.sidebar.number_input("Estimated Salary", value=87000, step=1000)

# --- Logic: Model Training ---
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

classifier = LogisticRegression(random_state=random_state)
classifier.fit(X_train_scaled, y_train)

# --- Sidebar: Real-time Prediction ---
if st.sidebar.button("Predict"):
    prediction = classifier.predict(sc.transform([[input_age, input_salary]]))
    result = "Purchased" if prediction[0] == 1 else "Not Purchased"
    color = "green" if prediction[0] == 1 else "red"
    st.sidebar.markdown(f"### Result: :{color}[{result}]")

# --- Layout: Top Metrics ---
y_pred = classifier.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy Score", f"{acc*100:.2f}%")
col2.metric("Training Samples", len(X_train))
col3.metric("Test Samples", len(X_test))

# --- Layout: Data & Visualization ---
tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Training Set Visual", "Test Set Visual"])

with tab1:
    st.subheader("Raw Data (First 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Confusion Matrix")
    st.write(cm)

# Visualization Helper Function
def plot_results(X_set, y_set, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Adjust resolution for performance
    x_min, x_max = X_set[:, 0].min() - 10, X_set[:, 0].max() + 10
    y_min, y_max = X_set[:, 1].min() - 1000, X_set[:, 1].max() + 1000
    
    # Step size for Age and Salary
    X1, X2 = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 500)) # Larger step for Salary to avoid lag
    
    Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
    
    ax.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['#FA8072', '#1E90FF']))
    ax.set_xlim(X1.min(), X1.max())
    ax.set_ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j, edgecolors='black')
        
    ax.set_title(title)
    ax.set_xlabel('Age')
    ax.set_ylabel('Estimated Salary')
    ax.legend()
    return fig

with tab2:
    st.subheader("Classification Results: Training Set")
    fig_train = plot_results(X_train, y_train, 'Logistic Regression (Training set)')
    st.pyplot(fig_train)

with tab3:
    st.subheader("Classification Results: Test Set")
    fig_test = plot_results(X_test, y_test, 'Logistic Regression (Test set)')
    st.pyplot(fig_test)
