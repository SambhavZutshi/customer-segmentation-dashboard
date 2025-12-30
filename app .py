import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧠 Customer Segmentation using K-Means")
st.markdown("Behavior-driven clustering with actionable business insights")

@st.cache_data
def load_data():
    df = pd.read_csv("customer_segmentation_data.csv")
    df = df.drop(columns=["id"])
    return df

df = load_data()

features = [
    "age",
    "income",
    "spending_score",
    "membership_years",
    "purchase_frequency",
    "last_purchase_amount"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("⚙️ Model Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 8, 4)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# ---- DOWNLOAD BUTTON ----
st.sidebar.markdown("### ⬇️ Export Results")
csv = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Clustered CSV",
    data=csv,
    file_name="customer_segmentation_clustered.csv",
    mime="text/csv"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "📈 Cluster Visuals", "🧾 Cluster Profiles", "💼 Business Insights"]
)

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["income"])
        ax.set_xlabel("Income")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.hist(df["spending_score"])
        ax.set_xlabel("Spending Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    for c in df["Cluster"].unique():
        subset = df[df["Cluster"] == c]
        ax.scatter(subset["income"], subset["spending_score"], label=f"Cluster {c}", alpha=0.6)
    ax.scatter(centroids[:, 1], centroids[:, 2], marker="X", s=200, label="Centroids")
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for c in df["Cluster"].unique():
        subset = df[df["Cluster"] == c]
        ax.scatter(subset["purchase_frequency"], subset["last_purchase_amount"], label=f"Cluster {c}", alpha=0.6)
    ax.scatter(centroids[:, 4], centroids[:, 5], marker="X", s=200, label="Centroids")
    ax.set_xlabel("Purchase Frequency")
    ax.set_ylabel("Last Purchase Amount")
    ax.legend()
    st.pyplot(fig)

with tab3:
    summary = df.groupby("Cluster").agg(
        Avg_Income=("income", "mean"),
        Avg_Spending_Score=("spending_score", "mean"),
        Avg_Purchase_Frequency=("purchase_frequency", "mean"),
        Avg_Last_Purchase_Amount=("last_purchase_amount", "mean"),
        Customers=("Cluster", "count")
    ).round(2)
    st.dataframe(summary)

with tab4:
    st.markdown("""
    ### 🟢 Premium Loyal Customers
    - Loyalty & VIP programs
    - Exclusive access
    - Personalized rewards

    ### 🔵 Price-Sensitive Elites
    - Discount-led campaigns
    - Value bundles

    ### 🟡 Deal-Driven Shoppers
    - Flash sales
    - Limited-time offers

    ### 🔴 At-Risk Customers
    - Re-engagement emails
    - Win-back discounts
    """)
