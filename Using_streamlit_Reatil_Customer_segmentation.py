"""
 RFM Customer Segmentation Dashboard with Streamlit
 Developed by: Sobikul Mia
 Description:
 This app loads online retail data, performs RFM analysis,
 applies KMeans clustering, and presents an interactive dashboard.
"""
#
import warnings
warnings.filterwarnings("ignore")    # Turn off warnings
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score
from  kneed import KneeLocator
import joblib
import streamlit as st 
# Streamlit app config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛍️ RFM Customer Segmentation Dashboard")
# Load and clean dataset
@st.cache_data
def load_and_clean_data(dataset):

    dataset= pd.read_excel(dataset)
    dataset = dataset.dropna(subset="CustomerID")
    categorical_col_fill = dataset.select_dtypes(include=object).columns
    for col in categorical_col_fill:
        dataset[col].fillna(dataset[col].mode()[0],inplace=True)
    dataset =dataset.drop(["StockCode","Description","Country"],axis=1)
    dataset = dataset[~dataset["InvoiceNo"].astype(str).str.startswith("C")]
    dataset["InvoiceDate"] =pd.to_datetime(dataset["InvoiceDate"])
    dataset["TotalPrice"]=dataset["Quantity"]*dataset["UnitPrice"]
    return dataset
# Visualize missing values
def vsualization_missing_value(data,titel ="Missing value"):
    missing =data.isna().sum()
    missing =missing[missing>0]
    if not missing.empty:
        plt.figure(figsize=(12,8))
        missing.plot(kind ="bar",color = "tomato")
        plt.title(titel)
        plt.ylabel("Number of missing value")
        plt.xticks(rotation =45)
        plt.tight_layout()
    else:
        print("No Missing value to dispaly")
# Correlation Heatmap
def ploat_Crroleation_Heatmap(corr_matrix,titel = "Crroleation Heatmap"):
    plt.figure(figsize=(5,4))
    sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".5f",linewidths=0.5,linecolor="white",vmax=1,vmin=1)
    plt.title(titel)
# Create RFM features
def create_rfm(df):
    ref_date =df["InvoiceDate"].max()+pd.Timedelta(days = 1)
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (ref_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    }).reset_index()
    rfm.columns = ["CustomerID","Recency","Frequency","Monetary"]
    return rfm
# Scale RFM data
def scale_rfm(rfm_data):
    x =rfm_data[["Recency","Monetary","Frequency"]]

    scaler = StandardScaler()
    x_scaler = scaler.fit_transform(x)
    return x_scaler,scaler
# Find optimal K using Elbow Method
def find_optmal_K(rfm_data):

    
    wcss = []
    k_range = range(1,15)
    for k in k_range:
        kmeans =KMeans(n_clusters=k,random_state=42)
        kmeans.fit(rfm_data)
        wcss.append(kmeans.inertia_)
    kn =KneeLocator(k_range,wcss,curve="convex",direction="decreasing")
    plt.figure(figsize=(6,4))
    plt.plot(k_range,wcss,marker ="o")
    plt.title("Elbow Mthode optimal K")
    plt.xlabel("Number of cluster")
    plt.ylabel("WCSS")
    plt.grid(True)

    return kn.knee
# Silhouette Score plot
def plot_silhouette_scores(x_scaler, max_k=10):
    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(x_scaler)
        score = silhouette_score(x_scaler, labels)
        silhouette_scores.append(score)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(K, silhouette_scores, marker='o', color='green')
    plt.title("Silhouette Scores for Different K", fontsize=14)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    best_k = K[silhouette_scores.index(max(silhouette_scores))]
    print(f" Best k by Silhouette Score: {best_k}")
    return best_k
# Apply KMeans model
def apply_kmeans(x_kmeans,n_clusters):
    kemeans = KMeans(n_clusters=n_clusters,random_state=42)
    x_kmeans= kemeans.fit_predict(x_kmeans)
    
    return x_kmeans,kemeans.cluster_centers_,kemeans
# Summary statistics per cluster
def cluster_summary(rfm_data):
    
    summary = rfm_data.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
    }).rename(columns={"CustomerID": "Num_Customers"}).reset_index()

    return summary
# Assign segment labels to each cluster
def assign_segment_labels(rfm):
    segment_map = {
        0: "Loyal Customers",
        1: "Lost Customers",
        2: "VIP Champions",
        3: "Potential Customers",
        4: "Big Spenders",
        5: "Other Customers" 
        
    }
    rfm["Segment"] = rfm["Cluster"].map(segment_map)
    new_customer_condition = (rfm["Recency"] <= 30) & (rfm["Frequency"] <= 1)
    rfm.loc[new_customer_condition, "Segment"] = "New Customer"
    return rfm
# Visualize clusters
def plot_cluster(rfm_scaled_data,centers):

    colors =["red","green","blue","cyan"]
    plt.figure(figsize=(10, 6))
    for i in rfm_scaled_data["Cluster"].unique():
        cluster_data = rfm_scaled_data[rfm_scaled_data["Cluster"] == i]
        sns.scatterplot(x = cluster_data["Recency"],y =cluster_data["Monetary"],label =f"Cluster{i}",color=colors[i % len(colors)])
    plt.scatter(x=centers[:, 0], y=centers[:, 2], marker="*", s=200, color="black", label="Centroid")
    plt.title("Cluster Visualization (scaled)", fontsize=14)
    plt.xlabel("Recency (scaled)")
    plt.ylabel("Monetary (scaled)")
    plt.grid(True)
    plt.show()
# Plot customer segment distribution
def segment_distibution(distibution):
    segment_counts = distibution["Segment"].value_counts()
    plt.figure(figsize=(10, 6))
    segment_counts.plot(kind="bar", color="skyblue")
    plt.title("Customer Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main app logic
def main_code():
    uploaded_file = st.file_uploader("Upload your Online Retail Excel file", type=["xlsx"])
    if uploaded_file:
        dataset=load_and_clean_data(uploaded_file)
        st.subheader("Missing Value Visualization")
        vsualization_missing_value(dataset)


        corr_matrix = dataset[["CustomerID", "Quantity", "UnitPrice", "TotalPrice"]].corr()
        st.subheader("Correlation Heatmap")
        ploat_Crroleation_Heatmap(corr_matrix)
        rfm = create_rfm(dataset)
        x_scaler,scaler =scale_rfm(rfm)

        st.subheader("Silhouette Score Analysis")
        best_k=plot_silhouette_scores(x_scaler, max_k=10)
        
        st.subheader("Elbow Method of Optimal_K")
        optimal_k = find_optmal_K(x_scaler)
        st.write(f"Optimal number of clusters (Elbow): {optimal_k}")
        
        st.subheader("Applying KMeans Clustering")
        cluster_label,center,kmeans_model =apply_kmeans(x_scaler,optimal_k)
        rfm["Cluster"] = cluster_label

        summary =cluster_summary(rfm)
        st.write("Cluster Summary:")
        st.dataframe(summary)

        rfm_labeled = assign_segment_labels(rfm)
        st.write("Sample Labeled RFM Data:")
        st.dataframe(rfm_labeled.head())
        
        new_customer_count = rfm_labeled[rfm_labeled["Segment"] == "New Customer"].shape[0]
        print(f"Total number of New Customers: {new_customer_count}")


        st.subheader("Cluster Visualization")
        rfm_scaled_data = pd.DataFrame(x_scaler, columns=["Recency","Frequency", "Monetary"])
        rfm_scaled_data["Cluster"] = rfm["Cluster"]
        
        plot_cluster(rfm_scaled_data,center)

        st.subheader("Customer Segment Distribution")
        segment_distibution(rfm_labeled)

        st.subheader("Predict Cluster for New Customer")
    r = st.number_input("Recency (days)", min_value=0)
    f = st.number_input("Frequency (number of purchases)", min_value=0)
    m = st.number_input("Monetary (total spent)", min_value=0.0, format="%.2f")
    if st.button("Predict Cluster"):
        new_data = np.array([[r, m, f]])
        new_scaled = scaler.transform(new_data)
        pred_cluster = kmeans_model.predict(new_scaled)[0]
        st.success(f"✅ This new customer belongs to Cluster {pred_cluster}")

    else:
     st.info("👆 Please upload a valid Online Retail Excel file to begin.")

# Run main app
if __name__ == "__main__":
    main_code()