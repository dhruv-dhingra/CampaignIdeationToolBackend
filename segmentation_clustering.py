import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to generate the dataset
def generate_dataset(num_entries=1000):
    np.random.seed(42)  # For reproducibility
    age = np.random.randint(18, 70, size=num_entries)

    # Define distributions for social caste, economic group, and religion
    social_caste = np.random.choice(['General', 'OBC', 'SC'], size=num_entries, p=[0.4, 0.4, 0.2])
    economic_group = np.random.choice(['Rural', 'Middle-Class', 'Affluent'], size=num_entries, p=[0.4, 0.4, 0.2])
    religion = np.random.choice(['Hinduism', 'Islam', 'Sikhism', 'Christianity'], size=num_entries, p=[0.5, 0.2, 0.2, 0.1])

    # Create variability in age based on the group
    age += np.random.choice([-5, 0, 5], size=num_entries, p=[0.3, 0.4, 0.3])  # Introduce slight variability

    # Map categorical data to numerical for clustering
    social_caste_numeric = np.array([1 if caste == 'General' else 2 if caste == 'OBC' else 3 for caste in social_caste])
    economic_group_numeric = np.array([1 if econ == 'Rural' else 2 if econ == 'Middle-Class' else 3 for econ in economic_group])
    religion_numeric = np.array([1 if rel == 'Hinduism' else 2 if rel == 'Islam' else 3 if rel == 'Sikhism' else 4 for rel in religion])

    primary_data = {
        'Customer ID': np.arange(1, num_entries + 1),
        'Age': age,
        'Social Caste': social_caste_numeric,
        'Economic Group': economic_group_numeric,
        'Religion': religion_numeric
    }
    
    primary_df = pd.DataFrame(primary_data)
    return primary_df

def kmeans_clustering(df, feature):
    clustering_data = df[[feature]]
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['KMeans Cluster'] = kmeans.fit_predict(clustering_data)
    return df

def dbscan_clustering(df, feature, eps, min_samples):
    clustering_data = df[[feature]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_data)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_features)
    df['DBSCAN Cluster'] = clusters
    return df

def hierarchical_clustering(df, feature):
    clustering_data = df[[feature]]
    hierarchical = AgglomerativeClustering(n_clusters=3)
    df['Hierarchical Cluster'] = hierarchical.fit_predict(clustering_data)
    return df

# Streamlit App
def main():
    st.title("CRM Segmentation Dataset Viewer")

    # Generate dataset
    df = generate_dataset()

    # Display a sample of the dataset
    st.subheader("Sample Dataset")
    st.write(df.sample(100))  # Display a sample of 100 rows

    # Clustering by Social Caste
    st.subheader("Clustering by Social Caste")
    df_kmeans = kmeans_clustering(df, 'Social Caste')

    # Visualize K-Means clusters
    st.subheader("K-Means Cluster Visualization (Social Caste)")
    plt.figure(figsize=(8, 4))
    plt.scatter(df['Age'], df['Social Caste'], c=df['KMeans Cluster'], cmap='viridis', alpha=0.6)
    plt.title('K-Means Clustering by Social Caste')
    plt.xlabel('Age')
    plt.ylabel('Social Caste')
    st.pyplot(plt)

    # Clustering by Economic Group
    st.subheader("Clustering by Economic Group")
    eps = st.slider("Select epsilon (eps) value for DBSCAN:", 0.1, 2.0, 0.5, 0.1)
    min_samples = st.slider("Select minimum samples for DBSCAN:", 1, 10, 5)

    df_dbscan = dbscan_clustering(df, 'Economic Group', eps, min_samples)

    # Visualize DBSCAN clusters
    st.subheader("DBSCAN Cluster Visualization (Economic Group)")
    plt.figure(figsize=(8, 4))
    plt.scatter(df['Age'], df['Economic Group'], c=df['DBSCAN Cluster'], cmap='plasma', alpha=0.6)
    plt.title('DBSCAN Clustering by Economic Group')
    plt.xlabel('Age')
    plt.ylabel('Economic Group')
    st.pyplot(plt)

    # Clustering by Religion
    st.subheader("Clustering by Religion")
    df_hierarchical = hierarchical_clustering(df, 'Religion')

    # Visualize Hierarchical clusters
    st.subheader("Hierarchical Cluster Visualization (Religion)")
    plt.figure(figsize=(8, 4))
    plt.scatter(df['Age'], df['Religion'], c=df['Hierarchical Cluster'], cmap='cividis', alpha=0.6)
    plt.title('Hierarchical Clustering by Religion')
    plt.xlabel('Age')
    plt.ylabel('Religion')
    st.pyplot(plt)

if __name__ == "__main__":
    main()

