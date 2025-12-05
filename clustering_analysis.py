import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

def prepare_clustering_data(df):
    """
    Prepare data for clustering
    """
    # Geographic features
    geo_features = df[['Latitude', 'Longitude']].copy()
    
    # Temporal features
    temporal_features = df[['Hour', 'Month', 'Day_of_Week', 'Is_Weekend']].copy()
    temporal_features['Day_of_Week'] = pd.factorize(temporal_features['Day_of_Week'])[0]
    
    # Normalize features
    scaler_geo = StandardScaler()
    scaler_temporal = StandardScaler()
    
    geo_scaled = scaler_geo.fit_transform(geo_features)
    temporal_scaled = scaler_temporal.fit_transform(temporal_features)
    
    return geo_scaled, temporal_scaled, geo_features, temporal_features

def geographic_kmeans_clustering(geo_scaled, geo_features, n_clusters=7):
    """
    K-Means clustering for geographic hotspots
    """
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING - Geographic Hotspots")
    print("="*50)
    
    # Elbow method to find optimal k
    inertias = []
    silhouette_scores = []
    K_range = range(3, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(geo_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(geo_scaled, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', color='orange')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs K')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kmeans_elbow_analysis.png', dpi=300)
    plt.close()
    
    # Final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(geo_scaled)
    
    # Evaluation metrics
    silhouette = silhouette_score(geo_scaled, labels)
    davies_bouldin = davies_bouldin_score(geo_scaled, labels)
    
    print(f"Number of Clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name="KMeans_Geographic"):
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.sklearn.log_model(kmeans, "model")
        mlflow.log_artifact('kmeans_elbow_analysis.png')
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(geo_features['Longitude'], geo_features['Latitude'], 
                         c=labels, cmap='viridis', alpha=0.3, s=1)
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], 
               c='red', marker='X', s=200, edgecolors='black', linewidths=2, 
               label='Centroids')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('K-Means Crime Hotspot Clusters')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('kmeans_geographic_clusters.png', dpi=300)
    plt.close()
    
    return labels, silhouette, davies_bouldin

def geographic_dbscan_clustering(geo_scaled, geo_features, eps=0.05, min_samples=50):
    """
    DBSCAN clustering for geographic hotspots
    """
    print("\n" + "="*50)
    print("DBSCAN CLUSTERING - Geographic Hotspots")
    print("="*50)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(geo_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of Clusters: {n_clusters}")
    print(f"Number of Noise Points: {n_noise}")
    
    # Calculate metrics (excluding noise points)
    if n_clusters > 1:
        mask = labels != -1
        silhouette = silhouette_score(geo_scaled[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(geo_scaled[mask], labels[mask])
    else:
        silhouette = -1
        davies_bouldin = -1
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name="DBSCAN_Geographic"):
        mlflow.log_param("algorithm", "DBSCAN")
        mlflow.log_param("eps", eps)
        mlflow.log_param("min_samples", min_samples)
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_noise", n_noise)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(geo_features['Longitude'], geo_features['Latitude'], 
                         c=labels, cmap='viridis', alpha=0.3, s=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'DBSCAN Crime Hotspot Clusters (eps={eps}, min_samples={min_samples})')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('dbscan_geographic_clusters.png', dpi=300)
    plt.close()
    
    return labels, silhouette, davies_bouldin

def geographic_hierarchical_clustering(geo_scaled, geo_features, n_clusters=7):
    """
    Hierarchical clustering for geographic hotspots
    """
    print("\n" + "="*50)
    print("HIERARCHICAL CLUSTERING - Geographic Hotspots")
    print("="*50)
    
    # Use a sample for dendrogram (too large otherwise)
    sample_size = min(5000, len(geo_scaled))
    sample_indices = np.random.choice(len(geo_scaled), sample_size, replace=False)
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    linkage_matrix = linkage(geo_scaled[sample_indices], method='ward')
    
    plt.figure(figsize=(15, 7))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.savefig('hierarchical_dendrogram.png', dpi=300)
    plt.close()
    
    # Full clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hierarchical.fit_predict(geo_scaled)
    
    # Evaluation metrics
    silhouette = silhouette_score(geo_scaled, labels)
    davies_bouldin = davies_bouldin_score(geo_scaled, labels)
    
    print(f"Number of Clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name="Hierarchical_Geographic"):
        mlflow.log_param("algorithm", "Hierarchical")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("linkage", "ward")
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.log_artifact('hierarchical_dendrogram.png')
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(geo_features['Longitude'], geo_features['Latitude'], 
                         c=labels, cmap='viridis', alpha=0.3, s=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Hierarchical Crime Hotspot Clusters')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig('hierarchical_geographic_clusters.png', dpi=300)
    plt.close()
    
    return labels, silhouette, davies_bouldin

def temporal_kmeans_clustering(temporal_scaled, n_clusters=5):
    """
    K-Means clustering for temporal patterns
    """
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING - Temporal Patterns")
    print("="*50)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(temporal_scaled)
    
    silhouette = silhouette_score(temporal_scaled, labels)
    davies_bouldin = davies_bouldin_score(temporal_scaled, labels)
    
    print(f"Number of Clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name="KMeans_Temporal"):
        mlflow.log_param("algorithm", "KMeans_Temporal")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_index", davies_bouldin)
        mlflow.sklearn.log_model(kmeans, "model")
    
    return labels, silhouette

# Main execution
if __name__ == "__main__":
    # Set MLflow tracking
    mlflow.set_experiment("PatrolIQ_Clustering")
    
    # Load data
    df = pd.read_csv('chicago_crimes_preprocessed.csv')
    
    # Prepare data
    geo_scaled, temporal_scaled, geo_features, temporal_features = prepare_clustering_data(df)
    
    # Geographic clustering
    kmeans_labels, kmeans_sil, kmeans_db = geographic_kmeans_clustering(geo_scaled, geo_features)
    dbscan_labels, dbscan_sil, dbscan_db = geographic_dbscan_clustering(geo_scaled, geo_features)
    hier_labels, hier_sil, hier_db = geographic_hierarchical_clustering(geo_scaled, geo_features)
    
    # Temporal clustering
    temporal_labels, temporal_sil = temporal_kmeans_clustering(temporal_scaled)
    
    # Save labels
    df['Geographic_Cluster_KMeans'] = kmeans_labels
    df['Geographic_Cluster_DBSCAN'] = dbscan_labels
    df['Geographic_Cluster_Hierarchical'] = hier_labels
    df['Temporal_Cluster'] = temporal_labels
    
    df.to_csv('chicago_crimes_with_clusters.csv', index=False)
    print("\nClustering complete. Results saved to 'chicago_crimes_with_clusters.csv'")
