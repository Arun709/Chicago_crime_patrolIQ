import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def prepare_features(df):
    """
    Prepare features for dimensionality reduction
    """
    feature_cols = ['Latitude', 'Longitude', 'Hour', 'Month', 'Is_Weekend',
                    'Crime_Severity_Score', 'Crime_Type_Encoded', 
                    'Location_Desc_Encoded']
    
    # Add day of week encoding
    df['Day_of_Week_Encoded'] = pd.factorize(df['Day_of_Week'])[0]
    feature_cols.append('Day_of_Week_Encoded')
    
    X = df[feature_cols].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols

def apply_pca(X_scaled, feature_cols, n_components=3):
    """
    Apply PCA for dimensionality reduction
    """
    print("\n" + "="*50)
    print("PCA - Principal Component Analysis")
    print("="*50)
    
    # Full PCA to see explained variance
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Scree plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.axhline(y=0.7, color='r', linestyle='--', label='70% threshold')
    plt.axhline(y=0.8, color='g', linestyle='--', label='80% threshold')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, min(11, len(pca_full.explained_variance_ratio_) + 1)),
            pca_full.explained_variance_ratio_[:10])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Component Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png', dpi=300)
    plt.close()
    
    # Apply PCA with n_components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = sum(pca.explained_variance_ratio_)
    print(f"Components: {n_components}")
    print(f"Total Explained Variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    print(f"\nIndividual Component Variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
    
    # Feature importance
    print(f"\nTop Features by Importance (PC1):")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.abs(pca.components_[0])
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(5))
    
    # Log to MLflow
    with mlflow.start_run(run_name="PCA_Analysis"):
        mlflow.log_param("n_components", n_components)
        mlflow.log_metric("explained_variance", explained_var)
        for i, var in enumerate(pca.explained_variance_ratio_):
            mlflow.log_metric(f"pc{i+1}_variance", var)
        mlflow.log_artifact('pca_variance_analysis.png')
    
    return X_pca, pca

def visualize_pca(X_pca, df, sample_size=10000):
    """
    Visualize PCA results
    """
    # Sample for visualization
    indices = np.random.choice(len(X_pca), min(sample_size, len(X_pca)), replace=False)
    X_sample = X_pca[indices]
    df_sample = df.iloc[indices]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color by crime type
    scatter1 = axes[0].scatter(X_sample[:, 0], X_sample[:, 1], 
                              c=df_sample['Crime_Type_Encoded'], 
                              cmap='tab20', alpha=0.5, s=10)
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    axes[0].set_title('PCA Visualization - Colored by Crime Type')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Color by hour
    scatter2 = axes[1].scatter(X_sample[:, 0], X_sample[:, 1], 
                              c=df_sample['Hour'], 
                              cmap='twilight', alpha=0.5, s=10)
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    axes[1].set_title('PCA Visualization - Colored by Hour of Day')
    plt.colorbar(scatter2, ax=axes[1], label='Hour')
    
    plt.tight_layout()
    plt.savefig('pca_visualizations.png', dpi=300)
    plt.close()

def apply_tsne(X_scaled, df, sample_size=10000, perplexity=30):
    """
    Apply t-SNE for 2D visualization
    """
    print("\n" + "="*50)
    print("t-SNE - t-Distributed Stochastic Neighbor Embedding")
    print("="*50)
    
    # Sample data (t-SNE is computationally expensive)
    indices = np.random.choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[indices]
    df_sample = df.iloc[indices]
    
    print(f"Running t-SNE on {len(X_sample)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)
    
    print("t-SNE transformation complete")
    
    # Log to MLflow
    with mlflow.start_run(run_name="TSNE_Analysis"):
        mlflow.log_param("n_components", 2)
        mlflow.log_param("perplexity", perplexity)
        mlflow.log_param("sample_size", len(X_sample))
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color by crime type
    scatter1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=df_sample['Crime_Type_Encoded'], 
                              cmap='tab20', alpha=0.5, s=10)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].set_title('t-SNE Visualization - Colored by Crime Type')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Color by hour
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=df_sample['Hour'], 
                              cmap='twilight', alpha=0.5, s=10)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title('t-SNE Visualization - Colored by Hour of Day')
    plt.colorbar(scatter2, ax=axes[1], label='Hour')
    
    plt.tight_layout()
    plt.savefig('tsne_visualizations.png', dpi=300)
    plt.close()
    
    return X_tsne

# Main execution
if __name__ == "__main__":
    # Set MLflow experiment
    mlflow.set_experiment("PatrolIQ_DimReduction")
    
    # Load data
    df = pd.read_csv('chicago_crimes_with_clusters.csv')
    
    # Prepare features
    X_scaled, feature_cols = prepare_features(df)
    
    # Apply PCA
    X_pca, pca = apply_pca(X_scaled, feature_cols, n_components=3)
    visualize_pca(X_pca, df)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X_scaled, df, sample_size=10000)
    
    # Save PCA results
    df['PCA_1'] = X_pca[:, 0]
    df['PCA_2'] = X_pca[:, 1]
    df['PCA_3'] = X_pca[:, 2]
    
    df.to_csv('chicago_crimes_final.csv', index=False)
    print("\nDimensionality reduction complete. Results saved to 'chicago_crimes_final.csv'")
