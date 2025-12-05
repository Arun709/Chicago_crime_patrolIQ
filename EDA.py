import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_preprocessed_data():
    """Load preprocessed data"""
    return pd.read_csv('chicago_crimes_preprocessed.csv')

def perform_eda(df):
    """
    Comprehensive exploratory data analysis
    """
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("\n1. DATASET OVERVIEW")
    print(f"Total Records: {len(df)}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Crime type distribution
    print("\n2. CRIME TYPE DISTRIBUTION")
    crime_counts = df['Primary Type'].value_counts()
    print(crime_counts.head(10))
    
    # Arrest rate
    arrest_rate = (df['Arrest'].sum() / len(df)) * 100
    print(f"\n3. ARREST RATE: {arrest_rate:.2f}%")
    
    # Domestic violence rate
    domestic_rate = (df['Domestic'].sum() / len(df)) * 100
    print(f"4. DOMESTIC VIOLENCE RATE: {domestic_rate:.2f}%")
    
    # Temporal patterns
    print("\n5. TEMPORAL PATTERNS")
    print(f"Crimes by Hour:\n{df['Hour'].value_counts().sort_index().head(10)}")
    print(f"\nCrimes by Day:\n{df['Day_of_Week'].value_counts()}")
    print(f"\nCrimes by Month:\n{df['Month'].value_counts().sort_index()}")
    
    return df

def create_eda_visualizations(df):
    """
    Create EDA visualizations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Top 10 Crime Types
    crime_counts = df['Primary Type'].value_counts().head(10)
    axes[0, 0].barh(crime_counts.index, crime_counts.values, color='steelblue')
    axes[0, 0].set_xlabel('Number of Crimes')
    axes[0, 0].set_title('Top 10 Crime Types')
    axes[0, 0].invert_yaxis()
    
    # 2. Crimes by Hour
    hourly_crimes = df['Hour'].value_counts().sort_index()
    axes[0, 1].plot(hourly_crimes.index, hourly_crimes.values, marker='o', color='crimson')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Number of Crimes')
    axes[0, 1].set_title('Crimes by Hour of Day')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Crimes by Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_crimes = df['Day_of_Week'].value_counts().reindex(day_order)
    axes[0, 2].bar(range(7), day_crimes.values, color='orange')
    axes[0, 2].set_xticks(range(7))
    axes[0, 2].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0, 2].set_title('Crimes by Day of Week')
    
    # 4. Arrest Rate by Crime Type
    top_crimes = df['Primary Type'].value_counts().head(10).index
    arrest_rates = df[df['Primary Type'].isin(top_crimes)].groupby('Primary Type')['Arrest'].mean() * 100
    axes[1, 0].barh(arrest_rates.index, arrest_rates.values, color='green')
    axes[1, 0].set_xlabel('Arrest Rate (%)')
    axes[1, 0].set_title('Arrest Rate by Crime Type')
    axes[1, 0].invert_yaxis()
    
    # 5. Monthly Crime Trend
    monthly_crimes = df['Month'].value_counts().sort_index()
    axes[1, 1].plot(monthly_crimes.index, monthly_crimes.values, marker='s', color='purple')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Crimes')
    axes[1, 1].set_title('Monthly Crime Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Geographic Distribution (Heatmap style scatter)
    sample = df.sample(n=min(10000, len(df)))
    axes[1, 2].scatter(sample['Longitude'], sample['Latitude'], 
                      c=sample['Crime_Severity_Score'], cmap='YlOrRd', 
                      alpha=0.3, s=1)
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    axes[1, 2].set_title('Geographic Crime Distribution')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    print("\nEDA visualizations saved to 'eda_visualizations.png'")
    plt.close()

# Main execution
if __name__ == "__main__":
    df = load_preprocessed_data()
    df = perform_eda(df)
    create_eda_visualizations(df)
