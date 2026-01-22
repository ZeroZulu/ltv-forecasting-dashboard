#!/usr/bin/env python3
"""
Quick setup script to regenerate all data files.
Run this after cloning to populate the data directory with fresh analysis.
"""

import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ltv_models import run_ltv_analysis
import json
import shutil

def setup():
    """Regenerate all data files."""
    print("=" * 60)
    print("LTV FORECASTING - DATA SETUP")
    print("=" * 60)
    
    # Run the analysis
    predictions, channel_insights, metrics, importance, features = run_ltv_analysis(
        n_players=10000
    )
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Export dashboard data
    dashboard_data = {
        'metrics': metrics,
        'total_players': len(predictions),
        'total_predicted_ltv': round(predictions['ltv_ensemble'].sum(), 2),
        'avg_ltv': round(predictions['ltv_ensemble'].mean(), 2)
    }
    
    with open('data/dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Export CSVs
    predictions.to_csv('data/player_predictions.csv', index=False)
    channel_insights.to_csv('data/channel_insights.csv', index=False)
    importance.to_csv('data/feature_importance.csv', index=False)
    
    # Copy to outputs as well
    shutil.copy('data/dashboard_data.json', 'outputs/')
    shutil.copy('data/player_predictions.csv', 'outputs/')
    shutil.copy('data/channel_insights.csv', 'outputs/')
    shutil.copy('data/feature_importance.csv', 'outputs/')
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  data/dashboard_data.json")
    print("  data/player_predictions.csv")
    print("  data/channel_insights.csv")
    print("  data/feature_importance.csv")
    print("\nYou can now run:")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    setup()
