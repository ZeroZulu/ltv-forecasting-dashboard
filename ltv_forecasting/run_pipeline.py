#!/usr/bin/env python3
"""
LTV Forecasting Pipeline - Main Execution Script
Generates all analysis outputs and visualizations
"""

import sys
sys.path.insert(0, '/home/claude/ltv_forecasting/src')

import numpy as np
import pandas as pd
import json
from datetime import datetime
from ltv_models import (
    GamingDataGenerator, 
    LTVFeatureEngineer,
    EnsembleLTVPredictor,
    run_ltv_analysis
)

def export_for_dashboard(predictions_df, channel_insights, metrics, 
                         importance_df, features_df, output_dir):
    """Export data for the interactive dashboard."""
    
    # Channel summary for charts
    channel_data = channel_insights.to_dict('records')
    
    # Time series simulation (cohort analysis)
    cohort_data = []
    for month in range(1, 13):
        for channel in channel_insights['acquisition_channel'].unique():
            channel_pred = predictions_df[
                predictions_df['acquisition_channel'] == channel
            ]
            # Simulate monthly LTV accumulation
            monthly_ltv = channel_pred['ltv_ensemble'].mean() * (month / 12) ** 0.7
            cohort_data.append({
                'month': month,
                'channel': channel,
                'cumulative_ltv': round(monthly_ltv, 2)
            })
    
    # Segment distribution
    segment_dist = features_df.groupby(['acquisition_channel', 'player_segment']).size()
    segment_dist = segment_dist.reset_index(name='count')
    
    # Survival curves by channel
    survival_data = []
    for channel in predictions_df['acquisition_channel'].unique():
        channel_df = predictions_df[predictions_df['acquisition_channel'] == channel]
        for day in [7, 14, 30, 60, 90, 180, 365]:
            # Simulate retention based on alive probability
            base_retention = channel_df['alive_probability'].mean()
            decay = np.exp(-day / 180)
            retention = base_retention * decay + (1 - decay) * 0.1
            survival_data.append({
                'channel': channel,
                'day': day,
                'retention_rate': round(retention * 100, 1)
            })
    
    # ROI Analysis
    roi_data = []
    for _, row in channel_insights.iterrows():
        roi_data.append({
            'channel': row['acquisition_channel'],
            'cac': row['avg_cac'],
            'ltv': row['avg_ltv'],
            'roi': round((row['avg_ltv'] - row['avg_cac']) / max(row['avg_cac'], 0.01) * 100, 1),
            'payback_days': round(row['avg_cac'] / max(row['avg_ltv'] / 365, 0.01), 0)
        })
    
    # Feature importance top 15
    top_features = importance_df.head(15).to_dict('records')
    
    # LTV distribution by channel
    ltv_dist = []
    for channel in predictions_df['acquisition_channel'].unique():
        channel_df = predictions_df[predictions_df['acquisition_channel'] == channel]
        ltv_dist.append({
            'channel': channel,
            'min': round(channel_df['ltv_ensemble'].min(), 2),
            'q25': round(channel_df['ltv_ensemble'].quantile(0.25), 2),
            'median': round(channel_df['ltv_ensemble'].median(), 2),
            'q75': round(channel_df['ltv_ensemble'].quantile(0.75), 2),
            'max': round(channel_df['ltv_ensemble'].quantile(0.95), 2),  # 95th to avoid outliers
            'mean': round(channel_df['ltv_ensemble'].mean(), 2)
        })
    
    # Compile all data
    dashboard_data = {
        'generated_at': datetime.now().isoformat(),
        'metrics': metrics,
        'channel_summary': channel_data,
        'cohort_analysis': cohort_data,
        'segment_distribution': segment_dist.to_dict('records'),
        'survival_curves': survival_data,
        'roi_analysis': roi_data,
        'feature_importance': top_features,
        'ltv_distribution': ltv_dist,
        'total_players': len(predictions_df),
        'total_predicted_ltv': round(predictions_df['ltv_ensemble'].sum(), 2),
        'avg_ltv': round(predictions_df['ltv_ensemble'].mean(), 2)
    }
    
    # Save JSON
    with open(f'{output_dir}/dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    # Save CSVs for detailed analysis
    predictions_df.to_csv(f'{output_dir}/player_predictions.csv', index=False)
    channel_insights.to_csv(f'{output_dir}/channel_insights.csv', index=False)
    importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    return dashboard_data


def main():
    """Run complete pipeline."""
    
    print("\n" + "=" * 70)
    print("LTV FORECASTING PIPELINE")
    print("=" * 70)
    
    # Run analysis
    predictions, channel_insights, metrics, importance, features = run_ltv_analysis(
        n_players=10000
    )
    
    # Export for dashboard
    output_dir = '/home/claude/ltv_forecasting/outputs'
    dashboard_data = export_for_dashboard(
        predictions, channel_insights, metrics, importance, features, output_dir
    )
    
    print("\n" + "=" * 70)
    print("OUTPUTS GENERATED")
    print("=" * 70)
    print(f"  - dashboard_data.json")
    print(f"  - player_predictions.csv")
    print(f"  - channel_insights.csv")
    print(f"  - feature_importance.csv")
    
    return dashboard_data


if __name__ == "__main__":
    data = main()
    print("\nPipeline completed successfully!")
