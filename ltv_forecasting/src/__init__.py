"""
LTV Forecasting Models Package

This package contains all the models and utilities for
lifetime value prediction by marketing channel.
"""

from .ltv_models import (
    GamingDataGenerator,
    LTVFeatureEngineer,
    BGNBDModel,
    GammaGammaModel,
    SurvivalAnalyzer,
    GradientBoostingLTV,
    QuantileLTVModel,
    EnsembleLTVPredictor,
    run_ltv_analysis
)

__version__ = "1.0.0"
__author__ = "Data Science Portfolio"
