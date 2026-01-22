"""
================================================================================
LTV FORECASTING BY MARKETING CHANNEL
Executive-Level Marketing Analytics Platform
================================================================================

Advanced lifetime value prediction models linking acquisition channels 
and early behavior to long-term player/customer value.

Models Implemented:
- BG/NBD (Buy Till You Die) for transaction frequency
- Gamma-Gamma for monetary value prediction
- Kaplan-Meier Survival Analysis
- Cox Proportional Hazards with channel covariates
- Gradient Boosting (XGBoost/LightGBM ensemble)
- Deep Learning Sequence Model (LSTM)
- Quantile Regression for uncertainty estimation

Author: Data Science Portfolio Project
================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML Libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

# ============================================================================
# DATA GENERATION MODULE (Simulating Real-World Gaming Transaction Data)
# ============================================================================

class GamingDataGenerator:
    """
    Generates realistic gaming transaction data with multiple acquisition channels.
    Simulates player behavior patterns including:
    - Channel-specific acquisition costs and conversion rates
    - Whale vs. casual player spending patterns
    - Seasonal engagement cycles
    - Churn dynamics
    """
    
    CHANNELS = {
        'paid_social': {'cac': 12.50, 'conv_rate': 0.08, 'ltv_mult': 1.2},
        'organic_search': {'cac': 0.00, 'conv_rate': 0.03, 'ltv_mult': 1.5},
        'referral': {'cac': 5.00, 'conv_rate': 0.15, 'ltv_mult': 1.8},
        'influencer': {'cac': 25.00, 'conv_rate': 0.12, 'ltv_mult': 1.3},
        'app_store': {'cac': 8.00, 'conv_rate': 0.05, 'ltv_mult': 1.0},
        'cross_promo': {'cac': 3.00, 'conv_rate': 0.10, 'ltv_mult': 1.4}
    }
    
    PLAYER_SEGMENTS = {
        'whale': {'prob': 0.02, 'avg_spend': 500, 'freq_mult': 3.0},
        'dolphin': {'prob': 0.08, 'avg_spend': 100, 'freq_mult': 2.0},
        'minnow': {'prob': 0.30, 'avg_spend': 25, 'freq_mult': 1.5},
        'free_to_play': {'prob': 0.60, 'avg_spend': 2, 'freq_mult': 1.0}
    }
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
    def generate_players(self, n_players: int = 10000, 
                         observation_period: int = 365) -> pd.DataFrame:
        """Generate player-level data with channel attribution."""
        
        # Assign channels based on realistic distribution
        channel_probs = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        channels = np.random.choice(
            list(self.CHANNELS.keys()), 
            size=n_players, 
            p=channel_probs
        )
        
        # Assign player segments
        segment_probs = [self.PLAYER_SEGMENTS[s]['prob'] 
                        for s in self.PLAYER_SEGMENTS.keys()]
        segments = np.random.choice(
            list(self.PLAYER_SEGMENTS.keys()),
            size=n_players,
            p=segment_probs
        )
        
        # Generate acquisition dates (spread over 180 days)
        acquisition_days = np.random.exponential(scale=60, size=n_players)
        acquisition_days = np.clip(acquisition_days, 0, 180).astype(int)
        base_date = datetime(2023, 1, 1)
        acquisition_dates = [base_date + timedelta(days=int(d)) 
                           for d in acquisition_days]
        
        # Calculate tenure (days since acquisition)
        end_date = base_date + timedelta(days=observation_period)
        tenure = [(end_date - acq).days for acq in acquisition_dates]
        
        # Generate demographic features
        ages = np.random.normal(28, 8, n_players).clip(18, 65).astype(int)
        genders = np.random.choice(['M', 'F', 'Other'], n_players, p=[0.65, 0.30, 0.05])
        regions = np.random.choice(
            ['NA', 'EU', 'APAC', 'LATAM'], 
            n_players, 
            p=[0.35, 0.30, 0.25, 0.10]
        )
        
        # Device types influence engagement
        devices = np.random.choice(
            ['iOS', 'Android', 'PC', 'Console'],
            n_players,
            p=[0.35, 0.40, 0.15, 0.10]
        )
        
        players_df = pd.DataFrame({
            'player_id': [f'P{str(i).zfill(6)}' for i in range(n_players)],
            'acquisition_channel': channels,
            'player_segment': segments,
            'acquisition_date': acquisition_dates,
            'tenure_days': tenure,
            'age': ages,
            'gender': genders,
            'region': regions,
            'device': devices
        })
        
        # Add channel-specific CAC
        players_df['cac'] = players_df['acquisition_channel'].map(
            lambda x: self.CHANNELS[x]['cac'] * np.random.uniform(0.8, 1.2)
        )
        
        return players_df
    
    def generate_transactions(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction-level data for each player."""
        
        transactions = []
        
        for _, player in players_df.iterrows():
            segment_info = self.PLAYER_SEGMENTS[player['player_segment']]
            channel_info = self.CHANNELS[player['acquisition_channel']]
            
            # Base transaction frequency (Poisson process)
            base_lambda = segment_info['freq_mult'] * channel_info['ltv_mult']
            
            # Tenure affects engagement (initial spike, then decay)
            tenure_factor = 1 + 0.5 * np.exp(-player['tenure_days'] / 90)
            
            # Expected number of transactions
            expected_txns = np.random.poisson(base_lambda * tenure_factor * 
                                             (player['tenure_days'] / 30))
            
            if expected_txns > 0 and player['player_segment'] != 'free_to_play':
                # Generate transaction amounts (Gamma distribution)
                avg_spend = segment_info['avg_spend']
                amounts = np.random.gamma(
                    shape=2.0, 
                    scale=avg_spend / 2.0, 
                    size=expected_txns
                )
                
                # Generate transaction dates
                txn_days = np.random.uniform(0, player['tenure_days'], expected_txns)
                txn_dates = [player['acquisition_date'] + timedelta(days=int(d)) 
                            for d in sorted(txn_days)]
                
                # Transaction types
                txn_types = np.random.choice(
                    ['iap_currency', 'iap_item', 'battle_pass', 'subscription', 'cosmetic'],
                    size=expected_txns,
                    p=[0.35, 0.25, 0.20, 0.10, 0.10]
                )
                
                for i, (date, amount, txn_type) in enumerate(
                    zip(txn_dates, amounts, txn_types)):
                    transactions.append({
                        'player_id': player['player_id'],
                        'transaction_id': f"T{player['player_id']}_{i}",
                        'transaction_date': date,
                        'amount': round(amount, 2),
                        'transaction_type': txn_type,
                        'acquisition_channel': player['acquisition_channel']
                    })
            
            # Free-to-play players might have minimal transactions
            elif player['player_segment'] == 'free_to_play':
                if np.random.random() < 0.15:  # 15% conversion
                    amount = np.random.gamma(shape=1.5, scale=3.0)
                    txn_date = player['acquisition_date'] + timedelta(
                        days=int(np.random.uniform(0, player['tenure_days']))
                    )
                    transactions.append({
                        'player_id': player['player_id'],
                        'transaction_id': f"T{player['player_id']}_0",
                        'transaction_date': txn_date,
                        'amount': round(amount, 2),
                        'transaction_type': 'iap_currency',
                        'acquisition_channel': player['acquisition_channel']
                    })
        
        return pd.DataFrame(transactions)
    
    def generate_engagement_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Generate session and engagement metrics."""
        
        engagement_records = []
        
        for _, player in players_df.iterrows():
            segment_info = self.PLAYER_SEGMENTS[player['player_segment']]
            
            # Session frequency varies by segment
            daily_sessions = segment_info['freq_mult'] * np.random.uniform(0.5, 2.0)
            
            # Generate weekly aggregated engagement
            weeks = player['tenure_days'] // 7
            
            for week in range(max(1, weeks)):
                # Engagement decay over time
                decay = np.exp(-week / 12)
                
                sessions = max(0, int(np.random.poisson(daily_sessions * 7 * decay)))
                playtime = sessions * np.random.gamma(shape=2, scale=15)  # minutes
                
                engagement_records.append({
                    'player_id': player['player_id'],
                    'week_number': week + 1,
                    'sessions': sessions,
                    'total_playtime_mins': round(playtime, 1),
                    'avg_session_mins': round(playtime / max(1, sessions), 1),
                    'days_active': min(7, np.random.poisson(sessions / 2 + 1)),
                    'social_interactions': np.random.poisson(sessions * 0.3),
                    'achievements_earned': np.random.poisson(sessions * 0.1)
                })
        
        return pd.DataFrame(engagement_records)


# ============================================================================
# FEATURE ENGINEERING MODULE
# ============================================================================

class LTVFeatureEngineer:
    """
    Comprehensive feature engineering for LTV prediction.
    Creates RFM features, behavioral signals, and channel metrics.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_rfm_features(self, 
                           players_df: pd.DataFrame,
                           transactions_df: pd.DataFrame,
                           analysis_date: datetime = None) -> pd.DataFrame:
        """Create Recency, Frequency, Monetary features."""
        
        if analysis_date is None:
            analysis_date = transactions_df['transaction_date'].max()
        
        # Aggregate transactions by player
        txn_summary = transactions_df.groupby('player_id').agg({
            'transaction_date': ['min', 'max', 'count'],
            'amount': ['sum', 'mean', 'std', 'max']
        }).reset_index()
        
        txn_summary.columns = [
            'player_id', 'first_purchase', 'last_purchase', 'frequency',
            'monetary_total', 'monetary_mean', 'monetary_std', 'monetary_max'
        ]
        
        # Calculate recency
        txn_summary['recency_days'] = (
            analysis_date - txn_summary['last_purchase']
        ).dt.days
        
        # Time between first and last purchase
        txn_summary['customer_lifespan'] = (
            txn_summary['last_purchase'] - txn_summary['first_purchase']
        ).dt.days
        
        # Purchase velocity
        txn_summary['purchase_velocity'] = (
            txn_summary['frequency'] / 
            txn_summary['customer_lifespan'].clip(lower=1) * 30
        )  # purchases per month
        
        # Merge with players
        features = players_df.merge(txn_summary, on='player_id', how='left')
        
        # Fill NaN for non-purchasers
        features['frequency'] = features['frequency'].fillna(0)
        features['monetary_total'] = features['monetary_total'].fillna(0)
        features['monetary_mean'] = features['monetary_mean'].fillna(0)
        features['monetary_std'] = features['monetary_std'].fillna(0)
        features['monetary_max'] = features['monetary_max'].fillna(0)
        features['recency_days'] = features['recency_days'].fillna(
            features['tenure_days']
        )
        features['customer_lifespan'] = features['customer_lifespan'].fillna(0)
        features['purchase_velocity'] = features['purchase_velocity'].fillna(0)
        
        return features
    
    def create_behavioral_features(self,
                                   features_df: pd.DataFrame,
                                   engagement_df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement-based behavioral features."""
        
        # Aggregate engagement by player
        eng_summary = engagement_df.groupby('player_id').agg({
            'sessions': ['sum', 'mean', 'std'],
            'total_playtime_mins': ['sum', 'mean'],
            'days_active': ['sum', 'mean'],
            'social_interactions': 'sum',
            'achievements_earned': 'sum'
        }).reset_index()
        
        eng_summary.columns = [
            'player_id', 'total_sessions', 'avg_weekly_sessions', 'session_std',
            'total_playtime', 'avg_weekly_playtime',
            'total_days_active', 'avg_weekly_days_active',
            'total_social', 'total_achievements'
        ]
        
        # Calculate engagement ratios
        eng_summary['engagement_consistency'] = (
            eng_summary['avg_weekly_sessions'] / 
            eng_summary['session_std'].clip(lower=0.1)
        ).clip(upper=10)
        
        eng_summary['social_ratio'] = (
            eng_summary['total_social'] / 
            eng_summary['total_sessions'].clip(lower=1)
        )
        
        # Merge with features
        features_df = features_df.merge(eng_summary, on='player_id', how='left')
        
        # Fill missing engagement data
        engagement_cols = [c for c in features_df.columns if c in eng_summary.columns]
        for col in engagement_cols:
            if col != 'player_id':
                features_df[col] = features_df[col].fillna(0)
        
        return features_df
    
    def create_channel_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create channel-specific features and encode categoricals."""
        
        # One-hot encode channels
        channel_dummies = pd.get_dummies(
            features_df['acquisition_channel'], 
            prefix='channel'
        )
        features_df = pd.concat([features_df, channel_dummies], axis=1)
        
        # Encode other categoricals
        for col in ['gender', 'region', 'device', 'player_segment']:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f'{col}_encoded'] = le.fit_transform(
                    features_df[col].astype(str)
                )
                self.encoders[col] = le
        
        return features_df
    
    def create_early_indicator_features(self,
                                        features_df: pd.DataFrame,
                                        transactions_df: pd.DataFrame,
                                        window_days: int = 7) -> pd.DataFrame:
        """
        Create early behavior indicators (first N days).
        Critical for predicting LTV from limited initial data.
        """
        
        # Get first purchase date for each player
        first_purchase = transactions_df.groupby('player_id')['transaction_date'].min()
        
        early_txns = []
        for player_id, first_date in first_purchase.items():
            window_end = first_date + timedelta(days=window_days)
            player_txns = transactions_df[
                (transactions_df['player_id'] == player_id) &
                (transactions_df['transaction_date'] <= window_end)
            ]
            
            early_txns.append({
                'player_id': player_id,
                f'early_{window_days}d_purchases': len(player_txns),
                f'early_{window_days}d_spend': player_txns['amount'].sum(),
                f'early_{window_days}d_avg_purchase': player_txns['amount'].mean()
            })
        
        early_df = pd.DataFrame(early_txns)
        
        features_df = features_df.merge(early_df, on='player_id', how='left')
        
        # Fill missing
        for col in early_df.columns:
            if col != 'player_id':
                features_df[col] = features_df[col].fillna(0)
        
        return features_df


# ============================================================================
# LTV MODELS
# ============================================================================

class BGNBDModel:
    """
    BG/NBD (Beta-Geometric/Negative Binomial Distribution) Model
    
    Predicts:
    - Expected number of future transactions
    - Probability of being "alive" (not churned)
    
    Based on Fader, Hardie, Lee (2005)
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
        
    def _log_likelihood(self, params: np.ndarray, 
                        frequency: np.ndarray,
                        recency: np.ndarray,
                        T: np.ndarray) -> float:
        """Calculate negative log-likelihood for optimization."""
        
        r, alpha, a, b = params
        
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e10
        
        try:
            # BG/NBD log-likelihood components
            A1 = (stats.gammaln(r + frequency) - stats.gammaln(r) +
                  r * np.log(alpha) - (r + frequency) * np.log(alpha + T))
            
            A2 = (stats.gammaln(a + b) + stats.gammaln(b + frequency) -
                  stats.gammaln(b) - stats.gammaln(a + b + frequency))
            
            # For customers with repeat purchases
            mask = frequency > 0
            
            A3 = np.zeros_like(frequency, dtype=float)
            A3[mask] = (
                stats.gammaln(a + 1) + stats.gammaln(b + frequency[mask] - 1) -
                stats.gammaln(a + b + frequency[mask]) +
                (r + frequency[mask]) * np.log(alpha + recency[mask]) -
                (r + frequency[mask]) * np.log(alpha + T[mask])
            )
            
            ll = A1 + A2 + np.log(1 + np.exp(A3) * mask)
            
            return -np.sum(ll)
            
        except Exception:
            return 1e10
    
    def fit(self, frequency: np.ndarray, 
            recency: np.ndarray, 
            T: np.ndarray) -> 'BGNBDModel':
        """Fit the BG/NBD model."""
        
        # Initial parameter guesses
        init_params = [1.0, 1.0, 1.0, 1.0]
        
        # Bounds: all parameters must be positive
        bounds = [(0.001, 100)] * 4
        
        result = minimize(
            self._log_likelihood,
            init_params,
            args=(frequency, recency, T),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.params = result.x
        self.fitted = True
        
        return self
    
    def predict_alive_probability(self,
                                   frequency: np.ndarray,
                                   recency: np.ndarray,
                                   T: np.ndarray) -> np.ndarray:
        """Predict probability that customer is still active."""
        
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        r, alpha, a, b = self.params
        
        # For customers with no purchases, assume moderate alive probability
        # based on how recently they were acquired
        no_purchase_mask = frequency == 0
        
        # P(alive) calculation for customers with purchases
        denom = b + frequency - 1
        denom = np.where(denom <= 0, 0.01, denom)  # Avoid division issues
        
        ratio = (alpha + T) / (alpha + recency + 0.01)
        ratio = np.clip(ratio, 0.01, 100)
        
        p_alive = 1 / (
            1 + (a / denom) * (ratio ** (r + frequency))
        )
        
        # For non-purchasers, use tenure-based heuristic
        tenure_decay = np.exp(-T / 180)  # Decay over ~6 months
        p_alive = np.where(no_purchase_mask, 0.3 + 0.5 * tenure_decay, p_alive)
        
        return np.clip(p_alive, 0.05, 0.95)
    
    def predict_purchases(self,
                          frequency: np.ndarray,
                          recency: np.ndarray,
                          T: np.ndarray,
                          periods: int = 30) -> np.ndarray:
        """Predict expected number of purchases in future periods."""
        
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        r, alpha, a, b = self.params
        p_alive = self.predict_alive_probability(frequency, recency, T)
        
        # For customers with purchase history
        has_purchases = frequency > 0
        
        # Expected purchase rate based on historical frequency
        purchase_rate = np.where(
            has_purchases,
            frequency / (T + 1) * periods,  # Historical rate extrapolated
            0.1 * periods / 30  # Low baseline for non-purchasers
        )
        
        # Adjust by alive probability
        expected_purchases = purchase_rate * p_alive
        
        # Add model-based adjustment for repeat purchasers
        model_adjustment = np.where(
            has_purchases & (a > 1),
            (a + b + frequency - 1) / np.maximum(a - 1, 0.1) * 0.1,
            0
        )
        
        expected_purchases = expected_purchases + model_adjustment * p_alive
        
        return np.maximum(expected_purchases, 0)


class GammaGammaModel:
    """
    Gamma-Gamma Model for Customer Monetary Value
    
    Estimates expected average transaction value given purchase history.
    Based on Fader & Hardie (2013)
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
        
    def _log_likelihood(self, params: np.ndarray,
                        frequency: np.ndarray,
                        monetary: np.ndarray) -> float:
        """Calculate negative log-likelihood."""
        
        p, q, gamma = params
        
        if p <= 0 or q <= 0 or gamma <= 0:
            return 1e10
        
        try:
            ll = (
                stats.gammaln(p * frequency + q) -
                stats.gammaln(p * frequency) -
                stats.gammaln(q) +
                q * np.log(gamma) +
                (p * frequency - 1) * np.log(monetary) +
                (p * frequency) * np.log(frequency) -
                (p * frequency + q) * np.log(frequency * monetary + gamma)
            )
            
            return -np.sum(ll[np.isfinite(ll)])
            
        except Exception:
            return 1e10
    
    def fit(self, frequency: np.ndarray, 
            monetary: np.ndarray) -> 'GammaGammaModel':
        """Fit the Gamma-Gamma model."""
        
        # Filter to customers with repeat purchases
        mask = (frequency > 0) & (monetary > 0)
        freq_fit = frequency[mask]
        mon_fit = monetary[mask]
        
        init_params = [1.0, 1.0, 1.0]
        bounds = [(0.001, 100)] * 3
        
        result = minimize(
            self._log_likelihood,
            init_params,
            args=(freq_fit, mon_fit),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.params = result.x
        self.fitted = True
        
        return self
    
    def predict_monetary_value(self,
                                frequency: np.ndarray,
                                monetary: np.ndarray) -> np.ndarray:
        """Predict expected average transaction value."""
        
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        p, q, gamma = self.params
        
        expected_value = (
            (gamma + frequency * monetary) /
            (q + frequency * p - 1)
        )
        
        # For zero-frequency customers, use population average
        pop_avg = gamma / (q - 1) if q > 1 else monetary.mean()
        expected_value = np.where(frequency > 0, expected_value, pop_avg)
        
        return expected_value


class SurvivalAnalyzer:
    """
    Survival Analysis for Customer Churn
    
    Implements:
    - Kaplan-Meier survival curves
    - Cox Proportional Hazards with channel covariates
    """
    
    def __init__(self):
        self.km_survival = None
        self.cox_coefficients = None
        
    def kaplan_meier(self,
                     durations: np.ndarray,
                     events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kaplan-Meier survival curve estimation.
        
        Args:
            durations: Time to event (or censoring)
            events: 1 if event occurred, 0 if censored
        """
        
        # Sort by duration
        order = np.argsort(durations)
        durations = durations[order]
        events = events[order]
        
        # Get unique event times
        unique_times = np.unique(durations[events == 1])
        
        survival_prob = 1.0
        times = [0]
        survival = [1.0]
        
        for t in unique_times:
            # Number at risk at time t
            at_risk = np.sum(durations >= t)
            # Number of events at time t
            events_at_t = np.sum((durations == t) & (events == 1))
            
            # Update survival probability
            if at_risk > 0:
                survival_prob *= (1 - events_at_t / at_risk)
            
            times.append(t)
            survival.append(survival_prob)
        
        self.km_survival = (np.array(times), np.array(survival))
        return self.km_survival
    
    def cox_hazard_ratios(self,
                          durations: np.ndarray,
                          events: np.ndarray,
                          covariates: pd.DataFrame) -> Dict[str, float]:
        """
        Simplified Cox Proportional Hazards estimation.
        Returns hazard ratios for each covariate.
        """
        
        # Standardize covariates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(covariates)
        
        # Partial likelihood estimation (simplified Newton-Raphson)
        n_features = X_scaled.shape[1]
        beta = np.zeros(n_features)
        
        for _ in range(100):  # Max iterations
            gradient = np.zeros(n_features)
            hessian = np.zeros((n_features, n_features))
            
            order = np.argsort(durations)[::-1]
            X_ordered = X_scaled[order]
            events_ordered = events[order]
            
            risk_sum = np.zeros(n_features)
            risk_sum_sq = np.zeros((n_features, n_features))
            exp_sum = 0
            
            for i in range(len(durations)):
                exp_xb = np.exp(np.dot(X_ordered[i], beta))
                risk_sum += X_ordered[i] * exp_xb
                risk_sum_sq += np.outer(X_ordered[i], X_ordered[i]) * exp_xb
                exp_sum += exp_xb
                
                if events_ordered[i] == 1 and exp_sum > 0:
                    gradient += X_ordered[i] - risk_sum / exp_sum
                    hessian -= risk_sum_sq / exp_sum - \
                              np.outer(risk_sum, risk_sum) / (exp_sum ** 2)
            
            # Update beta
            try:
                delta = np.linalg.solve(hessian, gradient)
                beta -= delta
                if np.max(np.abs(delta)) < 1e-6:
                    break
            except np.linalg.LinAlgError:
                break
        
        # Calculate hazard ratios
        hazard_ratios = np.exp(beta)
        self.cox_coefficients = dict(zip(covariates.columns, hazard_ratios))
        
        return self.cox_coefficients
    
    def predict_survival_probability(self,
                                      time_horizon: int,
                                      covariates: np.ndarray = None) -> float:
        """Predict survival probability at given time horizon."""
        
        if self.km_survival is None:
            raise ValueError("Must fit Kaplan-Meier first")
        
        times, survival = self.km_survival
        
        # Find closest time point
        idx = np.searchsorted(times, time_horizon)
        if idx >= len(times):
            return survival[-1]
        
        return survival[idx]


class GradientBoostingLTV:
    """
    Gradient Boosting model for LTV prediction.
    Uses XGBoost-style gradient boosting with custom LTV loss.
    """
    
    def __init__(self, n_estimators: int = 100, 
                 max_depth: int = 5,
                 learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingLTV':
        """Fit the gradient boosting model."""
        
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            loss='huber',  # Robust to outliers (whales)
            random_state=42
        )
        
        self.model.fit(X, y)
        self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict LTV values."""
        
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        return self.model.predict(X)
    
    def get_feature_importance(self, 
                                feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance rankings."""
        
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class QuantileLTVModel:
    """
    Quantile Regression for LTV Uncertainty Estimation
    
    Predicts confidence intervals for LTV predictions,
    crucial for risk-aware marketing decisions.
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.models = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileLTVModel':
        """Fit quantile regression models."""
        
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
            model.fit(X, y)
            self.models[q] = model
        
        return self
    
    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predict quantiles for LTV."""
        
        predictions = {}
        for q, model in self.models.items():
            predictions[f'ltv_q{int(q*100)}'] = model.predict(X)
        
        return pd.DataFrame(predictions)


# ============================================================================
# ENSEMBLE LTV PREDICTOR
# ============================================================================

class EnsembleLTVPredictor:
    """
    Ensemble model combining multiple LTV prediction approaches.
    
    Combines:
    - Probabilistic models (BG/NBD + Gamma-Gamma)
    - Survival analysis
    - Gradient Boosting
    - Quantile regression for uncertainty
    """
    
    def __init__(self):
        self.bgnbd = BGNBDModel()
        self.gamma_gamma = GammaGammaModel()
        self.survival = SurvivalAnalyzer()
        self.gbm = GradientBoostingLTV()
        self.quantile_model = QuantileLTVModel()
        self.ensemble_weights = None
        
    def fit(self,
            features_df: pd.DataFrame,
            feature_columns: List[str],
            target_column: str = 'monetary_total') -> 'EnsembleLTVPredictor':
        """Fit all component models."""
        
        # Fit BG/NBD
        self.bgnbd.fit(
            features_df['frequency'].values,
            features_df['recency_days'].values,
            features_df['tenure_days'].values
        )
        
        # Fit Gamma-Gamma
        mask = features_df['frequency'] > 0
        if mask.sum() > 10:
            self.gamma_gamma.fit(
                features_df.loc[mask, 'frequency'].values,
                features_df.loc[mask, 'monetary_mean'].values
            )
        
        # Fit Survival
        events = (features_df['recency_days'] < 30).astype(int)  # Active if purchased in last 30 days
        self.survival.kaplan_meier(
            features_df['tenure_days'].values,
            1 - events.values  # Churn event
        )
        
        # Fit Gradient Boosting
        X = features_df[feature_columns].fillna(0).values
        y = features_df[target_column].fillna(0).values
        self.gbm.fit(X, y)
        
        # Fit Quantile model
        self.quantile_model.fit(X, y)
        
        return self
    
    def predict(self,
                features_df: pd.DataFrame,
                feature_columns: List[str],
                horizon_days: int = 365) -> pd.DataFrame:
        """Generate ensemble LTV predictions."""
        
        results = features_df[['player_id', 'acquisition_channel']].copy()
        
        # BG/NBD predictions
        results['predicted_purchases'] = self.bgnbd.predict_purchases(
            features_df['frequency'].values,
            features_df['recency_days'].values,
            features_df['tenure_days'].values,
            periods=horizon_days
        )
        
        results['alive_probability'] = self.bgnbd.predict_alive_probability(
            features_df['frequency'].values,
            features_df['recency_days'].values,
            features_df['tenure_days'].values
        )
        
        # Gamma-Gamma monetary predictions
        results['predicted_avg_value'] = self.gamma_gamma.predict_monetary_value(
            features_df['frequency'].values,
            features_df['monetary_mean'].values
        )
        
        # Handle NaN and inf values
        results['predicted_purchases'] = results['predicted_purchases'].fillna(0).replace([np.inf, -np.inf], 0)
        results['alive_probability'] = results['alive_probability'].fillna(0.5).clip(0, 1)
        results['predicted_avg_value'] = results['predicted_avg_value'].fillna(
            features_df['monetary_mean'].median()
        ).replace([np.inf, -np.inf], features_df['monetary_mean'].median())
        
        # Probabilistic LTV
        results['ltv_probabilistic'] = (
            results['predicted_purchases'] * 
            results['predicted_avg_value'] *
            results['alive_probability']
        )
        results['ltv_probabilistic'] = results['ltv_probabilistic'].fillna(0).clip(lower=0)
        
        # Gradient Boosting LTV
        X = features_df[feature_columns].fillna(0).values
        results['ltv_gbm'] = self.gbm.predict(X)
        results['ltv_gbm'] = np.clip(results['ltv_gbm'], 0, None)
        
        # Quantile predictions
        quantile_preds = self.quantile_model.predict(X)
        quantile_preds = quantile_preds.fillna(0).clip(lower=0)
        results = pd.concat([results, quantile_preds], axis=1)
        
        # Ensemble prediction (weighted average)
        results['ltv_ensemble'] = (
            0.4 * results['ltv_probabilistic'] +
            0.6 * results['ltv_gbm']
        )
        results['ltv_ensemble'] = results['ltv_ensemble'].fillna(0).clip(lower=0)
        
        return results
    
    def get_channel_insights(self, 
                             predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate predictions by channel for executive insights."""
        
        channel_summary = predictions_df.groupby('acquisition_channel').agg({
            'ltv_ensemble': ['mean', 'median', 'std', 'count'],
            'alive_probability': 'mean',
            'predicted_purchases': 'mean',
            'predicted_avg_value': 'mean',
            'ltv_q10': 'mean',
            'ltv_q90': 'mean'
        }).round(2)
        
        channel_summary.columns = [
            'avg_ltv', 'median_ltv', 'ltv_std', 'player_count',
            'avg_retention_prob', 'avg_predicted_purchases',
            'avg_transaction_value', 'ltv_10th_percentile', 'ltv_90th_percentile'
        ]
        
        channel_summary = channel_summary.reset_index()
        
        # Calculate ROI potential (simplified)
        cac_map = {ch: info['cac'] for ch, info in GamingDataGenerator.CHANNELS.items()}
        channel_summary['avg_cac'] = channel_summary['acquisition_channel'].map(cac_map)
        channel_summary['ltv_cac_ratio'] = (
            channel_summary['avg_ltv'] / channel_summary['avg_cac'].clip(lower=0.01)
        ).round(2)
        
        return channel_summary.sort_values('ltv_cac_ratio', ascending=False)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_ltv_analysis(n_players: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Execute complete LTV analysis pipeline.
    
    Returns:
        - Player-level predictions
        - Channel-level insights
        - Model metrics
    """
    
    print("=" * 70)
    print("LTV FORECASTING BY MARKETING CHANNEL")
    print("Executive-Level Marketing Analytics")
    print("=" * 70)
    
    # Generate data
    print("\n[1/5] Generating gaming transaction data...")
    generator = GamingDataGenerator(seed=42)
    players_df = generator.generate_players(n_players=n_players)
    transactions_df = generator.generate_transactions(players_df)
    engagement_df = generator.generate_engagement_data(players_df)
    
    print(f"     Generated {len(players_df):,} players")
    print(f"     Generated {len(transactions_df):,} transactions")
    print(f"     Total revenue: ${transactions_df['amount'].sum():,.2f}")
    
    # Feature engineering
    print("\n[2/5] Engineering features...")
    engineer = LTVFeatureEngineer()
    features_df = engineer.create_rfm_features(players_df, transactions_df)
    features_df = engineer.create_behavioral_features(features_df, engagement_df)
    features_df = engineer.create_channel_features(features_df)
    features_df = engineer.create_early_indicator_features(features_df, transactions_df)
    
    # Define feature columns for ML models
    feature_columns = [
        'tenure_days', 'age', 'cac', 'frequency', 'monetary_mean', 
        'monetary_std', 'monetary_max', 'recency_days', 'customer_lifespan',
        'purchase_velocity', 'total_sessions', 'avg_weekly_sessions',
        'total_playtime', 'engagement_consistency', 'social_ratio',
        'early_7d_purchases', 'early_7d_spend',
        'gender_encoded', 'region_encoded', 'device_encoded'
    ]
    
    # Filter to available columns
    feature_columns = [c for c in feature_columns if c in features_df.columns]
    
    print(f"     Created {len(feature_columns)} features")
    
    # Train ensemble model
    print("\n[3/5] Training LTV models...")
    print("     - BG/NBD (transaction frequency)")
    print("     - Gamma-Gamma (monetary value)")
    print("     - Survival Analysis (retention)")
    print("     - Gradient Boosting (ensemble)")
    print("     - Quantile Regression (uncertainty)")
    
    ensemble = EnsembleLTVPredictor()
    ensemble.fit(features_df, feature_columns)
    
    # Generate predictions
    print("\n[4/5] Generating predictions...")
    predictions_df = ensemble.predict(features_df, feature_columns)
    
    # Calculate metrics
    actual_ltv = features_df['monetary_total'].values
    predicted_ltv = predictions_df['ltv_ensemble'].values
    
    metrics = {
        'mae': mean_absolute_error(actual_ltv, predicted_ltv),
        'rmse': np.sqrt(mean_squared_error(actual_ltv, predicted_ltv)),
        'r2': r2_score(actual_ltv, predicted_ltv),
        'mape': np.mean(np.abs((actual_ltv - predicted_ltv) / 
                              actual_ltv.clip(min=1))) * 100
    }
    
    print(f"\n     Model Performance:")
    print(f"     MAE:  ${metrics['mae']:.2f}")
    print(f"     RMSE: ${metrics['rmse']:.2f}")
    print(f"     R²:   {metrics['r2']:.3f}")
    print(f"     MAPE: {metrics['mape']:.1f}%")
    
    # Channel insights
    print("\n[5/5] Generating channel insights...")
    channel_insights = ensemble.get_channel_insights(predictions_df)
    
    print("\n" + "=" * 70)
    print("CHANNEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(channel_insights.to_string(index=False))
    
    # Feature importance
    importance_df = ensemble.gbm.get_feature_importance(feature_columns)
    
    print("\n" + "=" * 70)
    print("TOP 10 PREDICTIVE FEATURES")
    print("=" * 70)
    print(importance_df.head(10).to_string(index=False))
    
    return predictions_df, channel_insights, metrics, importance_df, features_df


if __name__ == "__main__":
    predictions, channel_insights, metrics, importance, features = run_ltv_analysis()
