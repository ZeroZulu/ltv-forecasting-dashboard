"""
LTV Model Improvements - Fixes for identified issues
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================================
# FIX 1: Better MAPE Calculation (handles zeros)
# ============================================================================

def symmetric_mape(actual, predicted):
    """
    Symmetric MAPE - handles zero values properly.
    More robust than standard MAPE for LTV prediction.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    denominator = np.where(denominator == 0, 1, denominator)
    
    smape = np.mean(np.abs(actual - predicted) / denominator) * 100
    return smape


def weighted_mape(actual, predicted, min_value=1.0):
    """
    Weighted MAPE - filters out very low values.
    Better for LTV where F2P players skew the metric.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Only include players with actual LTV > min_value
    mask = actual > min_value
    
    if mask.sum() == 0:
        return np.nan
    
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mape


def calculate_all_metrics(actual, predicted):
    """Calculate comprehensive metrics for LTV models."""
    
    metrics = {
        'r2': r2_score(actual, predicted),
        'mae': mean_absolute_error(actual, predicted),
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
        'mape_raw': np.mean(np.abs((actual - predicted) / np.clip(actual, 1, None))) * 100,
        'smape': symmetric_mape(actual, predicted),
        'wmape_10': weighted_mape(actual, predicted, min_value=10),
        'wmape_50': weighted_mape(actual, predicted, min_value=50),
    }
    
    return metrics


# ============================================================================
# FIX 2: Improved BG/NBD Model with Better Optimization
# ============================================================================

class ImprovedBGNBDModel:
    """
    BG/NBD Model with improved optimization using:
    - Differential Evolution for global optimization
    - Better parameter bounds
    - Multiple restarts
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
        
    def _log_likelihood(self, params, frequency, recency, T):
        """Calculate negative log-likelihood."""
        r, alpha, a, b = params
        
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e10
        
        try:
            # Ensure numerical stability
            frequency = np.clip(frequency, 0, 1000)
            recency = np.clip(recency, 0, 10000)
            T = np.clip(T, 1, 10000)
            
            # BG/NBD log-likelihood
            ln_A1 = (
                stats.gammaln(r + frequency) - stats.gammaln(r) +
                r * np.log(alpha) - (r + frequency) * np.log(alpha + T)
            )
            
            ln_A2 = (
                stats.gammaln(a + b) + stats.gammaln(b + frequency) -
                stats.gammaln(b) - stats.gammaln(a + b + frequency)
            )
            
            # Component for repeat purchasers
            mask = frequency > 0
            ln_A3 = np.zeros_like(frequency, dtype=float)
            
            if mask.any():
                ln_A3[mask] = (
                    stats.gammaln(a + 1) + 
                    stats.gammaln(np.maximum(b + frequency[mask] - 1, 0.001)) -
                    stats.gammaln(a + b + frequency[mask]) +
                    (r + frequency[mask]) * np.log(np.maximum(alpha + recency[mask], 0.001)) -
                    (r + frequency[mask]) * np.log(alpha + T[mask])
                )
            
            # Combine terms
            ll = ln_A1 + ln_A2
            
            # Add A3 component where applicable
            exp_A3 = np.exp(np.clip(ln_A3, -700, 700))
            ll = ll + np.log(1 + exp_A3 * mask)
            
            # Filter out invalid values
            ll = ll[np.isfinite(ll)]
            
            return -np.sum(ll) if len(ll) > 0 else 1e10
            
        except Exception as e:
            return 1e10
    
    def fit(self, frequency, recency, T):
        """Fit using Differential Evolution for global optimization."""
        
        # Filter valid data
        valid_mask = (T > 0) & np.isfinite(frequency) & np.isfinite(recency)
        freq_fit = frequency[valid_mask]
        rec_fit = recency[valid_mask]
        T_fit = T[valid_mask]
        
        # Use differential evolution for global optimization
        bounds = [(0.01, 10), (0.01, 100), (0.01, 10), (0.01, 10)]
        
        try:
            result = differential_evolution(
                self._log_likelihood,
                bounds,
                args=(freq_fit, rec_fit, T_fit),
                seed=42,
                maxiter=500,
                tol=1e-6,
                workers=1
            )
            self.params = result.x
        except Exception:
            # Fallback to L-BFGS-B with multiple restarts
            best_ll = 1e10
            best_params = [1.0, 1.0, 1.0, 1.0]
            
            for _ in range(5):
                init = [np.random.uniform(0.1, 5) for _ in range(4)]
                try:
                    result = minimize(
                        self._log_likelihood,
                        init,
                        args=(freq_fit, rec_fit, T_fit),
                        method='L-BFGS-B',
                        bounds=bounds
                    )
                    if result.fun < best_ll:
                        best_ll = result.fun
                        best_params = result.x
                except Exception:
                    continue
            
            self.params = best_params
        
        self.fitted = True
        return self
    
    def predict_alive_probability(self, frequency, recency, T):
        """Predict probability customer is still active."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        r, alpha, a, b = self.params
        
        # Handle edge cases
        frequency = np.array(frequency)
        recency = np.array(recency)
        T = np.array(T)
        
        # For zero-frequency customers, use tenure-based estimate
        zero_freq_mask = frequency == 0
        
        # Calculate for customers with purchases
        denom = np.maximum(b + frequency - 1, 0.01)
        ratio = np.clip((alpha + T) / (alpha + recency + 0.01), 0.01, 1000)
        
        p_alive = 1 / (1 + (a / denom) * np.power(ratio, r + frequency))
        
        # For zero-frequency, estimate based on tenure
        tenure_decay = np.exp(-T / 180)
        p_alive = np.where(zero_freq_mask, 0.3 + 0.4 * tenure_decay, p_alive)
        
        return np.clip(p_alive, 0.01, 0.99)
    
    def predict_purchases(self, frequency, recency, T, periods=365):
        """Predict expected purchases in future period."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        r, alpha, a, b = self.params
        p_alive = self.predict_alive_probability(frequency, recency, T)
        
        # Historical purchase rate
        frequency = np.array(frequency)
        T = np.array(T)
        
        # Expected rate based on history
        lambda_rate = np.where(
            frequency > 0,
            frequency / np.maximum(T, 1),
            0.01  # Small base rate for non-purchasers
        )
        
        # Adjust for model parameters and alive probability
        expected = lambda_rate * periods * p_alive
        
        return np.maximum(expected, 0)


# ============================================================================
# FIX 3: Improved Gamma-Gamma Model
# ============================================================================

class ImprovedGammaGammaModel:
    """
    Gamma-Gamma Model with better optimization.
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
        self.population_mean = None
        
    def _log_likelihood(self, params, frequency, monetary):
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
                (p * frequency - 1) * np.log(np.maximum(monetary, 0.01)) +
                (p * frequency) * np.log(frequency) -
                (p * frequency + q) * np.log(frequency * monetary + gamma)
            )
            
            ll = ll[np.isfinite(ll)]
            return -np.sum(ll) if len(ll) > 0 else 1e10
            
        except Exception:
            return 1e10
    
    def fit(self, frequency, monetary):
        """Fit the Gamma-Gamma model."""
        
        # Filter to valid data (customers with 2+ purchases)
        mask = (frequency > 1) & (monetary > 0) & np.isfinite(monetary)
        
        if mask.sum() < 10:
            # Not enough data, use simpler approach
            self.params = [1.0, 1.0, monetary[monetary > 0].mean() if (monetary > 0).any() else 1.0]
            self.population_mean = monetary[monetary > 0].mean() if (monetary > 0).any() else 0
            self.fitted = True
            return self
        
        freq_fit = frequency[mask]
        mon_fit = monetary[mask]
        
        self.population_mean = mon_fit.mean()
        
        # Optimization
        bounds = [(0.01, 10), (0.01, 10), (0.01, mon_fit.mean() * 10)]
        
        try:
            result = differential_evolution(
                self._log_likelihood,
                bounds,
                args=(freq_fit, mon_fit),
                seed=42,
                maxiter=300,
                tol=1e-6
            )
            self.params = result.x
        except Exception:
            # Fallback
            self.params = [1.0, 2.0, self.population_mean]
        
        self.fitted = True
        return self
    
    def predict_monetary_value(self, frequency, monetary):
        """Predict expected monetary value."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        p, q, gamma = self.params
        
        frequency = np.array(frequency)
        monetary = np.array(monetary)
        
        # For customers with purchase history
        has_history = (frequency > 0) & (monetary > 0)
        
        expected = np.full_like(frequency, self.population_mean, dtype=float)
        
        if has_history.any():
            expected[has_history] = (
                (gamma + frequency[has_history] * monetary[has_history]) /
                np.maximum(q + frequency[has_history] * p - 1, 0.01)
            )
        
        return np.clip(expected, 0, monetary.max() * 5 if monetary.max() > 0 else 1000)


# ============================================================================
# Updated Evaluation Function
# ============================================================================

def evaluate_ltv_model(actual, predicted, model_name="LTV Model"):
    """
    Comprehensive evaluation with proper handling of edge cases.
    """
    
    metrics = calculate_all_metrics(actual, predicted)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - EVALUATION METRICS")
    print('='*60)
    print(f"R² Score:              {metrics['r2']:.3f}")
    print(f"MAE:                   ${metrics['mae']:.2f}")
    print(f"RMSE:                  ${metrics['rmse']:.2f}")
    print(f"MAPE (raw):            {metrics['mape_raw']:.1f}%  ⚠️  (inflated by F2P)")
    print(f"SMAPE (symmetric):     {metrics['smape']:.1f}%  ✓  (recommended)")
    print(f"WMAPE (LTV>$10):       {metrics['wmape_10']:.1f}%  ✓  (paying users)")
    print(f"WMAPE (LTV>$50):       {metrics['wmape_50']:.1f}%  ✓  (engaged users)")
    print('='*60)
    
    # Interpretation
    print("\n📊 Interpretation:")
    if metrics['r2'] > 0.8:
        print("  ✅ R² > 0.80: Excellent predictive power")
    elif metrics['r2'] > 0.6:
        print("  ✓ R² > 0.60: Good predictive power")
    else:
        print("  ⚠️ R² < 0.60: Model needs improvement")
    
    if metrics['smape'] < 50:
        print("  ✅ SMAPE < 50%: Excellent accuracy")
    elif metrics['smape'] < 80:
        print("  ✓ SMAPE < 80%: Good accuracy")
    else:
        print("  ⚠️ SMAPE > 80%: Consider model improvements")
    
    return metrics


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Improved LTV Models Module")
    print("Import and use in your notebook:")
    print("  from ltv_improvements import ImprovedBGNBDModel, evaluate_ltv_model")
