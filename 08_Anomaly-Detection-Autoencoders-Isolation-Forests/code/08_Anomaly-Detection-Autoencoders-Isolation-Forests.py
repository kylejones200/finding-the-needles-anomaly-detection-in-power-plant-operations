import logging

logger = logging.getLogger(__name__)

# Extracted code from '08_Anomaly-Detection-Autoencoders-Isolation-Forests.md'
# Blocks appear in the same order as in the markdown article.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load production data and aggregate across all MSN codes
data_path = BASE_DIR / "data" / "pr_OK.csv"
df = pd.read_csv(data_path)

pr_cols = [col for col in df.columns if col.isdigit()]
year_totals = df[pr_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

ts = pd.Series(
    data=year_totals.values,
    index=pd.to_datetime(year_totals.index, format="%Y"),
).sort_index()

ts = ts.interpolate(method="linear")

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Prepare features for isolation forest
# Use rolling statistics as features
features = pd.DataFrame({
    'value': ts.values,
    'rolling_mean_3': ts.rolling(3, min_periods=1).mean().values,
    'rolling_std_3': ts.rolling(3, min_periods=1).std().values,
    'diff': ts.diff().fillna(0).values,
    'pct_change': ts.pct_change().fillna(0).values
})

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Fit Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)

anomalies_iso = iso_forest.fit_predict(features_scaled)
anomalies_iso = anomalies_iso == -1  # Convert to boolean

# Get anomaly scores
scores_iso = iso_forest.score_samples(features_scaled)

logger.info(f"Isolation Forest detected {anomalies_iso.sum()} anomalies")
logger.info(f"Anomaly rate: {anomalies_iso.sum() / len(anomalies_iso) * 100:.2f}%")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)

# Prepare sequences for autoencoder
def create_sequences(data, window=10):
    X = []
    for i in range(len(data) - window + 1):
        X.append(data[i:i+window])
    return np.array(X)

window_size = 10
X_sequences = create_sequences(ts.values, window_size)

# Normalize
scaler_ae = StandardScaler()
X_scaled = scaler_ae.fit_transform(X_sequences.reshape(-1, 1)).reshape(X_sequences.shape)

# Build autoencoder
input_dim = window_size
encoding_dim = 5  # Bottleneck dimension

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='linear')(encoder)

autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data (assume most data is normal)
# In practice, you'd filter out known anomalies first
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=0
)

# Reconstruct and calculate error
reconstructed = autoencoder.predict(X_scaled, verbose=0)
reconstruction_error = np.mean((X_scaled - reconstructed) ** 2, axis=1)

# Detect anomalies (high reconstruction error)
threshold_ae = np.percentile(reconstruction_error, 95)  # Top 5% as anomalies
anomalies_ae = reconstruction_error > threshold_ae

# Map back to original time series
anomalies_ae_full = np.zeros(len(ts), dtype=bool)
for i, is_anomaly in enumerate(anomalies_ae):
    if is_anomaly:
        anomalies_ae_full[i:i+window_size] = True

logger.info(f"Autoencoder detected {anomalies_ae_full.sum()} anomaly periods")

def detect_anomalies_statistical(ts, z_threshold=3, iqr_factor=1.5):
    """Detect anomalies using statistical methods"""
    anomalies = np.zeros(len(ts), dtype=bool)
    
    # Z-score method
    mean_val = ts.mean()
    std_val = ts.std()
    z_scores = np.abs((ts.values - mean_val) / std_val)
    anomalies_z = z_scores > z_threshold
    
    # IQR method
    Q1 = ts.quantile(0.25)
    Q3 = ts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    anomalies_iqr = (ts.values < lower_bound) | (ts.values > upper_bound)
    
    # Moving average method
    ma = ts.rolling(5, min_periods=1).mean()
    ma_std = ts.rolling(5, min_periods=1).std()
    anomalies_ma = np.abs(ts.values - ma.values) > (2 * ma_std.values)
    
    # Combine methods
    anomalies = anomalies_z | anomalies_iqr | anomalies_ma
    
    return anomalies, {
        'z_score': anomalies_z,
        'iqr': anomalies_iqr,
        'moving_avg': anomalies_ma
    }

anomalies_stat, methods = detect_anomalies_statistical(ts)

logger.info(f"Statistical methods detected {anomalies_stat.sum()} anomalies")
logger.info(f"  Z-score: {methods['z_score'].sum()}")
logger.info(f"  IQR: {methods['iqr'].sum()}")
logger.info(f"  Moving avg: {methods['moving_avg'].sum()}")

from sklearn.metrics import precision_recall_curve, roc_curve, auc

def evaluate_anomaly_detection(scores, threshold, actual_anomalies=None):
    """Evaluate anomaly detection with different thresholds"""
    predictions = scores > threshold
    
    if actual_anomalies is not None:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(actual_anomalies, predictions)
        recall = recall_score(actual_anomalies, predictions)
        f1 = f1_score(actual_anomalies, predictions)
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    return {'anomaly_count': predictions.sum(), 'anomaly_rate': predictions.mean()}

# Test different thresholds for autoencoder
thresholds = np.linspace(reconstruction_error.min(), reconstruction_error.max(), 50)
results = []

for threshold in thresholds:
    pred = reconstruction_error > threshold
    results.append({
        'threshold': threshold,
        'anomaly_count': pred.sum(),
        'anomaly_rate': pred.mean()
    })

threshold_df = pd.DataFrame(results)

# Visualize threshold selection
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(threshold_df['threshold'], threshold_df['anomaly_rate'], linewidth=2)
ax.axvline(threshold_ae, color='red', linestyle='--', label=f'Selected (95th percentile)')
ax.set_xlabel('Reconstruction Error Threshold', fontsize=11)
ax.set_ylabel('Anomaly Rate', fontsize=11)
ax.set_title('Threshold Selection for Autoencoder', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('threshold_selection.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize all methods
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Original time series
axes[0].plot(ts.index, ts.values, 'b-', linewidth=2, label='Energy Production')
axes[0].set_title('Original Time Series', fontweight='bold')
axes[0].set_ylabel('Production', fontsize=10)
axes[0].legend()
# Isolation Forest
axes[1].plot(ts.index, ts.values, 'b-', linewidth=1.5, alpha=0.5)
axes[1].scatter(ts.index[anomalies_iso], ts.values[anomalies_iso], 
               color='red', s=50, label='Anomalies', marker='x')
axes[1].set_title('Isolation Forest Detection', fontweight='bold')
axes[1].set_ylabel('Production', fontsize=10)
axes[1].legend()
# Autoencoder
axes[2].plot(ts.index, ts.values, 'b-', linewidth=1.5, alpha=0.5)
axes[2].scatter(ts.index[anomalies_ae_full], ts.values[anomalies_ae_full],
               color='orange', s=50, label='Anomalies', marker='x')
axes[2].set_title('Autoencoder Detection', fontweight='bold')
axes[2].set_ylabel('Production', fontsize=10)
axes[2].legend()
# Statistical
axes[3].plot(ts.index, ts.values, 'b-', linewidth=1.5, alpha=0.5)
axes[3].scatter(ts.index[anomalies_stat], ts.values[anomalies_stat],
               color='green', s=50, label='Anomalies', marker='x')
axes[3].set_title('Statistical Methods Detection', fontweight='bold')
axes[3].set_xlabel('Year', fontsize=11)
axes[3].set_ylabel('Production', fontsize=10)
axes[3].legend()
plt.tight_layout()
plt.savefig('anomaly_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
logger.info("\n" + "="*60)
logger.info("ANOMALY DETECTION COMPARISON")
logger.info("="*60)
logger.info(f"{'Method':<20} {'Anomalies':<15} {'Rate (%)':<15}")
logger.info("-"*60)
logger.info(f"{'Isolation Forest':<20} {anomalies_iso.sum():<15} {anomalies_iso.mean()*100:<15.2f}")
logger.info(f"{'Autoencoder':<20} {anomalies_ae_full.sum():<15} {anomalies_ae_full.mean()*100:<15.2f}")
logger.info(f"{'Statistical':<20} {anomalies_stat.sum():<15} {anomalies_stat.mean()*100:<15.2f}")

# Complete code for reproducibility
# See individual code blocks above for full implementation
