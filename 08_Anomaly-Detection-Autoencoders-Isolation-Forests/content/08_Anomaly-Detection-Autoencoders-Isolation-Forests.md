# Time Series Anomaly Detection with Autoencoders and Isolation Forests Anomaly detection identifies unusual patterns in time series. We compare autoencoders, isolation forests, and statistical methods on Oklahoma energy production data, showing how to select thresholds and evaluate performance.

### Time Series Anomaly Detection with Autoencoders and Isolation Forests
Anomaly detection finds the needle in the haystack. Equipment failures, data quality issues, or significant events appear as unusual patterns in time series. But how do you detect them automatically?

We compare three approaches on Oklahoma energy production data: isolation forests for statistical outliers, autoencoders for reconstruction-based detection, and statistical methods for rule-based identification. Each method has strengths for different anomaly types.

### Dataset: Oklahoma Energy Production
We use energy production data that may contain anomalies from equipment failures or data issues.


The series contains **54 annual observations from 1970–2023**, representing Oklahoma energy production totals. This horizon is long enough to include structural shifts and potential data issues, but short enough to manually review flagged anomalies for validation.
### Method 1: Isolation Forest
Isolation Forest identifies outliers by isolating them in feature space.


Isolation Forest is fast and works well for point anomalies.

### Method 2: Autoencoder
Autoencoders detect anomalies through reconstruction error.


Autoencoders excel at detecting complex, contextual anomalies.

### Method 3: Statistical Methods
Statistical methods provide interpretable, rule-based detection.


Statistical methods are interpretable and fast but miss complex patterns.

### Threshold Selection
Selecting the right threshold balances false positives and false negatives.


Threshold selection is critical—too low creates false alarms, too high misses real anomalies. In our experiment, we swept different anomaly score thresholds and visualized how the number of detected anomalies changed; the curve in `threshold_selection.png` makes it clear how aggressive vs. conservative settings affect Isolation Forest and autoencoder results.

### Comparison and Visualization
We compare all methods and visualize detected anomalies.


On the Oklahoma production series, the methods behaved quite differently:

- **Isolation Forest** flagged **6 anomalies**, corresponding to about **11.1%** of the observations.  
- The **autoencoder** identified **12 anomaly periods**, or roughly **22.2%** of the series, making it the most sensitive method.  
- The **statistical baselines** (Z-score, IQR, moving-average rules) detected **no anomalies** under the chosen thresholds.

The combined plot in `anomaly_comparison.png` overlays all three methods’ flags on the time series, making it easy to see where they agree (strong candidates) versus where only the more sensitive autoencoder fires (likely contextual or borderline anomalies).

### When to Use Each Method
Use Isolation Forest when:
- You need fast detection
- Point anomalies are the concern
- You have labeled normal data
- Interpretability is less important

Use Autoencoders when:
- Anomalies are contextual (depend on history)
- You have complex patterns
- You need to learn normal behavior
- You can tolerate longer training

Use Statistical Methods when:
- You need interpretable rules
- Simple thresholds suffice
- Speed is critical
- Domain experts need to understand detection logic

### Best Practices
- Combine methods Ensemble approaches catch more anomalies
- Tune thresholds carefully Use validation data to select optimal thresholds
- Consider context Some "anomalies" may be valid but unusual events
- Monitor performance Track false positive/negative rates over time
- Update models Retrain as normal behavior evolves

### Conclusion
Anomaly detection requires the right method for your specific problem. Isolation Forest is fast and effective for point anomalies. Autoencoders excel at contextual anomalies. Statistical methods provide interpretable baselines. Combining methods often yields the best results.


