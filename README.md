# Real_estate_housing-price_prediction
# California Housing Regression Project README

## Overview
This project aims to predict median house values in California districts using the California Housing dataset. We trained and evaluated several machine learning models—starting from a simple Decision Tree baseline and progressing to robust ensemble models, which we further optimized using hyperparameter tuning.

---

## Model Performance Summary

| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Decision Tree** (Baseline) | `0.6829` | `0.4332` | `0.6446` |
| **Random Forest** | `0.8051` | `0.3275` | `0.5053` |
| **HistGradientBoosting** (Default) | `0.8355` | `0.3096` | `0.4642` |
| **Tuned HistGradientBoosting** (Best) | **`0.8399`** | **`0.3011`** | **`0.4579`** |

*Note: $R^2$ score measures the proportion of variance explained by the model (higher is better). MAE and RMSE measure the average prediction errors (lower is better).* 

---

## Key Findings & Discussion

### 1. Baseline Decision Tree
- **Description**: A single decision tree with `max_depth=10`.
- **Performance**: $R^2 \approx 0.683$, MAE $\approx 0.433$.
- **Discussion**: The baseline model captured basic non-linear trends but suffered from high variance and error. As observed in the Actual vs. Predicted plots, its predictions are highly stratified into discrete horizontal bands, reflecting the hard thresholds of a single decision tree.

### 2. Random Forest
- **Description**: An ensemble of 100 decision trees using bagging.
- **Performance**: $R^2 \approx 0.805$, MAE $\approx 0.328$.
- **Discussion**: Bagging significantly reduced variance and generalized much better by averaging out predictions over independent trees, leading to a substantial performance leap.

### 3. Gradient Boosting (HistGradientBoosting)
- **Description**: Scikit-learn's optimized histogram-based gradient boosting regressor.
- **Performance (Tuned)**: $R^2 \approx 0.840$, MAE $\approx 0.301$.
- **Discussion**: This boosting technique achieved our highest performance. Unlike Random Forest, boosting iteratively builds trees to correct the errors of preceding trees. 
- By executing hyperparameter optimization using **GridSearchCV**, we discovered the optimal configuration: 
  - `learning_rate`: `0.1`
  - `max_leaf_nodes`: `63`
  - `min_samples_leaf`: `30`
- This configuration allowed deeper trees with higher leaf capacity to model complex continuous boundaries smoothly without overfitting.
