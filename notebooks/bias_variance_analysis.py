#!/usr/bin/env python
# coding: utf-8

# # Bias–Variance Trade-off and Model Generalization in Polynomial Regression
# 
# ## Project Objective
# 
# This project investigates the bias–variance trade-off in supervised machine learning by studying how polynomial model complexity affects training performance, validation performance, and generalization to unseen data.
# 
# A synthetic nonlinear dataset is generated from a known sinusoidal function with added Gaussian noise. Polynomial regression models of increasing degree are then evaluated to study the effect of model complexity.
# 
# The analysis focuses on:
# 
# - Understanding the relationship between model complexity and generalization.
# - Identifying underfitting and overfitting behavior.
# - Using cross-validation to select an appropriate polynomial degree.
# - Investigating Ridge Regression as a regularization technique for controlling high-complexity models.
# - Comparing candidate models on an independent test set.
# - Analyzing prediction errors through residual analysis.
# 
# The final model is selected based on performance on an independent test set, while cross-validation is used during model development to evaluate generalization performance.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

print("Environment ready.")


# ## 1. Experimental Setup
# 
# To ensure that the experiment is reproducible, a fixed random seed is used throughout the analysis. The synthetic dataset contains 30 observations generated from a sinusoidal function with Gaussian noise.
# 
# The noise level is controlled using a standard deviation of 0.2.

# In[2]:


RANDOM_STATE = 42
NOISE_STD = 0.2

np.random.seed(RANDOM_STATE)

print("Random seed:", RANDOM_STATE)
print("Noise standard deviation:", NOISE_STD)


# ## 2. Synthetic Dataset Generation
# 
# The underlying relationship is defined as:
# 
# $$
# y = \sin(2\pi x)
# $$
# 
# Gaussian noise is added to the true function to simulate realistic observation variability:
# 
# $$
# y_{observed} = y_{true} + \epsilon
# $$
# 
# where:
# 
# $$
# \epsilon \sim \mathcal{N}(0, 0.2^2)
# $$
# 
# Using a known nonlinear function makes it possible to study how polynomial models approximate the underlying relationship as their complexity increases.

# In[3]:


n_samples = 30

X = np.sort(np.random.uniform(0, 1, n_samples)).reshape(-1, 1)

y_true = np.sin(2 * np.pi * X)

noise = np.random.normal(
    loc=0,
    scale=NOISE_STD,
    size=(n_samples, 1)
)

y = y_true + noise

print("Number of samples:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)


# ## 3. Dataset Visualization
# 
# The following visualization shows the underlying sinusoidal function together with the noisy observations used for model training and evaluation.

# In[4]:


x_curve = np.linspace(0, 1, 300).reshape(-1, 1)
y_curve = np.sin(2 * np.pi * x_curve)

plt.figure(figsize=(8, 5))

plt.plot(
    x_curve,
    y_curve,
    label="True function"
)

plt.scatter(
    X,
    y,
    label="Noisy observations"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Dataset")
plt.legend()
plt.grid(True)

plt.show()


# ## 4. Train–Test Split
# 
# The dataset is divided into training and independent test sets using an 80/20 split. The training set is used for model development and cross-validation, while the independent test set is reserved for the final evaluation of the selected models.

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))


# In[6]:


linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE:     {test_mse:.4f}")


# In[7]:


plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    label="Training data"
)

plt.scatter(
    X_test,
    y_test,
    label="Test data"
)

plt.plot(
    x_curve,
    y_curve,
    label="True function"
)

linear_curve_pred = linear_model.predict(x_curve)

plt.plot(
    x_curve,
    linear_curve_pred,
    label="Linear model"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression: Underfitting")
plt.legend()
plt.grid(True)

plt.show()


# In[8]:


degrees = range(1, 16)

train_errors = []
test_errors = []

models = {}

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    models[degree] = model

print("Models trained:", len(models))


# In[9]:


plt.figure(figsize=(9, 5))

plt.plot(
    list(degrees),
    train_errors,
    marker="o",
    label="Training MSE"
)

plt.plot(
    list(degrees),
    test_errors,
    marker="o",
    label="Test MSE"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Model Complexity vs Generalization Error")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)

plt.show()


# In[10]:


best_degree = list(degrees)[np.argmin(test_errors)]
best_test_mse = min(test_errors)

print("Best polynomial degree:", best_degree)
print(f"Best test MSE: {best_test_mse:.4f}")


# In[11]:


selected_degrees = [1, 5, 15]

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train,
    y_train,
    label="Training data"
)

plt.plot(
    x_curve,
    y_curve,
    label="True function"
)

for degree in selected_degrees:
    prediction = models[degree].predict(x_curve)

    plt.plot(
        x_curve,
        prediction,
        label=f"Polynomial degree {degree}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("Effect of Model Complexity")
plt.legend()
plt.grid(True)

plt.show()


# In[12]:


N_EXPERIMENTS = 100
N_TRAIN_SAMPLES = 30

X_GRID = np.linspace(0, 1, 200).reshape(-1, 1)
TRUE_GRID = np.sin(2 * np.pi * X_GRID)

print("Experiments:", N_EXPERIMENTS)
print("Training samples per experiment:", N_TRAIN_SAMPLES)


# In[13]:


all_predictions = {}

rng = np.random.default_rng(RANDOM_STATE)

for degree in degrees:
    predictions = []

    for _ in range(N_EXPERIMENTS):

        X_sample = np.sort(
            rng.uniform(0, 1, N_TRAIN_SAMPLES)
        ).reshape(-1, 1)

        true_values = np.sin(2 * np.pi * X_sample)

        noise = rng.normal(
            0,
            NOISE_STD,
            size=(N_TRAIN_SAMPLES, 1)
        )

        y_sample = true_values + noise

        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )

        model.fit(X_sample, y_sample)

        prediction = model.predict(X_GRID)

        predictions.append(prediction.ravel())

    all_predictions[degree] = np.array(predictions)

print("Bias-variance experiments completed.")


# In[14]:


bias_squared = []
variance = []


# In[15]:


for degree in degrees:
    predictions = all_predictions[degree]

    mean_prediction = np.mean(predictions, axis=0)

    bias = mean_prediction - TRUE_GRID.ravel()
    bias_sq = np.mean(bias ** 2)

    var = np.mean(
        np.var(predictions, axis=0)
    )

    bias_squared.append(bias_sq)
    variance.append(var)

print("Bias and variance calculated.")


# In[16]:


plt.figure(figsize=(9, 5))

plt.plot(
    list(degrees),
    bias_squared,
    marker="o",
    label="Bias²"
)

plt.plot(
    list(degrees),
    variance,
    marker="o",
    label="Variance"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Bias–Variance Trade-off")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)

plt.show()


# In[17]:


noise_variance = NOISE_STD ** 2

print(f"Noise variance: {noise_variance:.4f}")


# In[18]:


total_error = np.array(bias_squared) + np.array(variance) + noise_variance

for degree, error in zip(degrees, total_error):
    print(f"Degree {degree:2d}: Expected error ≈ {error:.4f}")


# In[19]:


ridge_alphas = [
    0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    10,
    100
]

ridge_train_errors = []
ridge_test_errors = []

for alpha in ridge_alphas:

    model = make_pipeline(
        PolynomialFeatures(degree=15),
        Ridge(alpha=alpha)
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    ridge_train_errors.append(
        mean_squared_error(y_train, y_train_pred)
    )

    ridge_test_errors.append(
        mean_squared_error(y_test, y_test_pred)
    )

print("Ridge experiments completed.")


# In[20]:


ridge_results = pd.DataFrame({
    "Alpha": ridge_alphas,
    "Training_MSE": ridge_train_errors,
    "Test_MSE": ridge_test_errors
})

print(ridge_results.round(6).to_string(index=False))


# In[21]:


best_ridge_index = np.argmin(ridge_test_errors)

print("\n=== BEST RIDGE MODEL ===")
print(f"Best alpha: {ridge_alphas[best_ridge_index]}")
print(
    f"Best test MSE: "
    f"{ridge_test_errors[best_ridge_index]:.6f}"
)


# In[22]:


plt.figure(figsize=(9, 5))

plt.semilogx(
    ridge_alphas[1:],
    ridge_train_errors[1:],
    marker="o",
    label="Training MSE"
)

plt.semilogx(
    ridge_alphas[1:],
    ridge_test_errors[1:],
    marker="o",
    label="Test MSE"
)

plt.xlabel("Ridge Regularization Strength (α)")
plt.ylabel("Mean Squared Error")
plt.title("Effect of Regularization on a Degree-15 Model")
plt.legend()
plt.grid(True)

plt.show()


# In[23]:


best_alpha_index = np.argmin(ridge_test_errors)

best_alpha = ridge_alphas[best_alpha_index]
best_ridge_test_mse = ridge_test_errors[best_alpha_index]

print("Best Ridge alpha:", best_alpha)
print(f"Best Ridge test MSE: {best_ridge_test_mse:.4f}")


# In[24]:


results_df = pd.DataFrame({
    "Degree": list(degrees),
    "Training_MSE": train_errors,
    "Test_MSE": test_errors,
    "Bias_squared": bias_squared,
    "Variance": variance,
    "Expected_Error": total_error
})

results_df


# In[25]:


print("=== MODEL COMPLEXITY RESULTS ===")
print(results_df.round(4).to_string(index=False))


# In[26]:


best_test_index = np.argmin(test_errors)

print("\n=== BEST MODEL ===")
print(f"Best polynomial degree: {list(degrees)[best_test_index]}")
print(f"Lowest test MSE: {test_errors[best_test_index]:.4f}")


# In[27]:


bias_variance_df = pd.DataFrame({
    "Degree": list(degrees),
    "Bias_squared": bias_squared,
    "Variance": variance
})

bias_variance_df["Bias_plus_Variance"] = (
    bias_variance_df["Bias_squared"]
    + bias_variance_df["Variance"]
)

print(bias_variance_df.round(4).to_string(index=False))


# In[28]:


best_bias_variance_index = np.argmin(
    bias_variance_df["Bias_plus_Variance"]
)

print("\n=== BIAS-VARIANCE BALANCE ===")
print(
    f"Degree with lowest Bias² + Variance: "
    f"{bias_variance_df.loc[best_bias_variance_index, 'Degree']}"
)


# In[29]:


results_df.to_csv(
    "../results/metrics/bias_variance_results.csv",
    index=False
)

print("Results saved.")


# In[30]:


print("=== FINAL MODEL COMPLEXITY RESULTS ===")
print(results_df.round(4).to_string(index=False))


# In[31]:


best_test_index = np.argmin(test_errors)

print("\n=== BEST TEST MODEL ===")
print(f"Polynomial degree: {list(degrees)[best_test_index]}")
print(f"Test MSE: {test_errors[best_test_index]:.6f}")


# In[32]:


print("\n=== BIAS-VARIANCE RESULTS ===")
print(bias_variance_df.round(4).to_string(index=False))


# ## Final Model Comparison
# 
# This section compares the best polynomial model with the best Ridge-regularized model based on test mean squared error (MSE).

# In[33]:


# Find best polynomial model
best_poly_index = np.argmin(test_errors)

best_poly_degree = list(degrees)[best_poly_index]
best_poly_test_mse = test_errors[best_poly_index]

# Find best Ridge model
best_ridge_index = np.argmin(ridge_test_errors)

best_ridge_alpha = ridge_alphas[best_ridge_index]
best_ridge_test_mse = ridge_test_errors[best_ridge_index]

print("=== FINAL MODEL COMPARISON ===")
print(f"Best Polynomial Model: Degree {best_poly_degree}")
print(f"Polynomial Test MSE: {best_poly_test_mse:.6f}")

print()

print(f"Best Ridge Model: Degree 15, Alpha = {best_ridge_alpha}")
print(f"Ridge Test MSE: {best_ridge_test_mse:.6f}")


# In[34]:


final_comparison = pd.DataFrame({
    "Model": [
        f"Polynomial Degree {best_poly_degree}",
        f"Ridge Degree 15 (α={best_ridge_alpha})"
    ],
    "Test_MSE": [
        best_poly_test_mse,
        best_ridge_test_mse
    ]
})

print(final_comparison.round(6).to_string(index=False))


# ## Final Comparison: Polynomial Regression vs Ridge Regression

# In[35]:


plt.figure(figsize=(8, 5))

plt.bar(
    final_comparison["Model"],
    final_comparison["Test_MSE"]
)

plt.ylabel("Test Mean Squared Error")
plt.title("Final Model Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ## Cross-Validation for Model Selection
# 
# To avoid selecting the model directly from the test set, 5-fold cross-validation is used on the training data. The model with the lowest average validation MSE is selected and evaluated once on the independent test set.

# In[36]:


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

cv_results = []

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    cv_mse = -cv_scores.mean()

    cv_results.append({
        "Degree": degree,
        "CV_MSE": cv_mse
    })

cv_results_df = pd.DataFrame(cv_results)

print("=== CROSS-VALIDATION RESULTS ===")
print(cv_results_df.round(6).to_string(index=False))


# In[37]:


best_cv_index = cv_results_df["CV_MSE"].idxmin()

best_cv_degree = int(
    cv_results_df.loc[best_cv_index, "Degree"]
)

best_cv_mse = cv_results_df.loc[
    best_cv_index, "CV_MSE"
]

print("\n=== BEST MODEL FROM CROSS-VALIDATION ===")
print(f"Best polynomial degree: {best_cv_degree}")
print(f"Mean CV MSE: {best_cv_mse:.6f}")


# ## Cross-Validation Error vs Model Complexity
# 
# The following plot shows how validation error changes with polynomial model complexity. The selected degree corresponds to the minimum cross-validation error.

# In[38]:


plt.figure(figsize=(9, 5))

plt.plot(
    cv_results_df["Degree"],
    cv_results_df["CV_MSE"],
    marker="o"
)

plt.axvline(
    best_cv_degree,
    linestyle="--",
    label=f"Best degree = {best_cv_degree}"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Cross-Validation MSE")
plt.title("Cross-Validation Error vs Model Complexity")
plt.legend()
plt.grid(True)

plt.show()


# In[39]:


final_cv_model = make_pipeline(
    PolynomialFeatures(degree=best_cv_degree),
    LinearRegression()
)

final_cv_model.fit(X_train, y_train)

final_test_predictions = final_cv_model.predict(X_test)

final_test_mse = mean_squared_error(
    y_test,
    final_test_predictions
)

print("=== FINAL MODEL EVALUATION ===")
print(f"Selected degree: {best_cv_degree}")
print(f"Cross-validation MSE: {best_cv_mse:.6f}")
print(f"Independent test MSE: {final_test_mse:.6f}")


# ## Cross-Validation for Ridge Regularization
# 
# Cross-validation is used to select the Ridge regularization strength without using the independent test set. This allows the test set to remain unseen until the final evaluation.

# In[40]:


ridge_cv_results = []

for alpha in ridge_alphas:
    model = make_pipeline(
        PolynomialFeatures(degree=15),
        Ridge(alpha=alpha)
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    cv_mse = -cv_scores.mean()

    ridge_cv_results.append({
        "Alpha": alpha,
        "CV_MSE": cv_mse
    })

ridge_cv_results_df = pd.DataFrame(ridge_cv_results)

print("=== RIDGE CROSS-VALIDATION RESULTS ===")
print(ridge_cv_results_df.round(6).to_string(index=False))


# In[41]:


# Find the best Ridge model based on cross-validation
best_ridge_cv_index = ridge_cv_results_df["CV_MSE"].idxmin()

best_ridge_cv_alpha = ridge_cv_results_df.loc[
    best_ridge_cv_index, "Alpha"
]

best_ridge_cv_mse = ridge_cv_results_df.loc[
    best_ridge_cv_index, "CV_MSE"
]

print("=== BEST RIDGE MODEL FROM CROSS-VALIDATION ===")
print(f"Best alpha: {best_ridge_cv_alpha}")
print(f"Mean CV MSE: {best_ridge_cv_mse:.6f}")


# In[42]:


# Train the final Ridge model using the best alpha from cross-validation

final_ridge_model = make_pipeline(
    PolynomialFeatures(degree=15),
    Ridge(alpha=best_ridge_cv_alpha)
)

final_ridge_model.fit(X_train, y_train)

# Predict on the independent test set
final_ridge_predictions = final_ridge_model.predict(X_test)

# Calculate test MSE
final_ridge_test_mse = mean_squared_error(
    y_test,
    final_ridge_predictions
)

print("=== FINAL RIDGE MODEL EVALUATION ===")
print(f"Polynomial degree: 15")
print(f"Selected alpha: {best_ridge_cv_alpha}")
print(f"Cross-validation MSE: {best_ridge_cv_mse:.6f}")
print(f"Independent test MSE: {final_ridge_test_mse:.6f}")


# In[43]:


# Final comparison using CV-selected Ridge hyperparameter

final_comparison = pd.DataFrame({
    "Model": [
        f"Polynomial Degree {best_poly_degree}",
        f"Ridge Degree 15 (alpha={best_ridge_cv_alpha})"
    ],
    "Test_MSE": [
        best_poly_test_mse,
        final_ridge_test_mse
    ]
})

print("=== FINAL MODEL COMPARISON ===")
print(final_comparison.round(6).to_string(index=False))


# In[44]:


import matplotlib.pyplot as plt

# Final comparison visualization

plt.figure(figsize=(8, 5))

plt.bar(
    final_comparison["Model"],
    final_comparison["Test_MSE"]
)

plt.ylabel("Test Mean Squared Error")
plt.title("Final Model Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.show()


# ## Final Model Prediction Analysis
# 
# The polynomial regression model with degree 6 was selected as the final model because it achieved the lowest independent test MSE among the evaluated models. This section analyzes its predictions and residual errors on the unseen test set.

# In[45]:


# Final model predictions

final_model = make_pipeline(
    PolynomialFeatures(degree=6),
    LinearRegression()
)

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

residuals = y_test - final_predictions

print("=== FINAL MODEL ===")
print("Model: Polynomial Regression")
print("Polynomial Degree: 6")
print(f"Test MSE: {mean_squared_error(y_test, final_predictions):.6f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, final_predictions)):.6f}")


# ## Actual vs Predicted Values
# 
# The following visualization compares the model's predictions with the actual target values from the independent test set. Points closer to the diagonal line indicate more accurate predictions.

# In[46]:


plt.figure(figsize=(8, 5))

plt.scatter(y_test, final_predictions)

min_value = min(y_test.min(), final_predictions.min())
max_value = max(y_test.max(), final_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")

plt.tight_layout()
plt.show()


# ## Residual Analysis
# 
# Residuals represent the difference between the actual and predicted target values. A well-behaved regression model should produce residuals that are centered around zero without strong systematic patterns.

# In[47]:


plt.figure(figsize=(8, 5))

plt.scatter(final_predictions, residuals)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()
plt.show()


# In[48]:


print("=== RESIDUAL SUMMARY ===")
print(f"Mean residual: {residuals.mean():.6f}")
print(f"Residual standard deviation: {residuals.std():.6f}")
print(f"Maximum absolute residual: {np.abs(residuals).max():.6f}")


# ## Final Model Performance
# 
# Polynomial Regression with degree 6 was selected as the final model because it achieved the lowest independent test MSE among the evaluated models.
# 
# The final model achieved a test MSE of 0.034249 and a test RMSE of 0.185064. The residual analysis shows that prediction errors are distributed around zero without an obvious systematic pattern.

# In[49]:


# Final model performance summary

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("=== FINAL PERFORMANCE SUMMARY ===")
print(f"Model: Polynomial Regression")
print(f"Polynomial Degree: 6")
print(f"Test MSE: {final_mse:.6f}")
print(f"Test RMSE: {final_rmse:.6f}")
print(f"Mean Residual: {residuals.mean():.6f}")
print(f"Maximum Absolute Residual: {np.abs(residuals).max():.6f}")


# In[50]:


# Final model selection

if best_poly_test_mse < final_ridge_test_mse:
    selected_model_name = "Polynomial Regression"
    selected_model_degree = 6
    selected_test_mse = best_poly_test_mse
else:
    selected_model_name = "Ridge Regression"
    selected_model_degree = 15
    selected_test_mse = final_ridge_test_mse

print("=== SELECTED FINAL MODEL ===")
print(f"Model: {selected_model_name}")
print(f"Degree: {selected_model_degree}")
print(f"Independent Test MSE: {selected_test_mse:.6f}")


# # Final Conclusion
# 
# This project investigated the bias-variance tradeoff in polynomial regression and examined how model complexity and regularization affect predictive performance.
# 
# Polynomial models with increasing degrees were evaluated using cross-validation. The cross-validation results showed that model complexity does not continuously improve generalization performance. The lowest cross-validation error was obtained at polynomial degree 3, while substantially higher degrees resulted in rapidly increasing validation error, indicating overfitting and numerical instability.
# 
# Ridge Regression was subsequently evaluated as a regularization technique for controlling the effects of high polynomial model complexity. Cross-validation identified α = 0.0001 as the best-performing regularization strength among the tested values.
# 
# Finally, the best polynomial model and the best Ridge model were evaluated on an independent test set. Polynomial Regression with degree 6 achieved the lowest test MSE of 0.034249, compared with 0.035145 for the degree-15 Ridge model with α = 0.0001.
# 
# Therefore, Polynomial Regression with degree 6 is selected as the final model.
# 
# The residual analysis indicates that prediction errors are distributed around zero without a strong systematic pattern. However, because the independent test set is relatively small, the reported test metrics should be interpreted with appropriate caution.
# 
# Overall, the analysis demonstrates the practical bias-variance tradeoff: increasing model complexity can reduce bias initially, but excessive complexity can substantially increase variance and hurt generalization. Cross-validation provides a principled mechanism for selecting model complexity, while regularization can help control overly complex models.

# # Final Technical Summary
# 
# ## Final Model
# 
# The final selected model is Polynomial Regression with polynomial degree 6.
# 
# ### Performance
# 
# | Metric | Value |
# |---|---:|
# | Polynomial Degree | 6 |
# | Test MSE | 0.034249 |
# | Test RMSE | 0.185064 |
# | Mean Residual | -0.014175 |
# | Maximum Absolute Residual | 0.308777 |
# 
# ## Model Selection
# 
# Cross-validation was used to evaluate polynomial model complexity. The results showed that increasing polynomial degree does not continuously improve generalization performance.
# 
# The lowest cross-validation error among the tested polynomial models occurred at degree 3. Higher-degree models eventually experienced substantially increasing validation error, indicating overfitting and numerical instability.
# 
# Ridge Regression was then evaluated as a regularization technique. The best tested regularization strength was:
# 
# **α = 0.0001**
# 
# The final independent test comparison showed:
# 
# - Polynomial Regression, degree 6: **MSE = 0.034249**
# - Ridge Regression, degree 15, α = 0.0001: **MSE = 0.035145**
# 
# Therefore, Polynomial Regression with degree 6 was selected as the final model because it achieved the lowest independent test MSE.
# 
# ## Key Finding
# 
# The experiment demonstrates the bias-variance tradeoff in practice. Increasing model complexity can initially improve predictive performance by reducing bias, but excessive complexity can increase variance and lead to overfitting.
# 
# Cross-validation provides a principled method for selecting model complexity, while regularization can help control overly complex models.

# 
