# Bias–Variance Trade-off and Model Generalization in Machine Learning

A practical machine learning study investigating how model complexity affects bias, variance, overfitting, and generalization performance.

This project uses polynomial regression on a synthetic nonlinear dataset and experimentally investigates model complexity, the bias–variance trade-off, cross-validation, and Ridge regularization.

---

## Project Overview

A fundamental challenge in machine learning is developing models that perform well not only on training data but also on unseen data.

A model that is too simple may fail to capture the underlying relationship in the data, resulting in **underfitting and high bias**. In contrast, an excessively complex model may fit the training data too closely and become sensitive to noise, resulting in **overfitting and high variance**.

This project investigates these concepts experimentally by varying the complexity of polynomial regression models and evaluating their performance on unseen data.

The project also investigates whether **Ridge regularization** can reduce the effects of excessive model complexity.

---

## Research Question

**How does increasing model complexity affect bias, variance, overfitting, and the ability of a machine learning model to generalize to unseen data?**

---

## Objectives

The main objectives of this project are:

1. Investigate how polynomial model complexity affects predictive performance.
2. Demonstrate underfitting and overfitting experimentally.
3. Analyze the bias–variance trade-off.
4. Compare training and test errors across different polynomial degrees.
5. Use cross-validation to evaluate model complexity.
6. Investigate Ridge regression as a regularization technique.
7. Select an appropriate regularization strength using cross-validation.
8. Evaluate final models on an independent test set.
9. Analyze prediction errors using residual analysis.
10. Develop a practical understanding of model generalization.

---

## Experimental Methodology

The overall experimental workflow is:

```text
Synthetic Nonlinear Dataset
          |
          v
   Train / Test Split
          |
          v
   Polynomial Regression
          |
          v
 Evaluate Multiple Degrees
       1 → 15
          |
          v
 Bias–Variance Analysis
          |
          v
   Cross-Validation
          |
          v
  Ridge Regularization
          |
          v
 Hyperparameter Selection
          |
          v
 Independent Test Evaluation
          |
          v
   Residual Analysis
          |
          v
    Final Conclusions
```

---

## Dataset

A synthetic nonlinear dataset is generated using a sinusoidal relationship with Gaussian noise.

The underlying relationship is approximately:

```text
y = sin(2πx) + noise
```

A synthetic dataset was chosen because it provides direct control over the underlying relationship and makes the effects of model complexity easier to visualize and interpret.

The dataset is divided into training and independent test sets.

The training data is used for model development and cross-validation, while the independent test set is reserved for final performance evaluation.

---

## Polynomial Regression

Polynomial regression is used as the primary model because its complexity can be controlled directly through the polynomial degree.

The experiment evaluates polynomial models ranging from:

```text
Degree 1 → Degree 15
```

Increasing the polynomial degree increases the flexibility of the model.

### Expected behavior

At low degrees:

- The model is relatively simple.
- It may fail to capture the nonlinear relationship.
- Bias is relatively high.
- The model may underfit.

At moderate degrees:

- The model can capture more of the underlying relationship.
- Bias decreases.
- Generalization performance can improve.

At very high degrees:

- The model becomes highly flexible.
- Variance increases.
- The model becomes more sensitive to the training data.
- Test or validation error can increase substantially.
- Overfitting and numerical instability may occur.

---

## Bias–Variance Trade-off

The project estimates prediction bias and variance by repeatedly training models on different sampled training datasets.

The general relationship can be represented as:

```text
Low Model Complexity
        |
        v
   High Bias
 Low Variance
        |
        v
    Underfitting
        |
        | Increase Complexity
        v
 Balanced Region
        |
        | Further Increase Complexity
        v
    Low Bias
   High Variance
        |
        v
    Overfitting
```

The experiment demonstrates that increasing model complexity does not continuously improve generalization.

A more flexible model can reduce bias, but excessive flexibility can substantially increase variance.

---

## Model Complexity Experiment

Polynomial models with degrees from 1 to 15 were evaluated.

The estimated expected errors were:

| Degree | Expected Error |
|---:|---:|
| 1 | 0.2623 |
| 2 | 0.3064 |
| 3 | 0.0538 |
| 4 | 0.0657 |
| 5 | 0.0820 |
| 6 | 0.0709 |
| 7 | 0.1024 |
| 8 | 0.2868 |
| 9 | 0.7389 |
| 10 | 0.6888 |
| 11 | 0.5585 |
| 12 | 5.1010 |
| 13 | 8.6001 |
| 14 | 64.1531 |
| 15 | 64.5149 |

The results show that very high polynomial degrees can lead to a dramatic increase in expected prediction error.

This behavior is consistent with increasing variance and overfitting as model complexity becomes excessive.

---

## Cross-Validation

Five-fold cross-validation is used to evaluate polynomial model complexity using the training data.

Cross-validation provides a more reliable estimate of model performance during model selection than relying on a single validation split.

The cross-validation experiment identified:

```text
Best observed polynomial degree: 3
Mean CV MSE: 0.035505
```

The independent test set was kept separate from this cross-validation process.

---

## Ridge Regularization

High-degree polynomial models can produce very large coefficients and become sensitive to small changes in the training data.

Ridge regression introduces L2 regularization to penalize large model coefficients.

The Ridge experiment uses a high-complexity polynomial model:

```text
Polynomial Degree: 15
```

Several regularization strengths were evaluated using five-fold cross-validation.

The tested values of alpha were:

```text
0.000001
0.000010
0.000100
0.001000
0.010000
0.100000
1.000000
10.000000
100.000000
```

The best observed regularization strength was:

```text
α = 0.0001
```

with:

```text
Mean CV MSE = 0.038369
```

This experiment demonstrates how regularization can reduce the effects of excessive model complexity.

---

## Ridge Cross-Validation Results

| Alpha | Mean CV MSE |
|---:|---:|
| 0.000001 | 0.053010 |
| 0.000010 | 0.043000 |
| 0.000100 | **0.038369** |
| 0.001000 | 0.047224 |
| 0.010000 | 0.109255 |
| 0.100000 | 0.167425 |
| 1.000000 | 0.242402 |
| 10.000000 | 0.406811 |
| 100.000000 | 0.506922 |

The results show that regularization strength also has a trade-off.

Very small values provide limited regularization, while excessively large values can introduce too much bias and increase prediction error.

---

## Final Model Comparison

The final independent test-set comparison produced the following results:

| Model | Test MSE |
|---|---:|
| Polynomial Regression — Degree 6 | **0.034249** |
| Ridge Regression — Degree 15, α = 0.0001 | 0.035145 |

The degree-6 polynomial model achieved the lower observed test MSE.

The difference between the two models is:

```text
0.035145 - 0.034249 = 0.000896
```

Therefore, based on the observed independent test performance, Polynomial Regression with degree 6 performed slightly better than the evaluated Ridge model.

However, the difference is relatively small and should not be interpreted as evidence that degree 6 will always outperform Ridge regression on other datasets.

---

## Final Model

Based on the final independent test comparison, the selected model in this experiment is:

```text
Model: Polynomial Regression
Polynomial Degree: 6
```

Performance:

| Metric | Value |
|---|---:|
| Test MSE | **0.034249** |
| Test RMSE | **0.185064** |
| Mean Residual | **-0.014175** |
| Residual Standard Deviation | **0.184520** |
| Maximum Absolute Residual | **0.308777** |

---

## Actual vs Predicted Analysis

The final model's predictions are compared against the actual target values from the independent test set.

A prediction lying close to the diagonal reference line indicates that the predicted value is close to the actual value.

The visualization is included in the Jupyter notebook:

```text
notebooks/bias_variance_analysis.ipynb
```

The model predictions generally follow the expected relationship, although some prediction errors are present, particularly around extreme values.

---

## Residual Analysis

Residuals are calculated as:

```text
Residual = Actual Value - Predicted Value
```

For the final model:

```text
Mean Residual              = -0.014175
Residual Standard Deviation = 0.184520
Maximum Absolute Residual   = 0.308777
```

The residuals are distributed around zero without an obvious strong systematic pattern.

The mean residual being relatively close to zero suggests that there is no substantial overall directional error in the predictions.

However, the independent test set is relatively small, so the residual statistics should be interpreted with appropriate caution.

---

## Key Findings

### 1. Increasing complexity does not always improve generalization

Increasing polynomial degree makes the model more flexible, but excessive complexity can cause the model to fit noise rather than the underlying relationship.

### 2. Training performance alone is insufficient

A highly complex model can achieve very low training error while performing poorly on unseen data.

Therefore, model evaluation must consider generalization performance.

### 3. Bias and variance move in opposite directions

Increasing model flexibility generally reduces bias but can increase variance.

The optimal model is therefore not necessarily the most complex model.

### 4. Cross-validation is useful for model selection

Five-fold cross-validation provides a systematic way to compare candidate models using the training data.

### 5. Ridge regularization can control model complexity

Ridge regression reduces the influence of large coefficients and can stabilize high-degree polynomial models.

### 6. Regularization strength also requires tuning

Too little regularization may not sufficiently control model variance, while too much regularization can increase bias.

### 7. Simpler models can generalize well

The final experiment demonstrates that a moderately complex polynomial model can perform competitively with a highly complex regularized model.

---

## Limitations

This project has several limitations:

- The dataset is synthetic rather than real-world.
- The independent test set is relatively small.
- Results depend on the generated dataset and train/test split.
- Very high-degree polynomial regression can suffer from numerical instability.
- The experiments focus primarily on polynomial regression and Ridge regression.
- A single dataset cannot establish universally optimal hyperparameters or model complexity.

Therefore, the numerical results should be interpreted as results of this specific experimental setup rather than universal conclusions.

The primary purpose of the project is to demonstrate the concepts of bias, variance, model complexity, overfitting, generalization, cross-validation, and regularization through practical experimentation.

---

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
bias-variance-tradeoff/
│
├── notebooks/
│   └── bias_variance_analysis.ipynb
│
├── results/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd bias-variance-tradeoff
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/bias_variance_analysis.ipynb
```

Run all cells from the beginning.

---

## Reproducibility

The notebook is designed to be executed sequentially from a fresh Python environment.

The project uses a dedicated virtual environment for development, while the `.gitignore` file prevents local environment files from being committed to the repository.

To reproduce the experiment:

1. Clone the repository.
2. Create a new virtual environment.
3. Install the dependencies from `requirements.txt`.
4. Open the notebook.
5. Restart the kernel.
6. Run all cells from the beginning.

---

## Learning Outcomes

Through this project, the following machine learning concepts were investigated practically:

- Bias
- Variance
- Model generalization
- Underfitting
- Overfitting
- Polynomial regression
- Model complexity
- Training error
- Test error
- Mean Squared Error
- Root Mean Squared Error
- Cross-validation
- Hyperparameter selection
- Ridge regression
- L2 regularization
- Residual analysis
- Model evaluation

---

## Conclusion

This project demonstrates the practical importance of balancing model complexity and generalization performance.

Polynomial regression provided a useful framework for observing the bias–variance trade-off. Low-complexity models can underfit the data because of high bias, while excessively complex models can become highly sensitive to the training data and suffer from high variance.

The model complexity experiment showed that very high polynomial degrees can produce rapidly increasing prediction error, demonstrating the practical consequences of overfitting and numerical instability.

Ridge regression was then investigated as a regularization technique for a high-degree polynomial model. Cross-validation identified an alpha value of 0.0001 as the best-performing regularization strength among the tested values.

The final independent test comparison showed that Polynomial Regression with degree 6 achieved a test MSE of 0.034249, compared with 0.035145 for the evaluated degree-15 Ridge model.

Overall, the experiment demonstrates that successful machine learning is not simply about creating the most flexible model or minimizing training error. The objective is to find a model that achieves an appropriate balance between **bias, variance, complexity, and generalization**.

---

## Author

**Dhruvil Patel**

Master's Student in Software Engineering

This project was developed as a practical study of the bias–variance trade-off and model generalization in machine learning.