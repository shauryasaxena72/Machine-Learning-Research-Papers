# Double Descent: Empirical Evidence Against Classical Bias-Variance Theory

## Overview

This project reproduces the **Double Descent** phenomenon, one of the most important discoveries in modern machine learning theory. While classical learning theory suggests that increasing model complexity beyond an optimal point leads to worse generalization, Double Descent demonstrates that test error can decrease again once a model becomes sufficiently overparameterized.

The objective of this project is to experimentally reproduce this behavior using polynomial regression on synthetic data and investigate why highly overparameterized models can generalize surprisingly well.

---

## Research Question

Can increasing model complexity beyond the traditional overfitting region lead to improved generalization performance?

This project explores the limitations of the classical bias-variance tradeoff and demonstrates the emergence of a second descent in test error after the interpolation threshold.

---

## Key Concepts

* Bias-Variance Tradeoff
* Overfitting and Underfitting
* Model Complexity
* Interpolation Threshold
* Overparameterization
* Generalization Error
* Statistical Learning Theory

---

## Experimental Setup

### Dataset

Synthetic dataset generated from:

```math
y = sin(2πx) + ε
```

where:

* x ~ Uniform(0,1)
* ε ~ N(0,0.09)

### Configuration

| Parameter         | Value                 |
| ----------------- | --------------------- |
| Training Samples  | 50                    |
| Test Samples      | 300                   |
| Model Type        | Polynomial Regression |
| Polynomial Degree | 1 - 80                |
| Regularization    | Ridge Regression      |
| Lambda            | 1e-10                 |

### Evaluation Metric

* Mean Squared Error (MSE)

---

## Methodology

1. Generate noisy samples from a sinusoidal function.
2. Fit polynomial regression models of increasing degree.
3. Calculate training and test MSE for each model.
4. Identify the interpolation threshold.
5. Analyze model behavior before and after interpolation.
6. Visualize train and test error curves.

---

## Results

The experiment successfully reproduces the Double Descent phenomenon:

### Classical Regime

* Test error decreases as model complexity increases.
* Models learn meaningful patterns from data.

### Interpolation Threshold

* Test error spikes dramatically.
* Model becomes highly sensitive to noise.
* Generalization performance deteriorates.

### Overparameterized Regime

* Training error remains near zero.
* Test error decreases again.
* Large models generalize better than expected.

These findings challenge the traditional interpretation of the bias-variance tradeoff.

---

## Key Findings

* Classical theory captures only part of model behavior.
* The interpolation threshold is often the most unstable region.
* Highly overparameterized models can outperform models near the classical optimum.
* Modern deep learning systems operate predominantly in the overparameterized regime.

---

## Technologies Used

* Python
* NumPy
* Scikit-Learn
* Matplotlib

---

## Repository Structure

```text
Double-Descent/
├── paper.pdf
├── double_descent.py
├── results.png
├── figures/
└── README.md
```

---

## References

1. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-Off.

2. Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2020). Deep Double Descent.

3. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning.

4. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. Understanding Deep Learning Requires Rethinking Generalization.

---

## Future Work

* Epoch-wise Double Descent
* Sample-wise Double Descent
* Effects of Regularization
* Neural Network Implementations
* Analytical Study of Interpolation Thresholds

---

## Author

Shaurya Saxena

Artificial Intelligence & Machine Learning | Data Science | Quantitative Research
