import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# ==================================
# SETTINGS
# ==================================

N_SAMPLES = 15
MAX_DEGREE = 80
N_RUNS = 100

all_train_errors = np.zeros((N_RUNS, MAX_DEGREE))
all_test_errors = np.zeros((N_RUNS, MAX_DEGREE))

# ==================================
# EXPERIMENT
# ==================================

for run in range(N_RUNS):

    np.random.seed(run)

    X = np.random.uniform(-1, 1, N_SAMPLES)

    Y = (
        np.sin(5 * X)
        + np.random.normal(0, 0.1, N_SAMPLES)
    )

    X = X.reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=run
    )

    for degree in range(1, MAX_DEGREE + 1):

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=True
        )

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Minimum norm solution
        beta = np.linalg.pinv(X_train_poly) @ Y_train

        train_pred = X_train_poly @ beta
        test_pred = X_test_poly @ beta

        all_train_errors[run, degree - 1] = mean_squared_error(
            Y_train,
            train_pred
        )

        all_test_errors[run, degree - 1] = mean_squared_error(
            Y_test,
            test_pred
        )

# ==================================
# STATISTICS
# ==================================

degrees = np.arange(1, MAX_DEGREE + 1)

median_train = np.median(
    all_train_errors,
    axis=0
)

median_test = np.median(
    all_test_errors,
    axis=0
)

q25 = np.percentile(
    all_test_errors,
    25,
    axis=0
)

q75 = np.percentile(
    all_test_errors,
    75,
    axis=0
)

# ==================================
# SMOOTHING
# ==================================

window = 7

smooth_test = np.convolve(
    median_test,
    np.ones(window) / window,
    mode="same"
)

# ==================================
# SAVE RESULTS
# ==================================

results = pd.DataFrame({
    "Degree": degrees,
    "Median Train Error": median_train,
    "Median Test Error": median_test,
    "Q25": q25,
    "Q75": q75
})

results.to_csv(
    "results.csv",
    index=False
)

# ==================================
# INTERPOLATION THRESHOLD
# ==================================

n_train = int(N_SAMPLES * 0.8)
threshold = n_train - 1

# ==================================
# PLOT
# ==================================

plt.figure(figsize=(15, 8))

plt.plot(
    degrees,
    median_train,
    linewidth=2,
    label="Median Train Error"
)

plt.plot(
    degrees,
    smooth_test,
    linewidth=3,
    label="Median Test Error"
)

plt.fill_between(
    degrees,
    q25,
    q75,
    alpha=0.25,
    label="25%-75% Range"
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label=f"Interpolation Threshold (~{threshold})"
)

plt.yscale("log")

plt.xlim(1, 50)

plt.xlabel(
    "Polynomial Degree",
    fontsize=12
)

plt.ylabel(
    "Mean Squared Error (Log Scale)",
    fontsize=12
)

plt.title(
    "Double Descent in Polynomial Regression",
    fontsize=16
)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()