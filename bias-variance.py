import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# Generate raw data in [0, 6]
X_raw = np.sort(np.random.rand(20) * 6).reshape(-1, 1)
y_train = np.sin(X_raw.ravel()) + np.random.normal(0, 0.1, 20)

# Scale inputs to [-1, 1] to prevent numerical instability
X_train = (X_raw - 3) / 3

X_test_raw = np.linspace(0, 6, 100).reshape(-1, 1)
X_test = (X_test_raw - 3) / 3
y_test = np.sin(X_test_raw.ravel())

for degree in [1, 3, 12]:
    poly = PolynomialFeatures(degree=degree)
    X_tr_poly = poly.fit_transform(X_train)
    X_te_poly = poly.transform(X_test)

    model = LinearRegression().fit(X_tr_poly, y_train)

    tr_mse = mean_squared_error(y_train, model.predict(X_tr_poly))
    te_mse = mean_squared_error(y_test, model.predict(X_te_poly))

    print(
        f"Degree {degree:2d} | Train MSE: {tr_mse:.4f} | Test MSE: {te_mse:.4f}"
    )