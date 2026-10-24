"""Train and visualize a simple linear regression model with gradient descent.

The example reads ``data/train.csv``, whose columns are ``x`` and ``y``, then
fits a line and plots it against the observed data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
data = pd.read_csv(r'data\train.csv').dropna()

x_mean = data.x.mean()
x_std = data.x.std()
y_mean = data.y.mean()
y_std = data.y.std()

x = np.array((data.x - x_mean) / x_std)
y = np.array((data.y - y_mean) / y_std)


# ---------------------------------------------------------------------------
# Regression objective and optimization
# ---------------------------------------------------------------------------
def loss_function(m, b, points):
    """Calculate mean squared error for a line and a two-column DataFrame.

    Args:
        m: Slope of the line.
        b: Intercept of the line.
        points: DataFrame containing the input ``x`` and target ``y`` values.

    Returns:
        Mean squared prediction error across all rows.
    """
    total_loss = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_loss += (y - (m * x + b)) ** 2
    return total_loss / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    """Update the line's slope and intercept using per-row gradients.

    Args:
        m_now: Current slope.
        b_now: Current intercept.
        points: DataFrame containing the input ``x`` and target ``y`` values.
        L: Learning rate used to scale each parameter update.

    Returns:
        The updated slope and intercept.
    """
    m_gradient = 0
    b_gradient = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2 / len(points)) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / len(points)) * (y - (m_now * x + b_now))

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b


def gradient_descent_vectorized(m_now, b_now, points,L):
    """Update the line's parameters using vectorized gradient calculations.

    Args:
        m_now: Current slope.
        b_now: Current intercept.
        points: DataFrame containing the input ``x`` and target ``y`` values.
        L: Learning rate used to scale each parameter update.

    Returns:
        The updated slope and intercept.
    """
    n = len(points)

    y_pred = m_now * points.x + b_now
    error = points.y - y_pred

    m_gradient = -(2 / n) * np.sum(points.x * error)
    b_gradient = -(2 / n) * np.sum(error)

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
m =  1
b = 1

L= 0.00001
epochs = 100

for i in range(epochs):
    m, b = gradient_descent_vectorized(m, b, data, L)

print(f"Final values after {epochs} epochs: m = {m}, b = {b}")


# ---------------------------------------------------------------------------
# Model visualization
# ---------------------------------------------------------------------------
plt.scatter(data.x, data.y, color='blue')
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color='red')
plt.show()
