# Linear Regression, Built from Scratch

> A practical machine-learning project exploring how gradient descent learns to predict continuous values—from fitting a line to estimating medical insurance charges.

This project implements linear regression with Python and NumPy rather than relying on a machine-learning framework. It includes a simple regression example, a reusable multiple-linear-regression class, and a Jupyter notebook that applies the model to an insurance dataset.

## Project highlights

- **Learn the fundamentals:** follow how model parameters are updated with gradient descent.
- **Compare two regression settings:** fit one input feature in a simple example and multiple features in the insurance analysis.
- **Inspect model training:** record mean squared error (MSE) and visualize its change across epochs.
- **Evaluate predictions:** use a train/test split and the R² score to assess predictions on held-out insurance data.
- **Explore interactively:** use the notebook to see the data preparation, training, and evaluation workflow in one place.

## How it works

### 1. Simple linear regression

[`LR.py`](./LR.py) loads `data/train.csv`, which contains `x` and `y` columns. It updates the slope and intercept with gradient descent and plots the fitted line alongside the observations. The file also includes a mean squared error function and both loop-based and vectorized gradient-update implementations.

### 2. Multiple linear regression

[`multipleLinearRegression.py`](./multipleLinearRegression.py) defines a reusable `MultipleLinearRegression` class. It learns a coefficient for each input feature and a bias term, then provides methods to:

- fit weights with gradient descent;
- generate predictions;
- plot the training loss; and
- split feature and target arrays into training and test sets.

### 3. Insurance-charge analysis

[`MLR.ipynb`](./MLR.ipynb) uses `data/insurance.csv` to predict `charges` from selected customer attributes: age, sex, BMI, number of children, and smoking status. The notebook encodes sex and smoking status as numeric values, excludes region, standardizes the features, trains the model, and evaluates test predictions with R².

**Reading the metric:** R² is a regression score, not classification accuracy. A higher R² means the predictions explain more of the variation in the test targets; it does not mean the same thing as a percentage of correct classifications.

## Repository layout

```text
.
├── LR.py                         # Simple linear regression example
├── multipleLinearRegression.py   # Reusable multiple-regression model
├── MLR.ipynb                     # Insurance prediction walkthrough
└── data/
    ├── train.csv                 # Simple-regression training data (x, y)
    ├── test.csv                  # Additional simple-regression data
    └── insurance.csv             # Insurance attributes and charges
```

## Get started

Use Python 3. Install the libraries used by the scripts and notebook:

```bash
python -m pip install numpy pandas matplotlib jupyter
```

Run the simple-regression example from the project root:

```bash
python LR.py
```

Open the multiple-regression walkthrough:

```bash
jupyter notebook MLR.ipynb
```

The notebook imports `MultipleLinearRegression` from the project-root Python file, so launch Jupyter from the project directory.

## What this project demonstrates

This project is an educational implementation focused on understanding the mechanics behind linear regression: feature preparation, parameter initialization, gradient updates, prediction, loss tracking, and regression evaluation. It is a learning project rather than a production-ready prediction service.

## References

This project was built and adapted using insights and mathematical foundations from the following resources:

- **Kaggle:** [Multiple Linear Regression From Scratch](https://www.kaggle.com/code/jaypradipshah/multiple-linear-regression-from-scratch) by Jay Pradip Shah
- **Medium (Analytics Vidhya):** [Multiple Linear Regression from scratch using Python](https://medium.com/analytics-vidhya/multiple-linear-regression-from-scratch-using-python-db9368859f)
- **GeeksforGeeks:** [ML | Multiple Linear Regression using Python](https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/)
- **YouTube:** [Linear Regression From Scratch in Python (Mathematical)](https://youtu.be/VmbA0pi2cRQ) by NeuralNine
