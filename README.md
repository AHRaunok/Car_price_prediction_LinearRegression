# 🚗 Ford Car Price Prediction using Linear Regression

A machine learning project that predicts the price of used Ford cars from features like model, year, mileage, fuel type, transmission, tax, MPG, and engine size, using **Linear Regression**.

The notebook covers the full workflow: exploratory data analysis (EDA), feature encoding, scaling, model training, and evaluation. It also **compares two encoding strategies** (One-Hot vs. Label Encoding) to see which works better for a linear model.

## 📊 Dataset

- **Name:** Ford Car Price Prediction
- **Source:** [Kaggle – adhurimquku/ford-car-price-prediction](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction)
- **File used:** `ford.csv`
- **Target variable:** `price`

| Feature | Type | Description |
|---------|------|-------------|
| `model` | Categorical | Ford car model |
| `year` | Numerical | Year of registration |
| `transmission` | Categorical | Manual / Automatic / Semi-Auto |
| `mileage` | Numerical | Distance driven |
| `fuelType` | Categorical | Petrol / Diesel / Hybrid, etc. |
| `tax` | Numerical | Road tax |
| `mpg` | Numerical | Miles per gallon |
| `engineSize` | Numerical | Engine size in litres |

## 🛠️ Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook / Kaggle

## 🔍 Project Workflow

### 1. Data Exploration
- Inspected the data with `head()`, `shape`, `info()`, `describe()`
- Checked for missing values with `isnull().sum()`

### 2. Exploratory Data Analysis (EDA)
- Price distribution (histogram + KDE)
- Correlation heatmap of numerical features
- Price vs. **year**, **mileage**, **engine size** (scatter and box plots)
- Price vs. **transmission**, **fuel type**, and **model** (box plots)

### 3. Preprocessing
Two approaches were tested:

| Approach | Encoding | Scaling |
|----------|----------|---------|
| **A – One-Hot Encoding** | `pd.get_dummies(..., drop_first=True)` on `model`, `transmission`, `fuelType` | `StandardScaler` on `year`, `mileage`, `tax`, `mpg`, `engineSize` |
| **B – Label Encoding** | `LabelEncoder` on `model`, `transmission`, `fuelType` | `StandardScaler` on all features |

### 4. Model Training
- Train/test split: **67% / 33%** (`random_state=42`)
- Model: `sklearn.linear_model.LinearRegression`
- Trained and evaluated separately on both encoded datasets

### 5. Evaluation
- Metric: **R² score** on the test set

## 📈 Results

| Encoding Method | R² Score |
|-----------------|----------|
| One-Hot Encoding | `0.8396626991294073` |
| Label Encoding | `0.7310215557391141` |

> One-hot encoding is generally better suited to linear models, since label encoding imposes an artificial order on categories like `model`.

## 📁 Project Structure

```
Car_price_prediction_LinearRegression/
├── car-price-with-linear.ipynb   # Main notebook (EDA + modeling)
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AHRaunok/Car_price_prediction_LinearRegression.git
cd Car_price_prediction_LinearRegression
```

### 2. Install dependencies
```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter
```

### 3. Download the dataset
Download `ford.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction) and place it in the project folder. Then update the file path in the notebook:
```python
dt = pd.read_csv("ford.csv")
```

### 4. Run the notebook
```bash
jupyter notebook car-price-with-linear.ipynb
```

## 🔮 Future Improvements

- Report MAE and RMSE alongside R²
- Try Ridge, Lasso, Random Forest, and XGBoost
- Handle outliers and apply a log transform to `price`
- Add cross-validation and hyperparameter tuning
- Build a small web app (Streamlit/Flask) for live price predictions

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a branch, and open a pull request.

## 👤 Author

**AHRaunok**
GitHub: [@AHRaunok](https://github.com/AHRaunok)
