# Algorithmic Trading Pipeline

A comprehensive machine learning-based algorithmic trading system implementing Ridge Regression with Random Forest meta-labeling for equity portfolio optimization.

## 📁 Directory Structure

```
.
├── Feature Engineering & Data Cleaning.ipynb
├── Model Training & Strategy Formulation.ipynb
├── Backtesting & Performance Analysis.ipynb
├── Statistical Arbitrage Overlay.ipynb
├── archive/
│   └── anonymized_data/
│       ├── Asset_001.csv
│       ├── Asset_002.csv
│       └── ... (100 anonymized equity assets)
└── README.md
```

## 📊 Dataset

- **Assets**: 100 anonymized equity instruments
- **Time Period**: January 25, 2016 - January 16, 2026 (2,513 trading days)
- **Data Format**: Daily OHLCV (Open, High, Low, Close, Volume)
- **File Location**: `archive/anonymized_data/Asset_*.csv`

Each CSV contains:
- `Date`: Trading date
- `Open`, `High`, `Low`, `Close`: Price data
- `Volume`: Trading volume

## 📦 Installation & Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Or, manually:

```bash
pip install pandas numpy matplotlib seaborn joblib scikit-learn scipy statsmodels
```

**requirements.txt** is provided for reproducibility.

---

## 🛠️ Additional Notes
- All notebooks require the above dependencies.
- For statistical tests and advanced analysis, ensure `scipy` and `statsmodels` are installed.
- For visualization, `matplotlib` and `seaborn` are used extensively.

---

## 🔧 Dependencies

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

**Core Libraries**:
- `pandas` >= 1.3.0 - Data manipulation and analysis
- `numpy` >= 1.21.0 - Numerical computing
- `scikit-learn` >= 1.0.0 - Machine learning models (Ridge, RandomForest)
- `matplotlib` >= 3.4.0 - Visualization
- `seaborn` >= 0.11.0 - Statistical visualization
- `scipy` >= 1.7.0 - Statistical functions

## 🚀 How to Run

### Step 1: Feature Engineering & Data Cleaning
```bash
jupyter notebook "Feature Engineering & Data Cleaning.ipynb"
```

**What it does**:
- Loads 100 anonymized asset CSV files
- Engineers 47 technical features (momentum, volatility, volume-based)
- Creates forward returns as target variable
- Handles missing data and outliers
- Performs train/validation/test split (~50/10/40)
- Standardizes features using training set statistics

**Output**: Cleaned dataset with engineered features ready for modeling

---

### Step 2: Model Training & Strategy Formulation
```bash
jupyter notebook "Model Training & Strategy Formulation.ipynb"
```

**What it does**:
- Trains Ridge Regression (L2 regularization) for return prediction
- Implements Random Forest meta-labeling for position sizing
- Optimizes hyperparameters (alpha, max_depth, n_estimators)
- Validates no data leakage in pipeline
- Generates position signals: {-1, 0, +1}

**Output**: Trained models + trading signals for backtesting

---

### Step 3: Backtesting & Performance Analysis
```bash
jupyter notebook "Backtesting & Performance Analysis.ipynb"
```

**What it does**:
- Simulates realistic trading with transaction costs (0.1% per trade)
- Applies leverage (L=5) and position limits
- Implements risk management (stop-loss, position sizing)
- Calculates performance metrics (Sharpe, alpha, max drawdown)
- Generates IC analysis and strategy diagnostics

**Output**: 
- **Sharpe Ratio**: 1.35 (Net), 1.41 (Gross)
- **Annual Return**: 17.71% (Net), 18.56% (Gross)
- **Max Drawdown**: -6.45% (Net), -6.27% (Gross)
- **Turnover**: 0.42x annually
- **Cost Survival**: 94.1%

---

### Step 4: Statistical Arbitrage Overlay
```bash
jupyter notebook "Statistical Arbitrage Overlay.ipynb"
```

**What it does**:
- Implements pairs trading and mean-reversion strategies
- Identifies cointegrated asset pairs using statistical tests
- Applies Z-score based entry/exit signals
- Combines with base strategy for enhanced returns

**Output**: Enhanced strategy with statistical arbitrage signals overlaid on ML predictions

## 📈 Methodology Overview

### Pipeline Architecture

```
Raw Data → Feature Engineering → Ridge Regression → Meta-Labeling (RF) → Portfolio → Backtesting
```

### Key Features

1. **Feature Engineering**
   - Momentum indicators (5, 10, 20, 60-day returns)
   - Volatility measures (rolling std, Bollinger Bands)
   - Volume analysis (VWAP, volume ratios)
   - Technical indicators (RSI, MACD, correlation features)

2. **Modeling Approach**
   - **Primary Model**: Ridge Regression (alpha=1.0)
   - **Meta-Labeling**: Random Forest (100 trees, max_depth=10)
   - Cross-validated on validation set
   - Lookahead bias prevention throughout

3. **Risk Management**
   - Position limits: [-1, 0, +1] per asset
   - Portfolio leverage: L=5
   - Transaction costs: 0.1% per trade
   - Daily rebalancing

4. **Performance Metrics**
   - Sharpe ratio (annualized)
   - Information Coefficient (IC)
   - Alpha vs buy-and-hold
   - Turnover analysis
   - Cost survival rate

## 🎯 Results Summary

| Metric | Value |
|--------|-------|
| **Sharpe Ratio (Net)** | 1.35 |
| **Sharpe Ratio (Gross)** | 1.41 |
| **Annual Return (Net)** | 17.71% |
| **Max Drawdown** | -6.45% |
| **Win Rate** | 57.6% overall |
| **Long Win Rate** | 61.5% (504 trades) |
| **Short Win Rate** | 45.9% (172 trades) |
| **Test IC** | 0.0691 |
| **Validation IC** | 0.0345 |
| **Annual Turnover** | 0.42x |
| **Cost Survival** | 94.1% |

### Yearly Performance

| Year | Sharpe (Net) | Sharpe (Gross) | Strategy Return | Benchmark Return |
|------|--------------|----------------|-----------------|------------------|
| 2023 | 1.40 | 1.45 | +23.9% | +12.9% |
| 2024 | 2.20 | 2.30 | +17.4% | +16.3% |
| 2025 | 1.28 | 1.34 | +14.8% | +13.0% |

## ⚠️ Important Notes

1. **Data Leakage Prevention**: All feature standardization uses training set statistics only
2. **Realistic Costs**: 0.1% transaction costs applied per trade
3. **Meta-Labeling**: Trained on validation data (historical outcomes), applied to test data (unknown outcomes)
4. **Target Availability**: Target variable is NEVER available at prediction time

## 📝 Approach Followed

### Phase 1: Data Preparation
- Loaded 100 asset CSVs with 2,513 days each
- Created 47 technical features per asset
- Split: ~50% train (2017-2021), ~10% validation (2022), ~40% test (2023-2025)

### Phase 2: Model Development
- Ridge Regression for directional prediction
- Random Forest meta-labeling for confidence scoring
- Hyperparameter tuning via grid search on validation set

### Phase 3: Strategy Implementation
- Combined predictions into portfolio weights
- Applied leverage (L=5) and position constraints
- Implemented daily rebalancing with transaction costs

### Phase 4: Rigorous Backtesting
- Temporal testing on out-of-sample test set
- Calculated comprehensive performance metrics
- Analyzed IC patterns and cost impacts
