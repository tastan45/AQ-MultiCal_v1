import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

try:
    from scipy.stats import loguniform, uniform, randint
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from skopt.space import Real, Integer, Categorical
    _SKOPT_SPACE_AVAILABLE = True
except ImportError:
    _SKOPT_SPACE_AVAILABLE = False

# Safe fallback objects so this module can be imported even when scikit-optimize is not installed.
# Bayesian Optimization itself is still disabled through _SKOPT_SPACE_AVAILABLE=False.
if not _SKOPT_SPACE_AVAILABLE:
    class _UnavailableSkoptDimension:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    Real = Integer = Categorical = _UnavailableSkoptDimension


# ============================================================
# FAIR 18-ITERATION SEARCH SPACES
# GS uses compact discrete grids with 18 combinations where applicable.
# RS and BO use the same lower/upper limits or categorical choices as GS.
# ============================================================

COMMON_PARAM_GRID = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 30, None],
        "min_samples_split": [2, 5],
    },
    "Decision Tree": {
        "max_depth": [5, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "loss": ["linear", "exponential"],
    },
    "k-Nearest Neighbors (kNN)": {
        "n_neighbors": [20, 30, 40],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
    "ElasticNet Regression": {
        "alpha": [0.01, 1.0, 5.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "fit_intercept": [True, False],
    },
    "Ridge Regression": {
        "alpha": [0.01, 10.0, 100.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "lsqr"],
    },
    "Lasso Regression": {
        "alpha": [0.01, 1.0, 100.0],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000, 10000],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 7],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 40],
    },
    "CatBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6],
    },
    "Linear Regression": {
        "fit_intercept": [True, False],
    },
}


RANDOM_PARAM_SPACES = {
    "Random Forest": {
        "n_estimators": randint(100, 301),
        "max_depth": [5, 30, None],
        "min_samples_split": randint(2, 6),
    },
    "Decision Tree": {
        "max_depth": [5, 20, None],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 3),
    },
    "Gradient Boosting": {
        "n_estimators": randint(100, 301),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(3, 6),
    },
    "AdaBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.2),
        "loss": ["linear", "exponential"],
    },
    "k-Nearest Neighbors (kNN)": {
        "n_neighbors": randint(20, 41),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
    "ElasticNet Regression": {
        "alpha": loguniform(0.01, 5.0),
        "l1_ratio": uniform(0.1, 0.8),
        "fit_intercept": [True, False],
    },
    "Ridge Regression": {
        "alpha": loguniform(0.01, 100.0),
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "lsqr"],
    },
    "Lasso Regression": {
        "alpha": loguniform(0.01, 100.0),
        "fit_intercept": [True, False],
        "max_iter": randint(1000, 10001),
    },
    "XGBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "max_depth": randint(3, 8),
    },
    "LightGBM": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "num_leaves": randint(20, 41),
    },
    "CatBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "depth": randint(4, 7),
    },
    "Linear Regression": {
        "fit_intercept": [True, False],
    },
}


BAYES_PARAM_SPACES = {
    "Random Forest": {
        "n_estimators": Integer(100, 300, name="n_estimators"),
        "max_depth": Categorical([5, 30, None], name="max_depth"),
        "min_samples_split": Integer(2, 5, name="min_samples_split"),
    },
    "Decision Tree": {
        "max_depth": Categorical([5, 20, None], name="max_depth"),
        "min_samples_split": Integer(2, 10, name="min_samples_split"),
        "min_samples_leaf": Integer(1, 2, name="min_samples_leaf"),
    },
    "Gradient Boosting": {
        "n_estimators": Integer(100, 300, name="n_estimators"),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform", name="learning_rate"),
        "max_depth": Integer(3, 5, name="max_depth"),
    },
    "AdaBoost": {
        "n_estimators": Integer(50, 200, name="n_estimators"),
        "learning_rate": Real(0.01, 0.2, prior="log-uniform", name="learning_rate"),
        "loss": Categorical(["linear", "exponential"], name="loss"),
    },
    "k-Nearest Neighbors (kNN)": {
        "n_neighbors": Integer(20, 40, name="n_neighbors"),
        "weights": Categorical(["uniform", "distance"], name="weights"),
        "metric": Categorical(["euclidean", "manhattan", "chebyshev"], name="metric"),
    },
    "ElasticNet Regression": {
        "alpha": Real(0.01, 5.0, prior="log-uniform", name="alpha"),
        "l1_ratio": Real(0.1, 0.9, prior="uniform", name="l1_ratio"),
        "fit_intercept": Categorical([True, False], name="fit_intercept"),
    },
    "Ridge Regression": {
        "alpha": Real(0.01, 100.0, prior="log-uniform", name="alpha"),
        "fit_intercept": Categorical([True, False], name="fit_intercept"),
        "solver": Categorical(["auto", "svd", "lsqr"], name="solver"),
    },
    "Lasso Regression": {
        "alpha": Real(0.01, 100.0, prior="log-uniform", name="alpha"),
        "fit_intercept": Categorical([True, False], name="fit_intercept"),
        "max_iter": Integer(1000, 10000, name="max_iter"),
    },
    "XGBoost": {
        "n_estimators": Integer(50, 200, name="n_estimators"),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform", name="learning_rate"),
        "max_depth": Integer(3, 7, name="max_depth"),
    },
    "LightGBM": {
        "n_estimators": Integer(50, 200, name="n_estimators"),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform", name="learning_rate"),
        "num_leaves": Integer(20, 40, name="num_leaves"),
    },
    "CatBoost": {
        "n_estimators": Integer(50, 200, name="n_estimators"),
        "learning_rate": Real(0.01, 0.1, prior="log-uniform", name="learning_rate"),
        "depth": Integer(4, 6, name="depth"),
    },
    "Linear Regression": {
        "fit_intercept": Categorical([True, False], name="fit_intercept"),
    },
}


# --- TOP 3 PARAMETERS FOR AUTOMATIC OPTIMIZATION ---
# Auto Search is intentionally aligned with AUTO_OPTIMIZE_PARAMS for fair GS/RS/BO comparison.
AUTO_OPTIMIZE_PARAMS = {
    "Random Forest": ["n_estimators", "max_depth", "min_samples_split"],
    "Decision Tree": ["max_depth", "min_samples_split", "min_samples_leaf"],
    "Gradient Boosting": ["n_estimators", "learning_rate", "max_depth"],
    "AdaBoost": ["n_estimators", "learning_rate", "loss"],
    "k-Nearest Neighbors (kNN)": ["n_neighbors", "weights", "metric"],
    "ElasticNet Regression": ["alpha", "l1_ratio", "fit_intercept"],
    "Ridge Regression": ["alpha", "fit_intercept", "solver"],
    "Lasso Regression": ["alpha", "fit_intercept", "max_iter"],
    "XGBoost": ["n_estimators", "learning_rate", "max_depth"],
    "LightGBM": ["n_estimators", "learning_rate", "num_leaves"],
    "CatBoost": ["n_estimators", "learning_rate", "depth"],
    "Linear Regression": ["fit_intercept"],
}

AUTO_SEARCH_PARAMS = dict(AUTO_OPTIMIZE_PARAMS)


# --- Model Instances (using default parameters, will be overridden if optimized) ---
MODELS = {
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "k-Nearest Neighbors (kNN)": KNeighborsRegressor(n_jobs=-1),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=42), random_state=42),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "ElasticNet Regression": ElasticNet(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
}


# ============================================================
# MANUAL SELECTION FIX: full model-specific parameter spaces
# Manual mode keeps additional optional parameters visible, while the
# top-3 parameters are aligned with the fair ranges above.
# ============================================================

EXTENDED_COMMON_PARAM_GRID = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 30, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        "bootstrap": [True, False],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 3, 5],
    },
    "k-Nearest Neighbors (kNN)": {
        "n_neighbors": [20, 30, 40],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev"],
        "p": [1, 2],
    },
    "Linear Regression": {"fit_intercept": [True, False]},
    "Decision Tree": {
        "max_depth": [5, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", None, 1.0],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "loss": ["linear", "exponential"],
        "estimator__max_depth": [1, 2, 3],
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "lsqr"],
    },
    "Lasso Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "max_iter": [1000, 5000, 10000],
    },
    "ElasticNet Regression": {
        "alpha": [0.01, 0.1, 1.0, 5.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "fit_intercept": [True, False],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40],
        "max_depth": [-1, 5, 7],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    },
    "CatBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6],
        "l2_leaf_reg": [1, 3, 5],
    },
}

EXTENDED_RANDOM_PARAM_SPACES = dict(RANDOM_PARAM_SPACES)
EXTENDED_RANDOM_PARAM_SPACES.update({
    "Random Forest": {
        "n_estimators": randint(100, 301),
        "max_depth": [5, 30, None],
        "min_samples_split": randint(2, 6),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        "bootstrap": [True, False],
    },
    "Gradient Boosting": {
        "n_estimators": randint(100, 301),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(3, 6),
        "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 6),
    },
    "k-Nearest Neighbors (kNN)": {
        "n_neighbors": randint(20, 41),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "chebyshev"],
        "p": [1, 2],
    },
    "Decision Tree": {
        "max_depth": [5, 20, None],
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 3),
        "max_features": ["sqrt", "log2", None, 1.0],
    },
    "AdaBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.2),
        "loss": ["linear", "exponential"],
        "estimator__max_depth": [1, 2, 3],
    },
    "XGBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "max_depth": randint(3, 8),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "LightGBM": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "num_leaves": randint(20, 41),
        "max_depth": [-1, 5, 7],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "CatBoost": {
        "n_estimators": randint(50, 201),
        "learning_rate": loguniform(0.01, 0.1),
        "depth": randint(4, 7),
        "l2_leaf_reg": randint(1, 6),
    },
})

EXTENDED_BAYES_PARAM_SPACES = dict(BAYES_PARAM_SPACES)
if _SKOPT_SPACE_AVAILABLE:
    EXTENDED_BAYES_PARAM_SPACES.update({
        "Random Forest": {
            "n_estimators": Integer(100, 300),
            "max_depth": Categorical([5, 30, None]),
            "min_samples_split": Integer(2, 5),
            "min_samples_leaf": Integer(1, 4),
            "max_features": Categorical(["sqrt", "log2", 0.6, 0.8, 1.0]),
            "bootstrap": Categorical([True, False]),
        },
        "Gradient Boosting": {
            "n_estimators": Integer(100, 300),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "max_depth": Integer(3, 5),
            "subsample": Real(0.6, 1.0),
            "min_samples_leaf": Integer(1, 5),
        },
        "k-Nearest Neighbors (kNN)": {
            "n_neighbors": Integer(20, 40),
            "weights": Categorical(["uniform", "distance"]),
            "metric": Categorical(["euclidean", "manhattan", "chebyshev"]),
            "p": Categorical([1, 2]),
        },
        "Decision Tree": {
            "max_depth": Categorical([5, 20, None]),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 2),
            "max_features": Categorical(["sqrt", "log2", None, 1.0]),
        },
        "AdaBoost": {
            "n_estimators": Integer(50, 200),
            "learning_rate": Real(0.01, 0.2, prior="log-uniform"),
            "loss": Categorical(["linear", "exponential"]),
            "estimator__max_depth": Categorical([1, 2, 3]),
        },
        "XGBoost": {
            "n_estimators": Integer(50, 200),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "max_depth": Integer(3, 7),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
        },
        "LightGBM": {
            "n_estimators": Integer(50, 200),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "num_leaves": Integer(20, 40),
            "max_depth": Categorical([-1, 5, 7]),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
        },
        "CatBoost": {
            "n_estimators": Integer(50, 200),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "depth": Integer(4, 6),
            "l2_leaf_reg": Integer(1, 5),
        },
    })

ALL_OPTIMIZABLE_PARAM_NAMES = {model_name: list(params.keys()) for model_name, params in EXTENDED_COMMON_PARAM_GRID.items()}
