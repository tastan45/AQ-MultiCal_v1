# AQ-MultiCal v1: Standardized Low-Cost CO₂ Sensor Calibration Framework

This repository contains the refactored, production-ready source code for the **AQ-MultiCal** platform, accompanying our comprehensive evaluation framework for low-cost sensor (LCS) calibration. The platform delivers an interactive, web-based environment tailored for machine learning modeling and hyperparameter optimization.

## 📖 Main Citation & Publication Info
If you utilize this framework, codebase, or parts of the platform architecture in your academic work, please cite our corresponding study:

* **Title:** *Comparative Evaluation of Machine Learning and Hyperparameter Optimization Methods for Low-Cost CO₂ Sensor Calibration in Terms of Performance and Computational Cost*
* **Authors:** Eren Cihan Karsu Asal, Mehmet Taştan, Hayrettin Gökozan, Müge Erel Özçevik, Yusuf Özçevik
* **Affiliation:** Manisa Celal Bayar University, Turkey
* **Journal:** MDPI Sensors (Under Review)
* **Interactive Live Web Application:** [AQ-MultiCal Live Tool](https://aq-multical-73g6ufjhbbpplxpvza5efe.streamlit.app)

---

## 🎯 Research Overview & Core Framework
This iteration of AQ-MultiCal establishes a standardized and strictly controlled benchmarking pipeline to evaluate different machine learning architectures against various hyperparameter optimization (HPO) techniques under identical computational resource budgets.

### 1. Supported Machine Learning Models (10 Architectures)
The framework systematically integrates ten diverse predictive algorithms categorized across structural groups:
* **Linear & Regularized:** Ridge Regression, ElasticNet
* **Distance-Based:** k-Nearest Neighbors (kNN)
* **Tree-Based:** Decision Tree (DT), Random Forest (RF), Extra Trees (ET)
* **Advanced Ensemble Boosting:** Gradient Boosting (GB), AdaBoost, XGBoost, LightGBM, CatBoost

### 2. Hyperparameter Optimization (HPO) Strategies
To ensure a fair and rigorous comparison, all optimization techniques are strictly bound to a uniform budget of **18 evaluations/iterations**:
* **Grid Search (GS):** Evaluates a compact, strategically discretized hyperparameter mesh.
* **Random Search (RS):** Samples non-discretized continuous and integer distributions randomly.
* **Bayesian Optimization (BO):** Employs Gaussian Processes (`scikit-optimize / BayesSearchCV`) to sequentially explore parameter spaces based on prior performance.

### 3. Datasets & Baseline Setup
* **Hardware:** Five NDIR-based low-cost CO₂ sensors deployed alongside a high-precision reference instrument (Dienmern DM72b).
* **Validation Split:** Structured multi-location tracking with options for Train/Validation/Test data allocation (e.g., 60% Train, 20% Val, 20% Test) supporting multi-sensor generalizability tests.

---

## 📊 Key Findings Incorporated in this Version
* **Hyperparameter Sensitivity:** Distance-based models like **kNN** benefit the most from HPO execution, demonstrating an impressive accuracy boost, driving the test RMSE down to **54.4 ppm** for short-term and **26.2 ppm** for long-term profiles. Tree-based ensembles (like Random Forest) show excellent out-of-the-box structural stability with strong baseline configurations.
* **Optimization Convergence:** Given well-bounded search intervals, Grid Search, Random Search, and Bayesian Optimization frequently converge to structurally identical optimal hyperparameter regions within the 18-iteration constraint.
* **The Computational Trade-off:** While predictive performance remains tightly bounded among different HPO methods, the computational overhead expands drastically (up to a 40-fold increase) for massive ensemble architectures like RF and Gradient Boosting, establishing **Random Search (RS)** as an exceptionally cost-effective option for resource-constrained IoT nodes.

---

## 🛠️ Repository Architecture & Local Deployment
The platform is fully containerized/packaged via **Streamlit** and written in **Python 3.12**.

### File Structure
* `app_refactor_1.py`: The main user interface and application driver.
* `config.py`: Global styling parameters, Plotly layout adjustments, and pollutant physical units.
* `data_processing.py`: Automated delimiter inference, file merging, and long-form temporal synchronization.
* `model_registry.py`: Defines search limits, core hyperparameters, and cross-validation definitions for all 10 models.
* `training.py`: Handles optimization loops, metric evaluation ($R^2$, RMSE, MAE), and permutation feature importance computation.
* `plotting.py`: Generates publication-ready figures including 1:1 prediction fits, residual error density charts, and time-series plots.

### Requirements & Installation
To run the server locally or build it on a remote system, ensure you install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt