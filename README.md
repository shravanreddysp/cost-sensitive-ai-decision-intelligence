# Cost-Sensitive AI Decision Intelligence

**Research title:** Cost-Sensitive AI Decision Intelligence: A Data-Driven Framework for Utility-Optimized Decisions under Class Imbalance

This repository is configured to run entirely in **GitHub Actions**. No RunPod or local Python installation is required.

## Easiest run method

1. Open the repository on GitHub.
2. Click **Actions**.
3. Select **Run AI Decision Intelligence Experiment**.
4. Click **Run workflow** and confirm **Run workflow**.
5. Open the new workflow run and wait until the `experiment` job completes.
6. At the bottom of the workflow-run page, under **Artifacts**, download **cost-sensitive-ai-di-results**.
7. Unzip it. The main files are:
   - `summary.md`
   - `tables/model_metrics_test.csv`
   - `tables/decision_policy_costs_test.csv`
   - `tables/bootstrap_ci.csv`
   - publication figures under `figures/`

## Experiment design

Dataset: official UCI Default of Credit Card Clients dataset (ID 350).

Models:
- Logistic Regression
- Random Forest
- XGBoost
- Class-weighted XGBoost

Decision strategies:
1. Conventional threshold = 0.50
2. F1-optimized threshold
3. Bayes cost-sensitive threshold
4. Empirically utility-optimized threshold
5. Robust multi-scenario utility optimization
6. Class-weighted XGBoost

Probability estimates are independently Platt calibrated.

Data split:
- Training: 60%
- Probability calibration: 15%
- Decision-threshold validation: 10%
- Final untouched test: 15%

Cost-ratio scenarios:
- `C_FN / C_FP = 2, 3, 5, 8, 10`

Important: these cost ratios are normalized decision-loss scenarios. They must not be described as real monetary losses unless externally validated business cost data are added.

## Optional local run

```bash
chmod +x run_all.sh
./run_all.sh
```
