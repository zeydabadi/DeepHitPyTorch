## 1. Overview

This project tests **DeepHit**, a discrete‐time competing‐risks survival model, implemented using **PyTorch** and **Pycox**. The pipeline consists of the following steps:

- **Data loading & splitting**  
  Stratified train/validation/test splits that preserve event‐rate distributions.

- **Preprocessing**  
  Feature standardization and discrete‐time label transformation (`NUM_DURATIONS = 16`, equidistant intervals).

- **Model**  
  A simple 2‐layer MLP predicting per‐event, per‐time‐step risks.

- **Training**  
  Uses Adam optimizer with ranking + likelihood loss (from DeepHit), and early stopping on validation loss.

- **Evaluation**  
  - Time‐dependent concordance (Antolini)
  - IPCW C-index at quartile time points
  - Brier scores for each event at specified horizons

---

## 2. Usage Instructions

### 2.1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2.2. Run training and evaluation

```bash
python src/main.py --data ./data/SYNTHETIC/synthetic_comprisk.csv --durations 16 --batch-size 128 --epochs 100 --lr 0.005
```

### 2.3. Inspect results and plots

- Loss curves and CIF plots are saved to `outputs/figures/`
- Evaluation metrics are logged in `outputs/metrics.json`

---

## 3. Edge Cases

- **Single Event (No Competing Risk)**  
  To test with only one critical event:

  ```bash
  python src/main.py --data ./data/SYNTHETIC/synthetic_one_label.csv --durations 16 --batch-size 128 --epochs 100 --lr 0.005
  ```

- **All Censored Observations**  
  If all observations are censored, there is no direct information on event-time distributions, only that events exceed the last follow-up.  
  In this case, neither nonparametric nor semiparametric methods can estimate survival beyond confirming it remains at 1 until the last censoring time.
