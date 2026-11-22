# F1 Race Winner Predictor: 2026 Japanese GP (Suzuka)

Predicts win probabilities for the 2026 Japanese Grand Prix by blending an ML ensemble trained
on past Suzuka races with hand-encoded 2026 car and driver form. Single script, scikit-learn.

## How it works

1. **Historical training data** (hard-coded in the script): top finishers at Suzuka 2018, 2019,
   2022, 2023, 2024 and 2025 (2020 cancelled, 2021 not held), with grid slot, finish (DNF = 20),
   Suzuka wins, career wins and poles, and `won` as the label.
2. **Features:** `f_grid = 1/grid`, Suzuka wins / 5, scaled career wins and poles, after
   `StandardScaler`.
3. **Ensemble:** Random Forest (300 trees, depth 4), Gradient Boosting (300, lr 0.04, depth 3)
   and Logistic Regression (C = 0.5). Each model's cross-validated **ROC-AUC** is printed and
   used as its weight in the averaged win probability.
4. **2026 field:** 10 drivers (Antonelli, Russell, Leclerc, Hamilton, Norris, Piastri, Verstappen,
   Bearman, Gasly, Hadjar) with 2026 team, form score (average finish after 2 rounds,
   normalised), car score (constructor strength) and career/Suzuka stats.
5. **Final score** = 40% car + 40% form + 20% ML ensemble. Current-season pace dominates, and the ML
   part acts as a grid/history tiebreaker.
6. Two grid scenarios (Antonelli on pole, Russell on pole) are averaged. Output is a ranked
   table, predicted winner and P2, and the probability of a Mercedes 1-2.

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # pandas, numpy, scikit-learn
python f1_japan_gp_predictor.py
```

## Limitations

- Training set is tiny (about 5 drivers × 6 races), so the CV AUCs are noisy. The 2026 inputs are
  hand-set judgement values, so the result is mostly driven by those assumptions, not learned.
- All data is inline in the script; there's no data loader or API (e.g. FastF1/Ergast) yet.
