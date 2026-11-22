"""
=============================================================
  F1 2026 JAPANESE GRAND PRIX — WINNER PREDICTION MODEL v3
=============================================================

Historical data: Suzuka 2018, 2019, 2022, 2023, 2024, 2025
(2020 & 2021 NOT held — COVID / calendar changes)
Feature priority (as requested):
  1. Current car performance (2026 constructor dominance)  — HIGH weight   
  2. Driver current form (2026 points, finishes)           — HIGH weight
  3. All-time driver stats (career wins, poles)            — MEDIUM weight
  4. Suzuka-specific history                               — MEDIUM weight
  5. Grid position                                         — MEDIUM weight

=============================================================

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. HISTORICAL SUZUKA DATA
#    ONLY races actually held: 2018, 2019, 2022, 2023, 2024, 2025
#    2020 CANCELLED (COVID), 2021 NOT ON CALENDAR
# ─────────────────────────────────────────────────────────────
DNF = 20
historical_data = [
    # 2018 — Hamilton wins from P1
    {"year":2018,"driver":"Hamilton",  "grid":1,"finish":1,"suzuka_wins":3,"career_wins":68,"career_poles":83,"won":1},
    {"year":2018,"driver":"Raikkonen", "grid":2,"finish":2,"suzuka_wins":2,"career_wins":21,"career_poles":18,"won":0},
    {"year":2018,"driver":"Verstappen","grid":3,"finish":3,"suzuka_wins":0,"career_wins":2, "career_poles":2, "won":0},
    {"year":2018,"driver":"Bottas",    "grid":5,"finish":4,"suzuka_wins":0,"career_wins":4, "career_poles":7, "won":0},
    {"year":2018,"driver":"Vettel",    "grid":9,"finish":6,"suzuka_wins":0,"career_wins":52,"career_poles":57,"won":0},

    # 2019 — Bottas wins from P1; Hamilton P5 after strategy
    {"year":2019,"driver":"Bottas",    "grid":1,"finish":1,"suzuka_wins":0,"career_wins":4, "career_poles":7, "won":1},
    {"year":2019,"driver":"Leclerc",   "grid":2,"finish":2,"suzuka_wins":0,"career_wins":1, "career_poles":5, "won":0},
    {"year":2019,"driver":"Vettel",    "grid":3,"finish":3,"suzuka_wins":0,"career_wins":53,"career_poles":57,"won":0},
    {"year":2019,"driver":"Hamilton",  "grid":4,"finish":4,"suzuka_wins":5,"career_wins":82,"career_poles":87,"won":0},
    {"year":2019,"driver":"Verstappen","grid":5,"finish":DNF,"suzuka_wins":0,"career_wins":7,"career_poles":3,"won":0},

    # 2022 — Verstappen wins, clinches title
    {"year":2022,"driver":"Verstappen","grid":1,"finish":1,"suzuka_wins":0,"career_wins":32,"career_poles":20,"won":1},
    {"year":2022,"driver":"Leclerc",   "grid":2,"finish":2,"suzuka_wins":0,"career_wins":6, "career_poles":18,"won":0},
    {"year":2022,"driver":"Perez",     "grid":3,"finish":3,"suzuka_wins":0,"career_wins":6, "career_poles":3, "won":0},
    {"year":2022,"driver":"Hamilton",  "grid":8,"finish":4,"suzuka_wins":5,"career_wins":103,"career_poles":103,"won":0},
    {"year":2022,"driver":"Russell",   "grid":6,"finish":5,"suzuka_wins":0,"career_wins":1, "career_poles":1, "won":0},

    # 2023 — Verstappen wins
    {"year":2023,"driver":"Verstappen","grid":1,"finish":1,"suzuka_wins":1,"career_wins":45,"career_poles":28,"won":1},
    {"year":2023,"driver":"Perez",     "grid":2,"finish":2,"suzuka_wins":0,"career_wins":6, "career_poles":3, "won":0},
    {"year":2023,"driver":"Alonso",    "grid":3,"finish":3,"suzuka_wins":0,"career_wins":32,"career_poles":22,"won":0},
    {"year":2023,"driver":"Hamilton",  "grid":4,"finish":4,"suzuka_wins":5,"career_wins":103,"career_poles":103,"won":0},
    {"year":2023,"driver":"Norris",    "grid":5,"finish":5,"suzuka_wins":0,"career_wins":2, "career_poles":4, "won":0},

    # 2024 — Verstappen wins
    {"year":2024,"driver":"Verstappen","grid":1,"finish":1,"suzuka_wins":2,"career_wins":59,"career_poles":40,"won":1},
    {"year":2024,"driver":"Norris",    "grid":2,"finish":2,"suzuka_wins":0,"career_wins":3, "career_poles":5, "won":0},
    {"year":2024,"driver":"Leclerc",   "grid":3,"finish":3,"suzuka_wins":0,"career_wins":7, "career_poles":24,"won":0},
    {"year":2024,"driver":"Sainz",     "grid":4,"finish":4,"suzuka_wins":0,"career_wins":3, "career_poles":5, "won":0},
    {"year":2024,"driver":"Russell",   "grid":5,"finish":5,"suzuka_wins":0,"career_wins":3, "career_poles":3, "won":0},

    # 2025 — Verstappen wins (5th Suzuka win)
    {"year":2025,"driver":"Verstappen","grid":1,"finish":1,"suzuka_wins":3,"career_wins":64,"career_poles":45,"won":1},
    {"year":2025,"driver":"Norris",    "grid":2,"finish":2,"suzuka_wins":0,"career_wins":4, "career_poles":8, "won":0},
    {"year":2025,"driver":"Piastri",   "grid":3,"finish":3,"suzuka_wins":0,"career_wins":3, "career_poles":3, "won":0},
    {"year":2025,"driver":"Leclerc",   "grid":4,"finish":4,"suzuka_wins":0,"career_wins":7, "career_poles":26,"won":0},
    {"year":2025,"driver":"Russell",   "grid":5,"finish":5,"suzuka_wins":0,"career_wins":5, "career_poles":5, "won":0},

]

df_hist = pd.DataFrame(historical_data)
df_hist["f_grid"]         = (1 / df_hist["grid"])
df_hist["f_suzuka"]       = (df_hist["suzuka_wins"] / 5)
df_hist["f_career"]       = (df_hist["career_wins"] / 110) * 0.8
df_hist["f_career_poles"] = (df_hist["career_poles"] / 110) * 0.6

FEATURES = ["f_grid", "f_suzuka", "f_career", "f_career_poles"]
X = df_hist[FEATURES]
y = df_hist["won"]

# ─────────────────────────────────────────────────────────────
# 2. TRAIN ENSEMBLE
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.04, max_depth=3, random_state=42)
lr = LogisticRegression(C=0.5, max_iter=2000, random_state=42)

rf.fit(X_sc, y); gb.fit(X_sc, y); lr.fit(X_sc, y)

rf_cv = cross_val_score(rf, X_sc, y, cv=5, scoring="roc_auc").mean()
gb_cv = cross_val_score(gb, X_sc, y, cv=5, scoring="roc_auc").mean()
lr_cv = cross_val_score(lr, X_sc, y, cv=5, scoring="roc_auc").mean()

print("=" * 65)
print("  CV AUC  |  RF:{:.3f}  GB:{:.3f}  LR:{:.3f}".format(rf_cv, gb_cv, lr_cv))
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 3. 2026 DRIVER DATA — accurate after 2 rounds
#
#   ACTUAL RESULTS:
#   R1 AUS: 1-Russell 2-Antonelli 3-Leclerc 4-Hamilton 5-Norris 6-Verstappen
#   R2 CHN: 1-Antonelli 2-Russell 3-Hamilton 4-Leclerc (Verstappen DNF)
#
#   STANDINGS: Russell 51 | Antonelli 47 | Leclerc 34 | Hamilton 33
#              Bearman 17 | Norris 15 | Gasly 9 | Verstappen 8
#   CONSTRUCTORS: Mercedes 98 | Ferrari 67 | McLaren 18 | Haas 17
# ─────────────────────────────────────────────────────────────

# 2026 form: average finish position (lower=better), normalised to 0-1
# Russell: avg finish 1.5 → elite; Antonelli: 1.5 → elite; etc.
# Score = 1 - (avg_finish - 1) / 19  clamped 0-1
def finish_score(avg_finish):
    return max(0, 1 - (avg_finish - 1) / 19)

form_2026 = {
    "Antonelli":  finish_score(1.5),  # P2 + P1
    "Russell":    finish_score(1.5),  # P1 + P2 — dead equal
    "Hamilton":   finish_score(3.5),  # P4 + P3
    "Leclerc":    finish_score(3.5),  # P3 + P4
    "Bearman":    finish_score(7.0),  # P7 + ~P7
    "Norris":     finish_score(10.0), # P5 R1, DNS R2 (DNS counts as ~15)
    "Gasly":      finish_score(11.0),
    "Verstappen": finish_score(11.0), # P6 R1, DNF R2
    "Piastri":    finish_score(17.0), # DNS R1 crash, DNS R2
    "Hadjar":     finish_score(15.0), # DNF R1, no finish R2
}

# Car (constructor) performance — normalised to Mercedes = 1.0
car_2026 = {
    "Antonelli":  1.00,   # Mercedes 98 pts
    "Russell":    1.00,   # Mercedes 98 pts
    "Leclerc":    0.68,   # Ferrari 67 pts
    "Hamilton":   0.68,   # Ferrari 67 pts
    "Norris":     0.18,   # McLaren 18 pts (DNF/DNS issues)
    "Piastri":    0.18,   # McLaren (if car works, fast)
    "Bearman":    0.17,   # Haas 17 pts
    "Gasly":      0.09,   # Alpine
    "Verstappen": 0.12,   # Red Bull (unreliable)
    "Hadjar":     0.12,   # Red Bull
}

# Career stats as of 2026 Japan
career_data = {
    "Russell":    (6,   6),
    "Antonelli":  (1,   1),
    "Leclerc":    (8,   28),
    "Hamilton":   (104, 104),
    "Norris":     (4,   8),
    "Bearman":    (0,   0),
    "Gasly":      (1,   0),
    "Verstappen": (64,  45),
    "Piastri":    (3,   3),
    "Hadjar":     (0,   0),
}

suzuka_wins_2026 = {
    "Verstappen": 5, "Hamilton": 5, "Leclerc": 0, "Russell": 0,
    "Antonelli": 0,  "Norris": 0,  "Piastri": 0, "Bearman": 0,
    "Gasly": 0,      "Hadjar": 0,
}

teams_2026 = {
    "Russell": "Mercedes", "Antonelli": "Mercedes",
    "Leclerc": "Ferrari",  "Hamilton": "Ferrari",
    "Norris": "McLaren",   "Piastri": "McLaren",
    "Bearman": "Haas",     "Gasly": "Alpine",
    "Verstappen": "Red Bull", "Hadjar": "Red Bull",
}

# ─────────────────────────────────────────────────────────────
# 4. DUAL-POLE SCENARIO: average Antonelli-pole & Russell-pole
#    Either Mercedes driver can take pole at Suzuka — 50/50
# ─────────────────────────────────────────────────────────────
def build_and_predict(pole_driver):
    merc_other = "Russell" if pole_driver == "Antonelli" else "Antonelli"
    rest = ["Leclerc","Hamilton","Norris","Piastri","Verstappen","Bearman","Gasly","Hadjar"]
    order = [pole_driver, merc_other] + rest

    rows = []
    for pos, drv in enumerate(order, 1):
        wins, poles = career_data[drv]
        rows.append({
            "driver": drv,
            "grid":   pos,
            "suzuka_wins":   suzuka_wins_2026[drv],
            "career_wins":   wins,
            "career_poles":  poles,
            "form_2026":     form_2026[drv],
            "car_2026":      car_2026[drv],
        })
    df = pd.DataFrame(rows)

    df["f_grid"]         = 1 / df["grid"]
    df["f_suzuka"]       = df["suzuka_wins"] / 5
    df["f_career"]       = (df["career_wins"] / 110) * 0.8
    df["f_career_poles"] = (df["career_poles"] / 110) * 0.6

    X_p = scaler.transform(df[FEATURES])
    total_w = rf_cv + gb_cv + lr_cv
    base_ml = (
        rf_cv * rf.predict_proba(X_p)[:, 1] +
        gb_cv * gb.predict_proba(X_p)[:, 1] +
        lr_cv * lr.predict_proba(X_p)[:, 1]
    ) / total_w

    # ── FINAL SCORE: car+form is PRIMARY signal, ML base is a tiebreaker ──
    # Weights: car 40%, form 40%, ML (captures grid+history) 20%
    car_norm  = df["car_2026"].values / df["car_2026"].values.sum()
    form_norm = df["form_2026"].values / df["form_2026"].values.sum()
    ml_norm   = base_ml / base_ml.sum() if base_ml.sum() > 0 else base_ml

    final_score = 0.40 * car_norm + 0.40 * form_norm + 0.20 * ml_norm
    return df["driver"].values, final_score / final_score.sum() * 100

drivers_a, probs_a = build_and_predict("Antonelli")
drivers_r, probs_r = build_and_predict("Russell")

# Average both scenarios
avg_probs = (probs_a + probs_r) / 2
drivers_list = list(drivers_a)

df_result = pd.DataFrame({
    "driver":          drivers_list,
    "team":            [teams_2026[d] for d in drivers_list],
    "suzuka_wins":     [suzuka_wins_2026[d] for d in drivers_list],
    "form_score":      [round(form_2026[d], 2) for d in drivers_list],
    "car_score":       [car_2026[d] for d in drivers_list],
    "win_probability": avg_probs,
}).sort_values("win_probability", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# 5. PRINT RESULTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  2026 JAPANESE GP — PREDICTED WIN PROBABILITIES")
print("  (avg of Antonelli-pole & Russell-pole scenarios)")
print("=" * 68)
print(f"{'Rk':<4}{'Driver':<13}{'Team':<13}{'Suz.W':<7}{'Form':<7}{'Car':<7}{'Win Prob'}")
print("-" * 68)
for i, row in df_result.iterrows():
    bar = "█" * int(row["win_probability"] / 1.2)
    print(f"  {i+1:<3}{row['driver']:<13}{row['team']:<13}{int(row['suzuka_wins']):<7}"
          f"{row['form_score']:<7}{row['car_score']:<7.2f}{row['win_probability']:.1f}%  {bar}")

w1 = df_result.iloc[0]
w2 = df_result.iloc[1]
merc_total = df_result[df_result["team"] == "Mercedes"]["win_probability"].sum()

print("\n" + "=" * 68)
print(f"  🏁  PREDICTED WINNER:   {w1['driver']} ({w1['team']})  —  {w1['win_probability']:.1f}%")
print(f"  🥈  PREDICTED P2:       {w2['driver']} ({w2['team']})  —  {w2['win_probability']:.1f}%")
print(f"\n  ⭐  Mercedes 1-2 lock-out probability: {merc_total:.1f}%")
print("=" * 68)

print("\n  FEATURE IMPORTANCE (base RF model)")
print("  " + "-" * 40)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, val in imp.items():
    print(f"  {feat:<22} {val:.3f}  {'█' * int(val * 40)}")

print("""
  ─────────────────────────────────────────────────────────────
  2026 SEASON CONTEXT (2 rounds complete)
  ─────────────────────────────────────────────────────────────
  R1 AUS: Russell > Antonelli > Leclerc > Hamilton > Norris > Verstappen
  R2 CHN: Antonelli > Russell > Hamilton > Leclerc (Verstappen DNF)

  Mercedes 98 pts | Ferrari 67 | McLaren 18 | Red Bull 12

  Both Mercedes drivers have identical 2026 pace — one win each,
  one P2 each, same car. Model treats them as co-favourites.
  Hamilton & Verstappen's 5 Suzuka wins each are the only real
  threat to a Mercedes lockout in Japan.
  ─────────────────────────────────────────────────────────────
""")