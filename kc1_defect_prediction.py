"""
Comparative Empirical Evaluation of ML Algorithms for Software Defect Prediction
Dataset: NASA KC1 (crudo)

Uso:
    python kc1_defect_prediction.py [ruta_kc1.csv] [carpeta_salida]

Requisitos: numpy, pandas, scipy, scikit-learn, matplotlib
"""


# ==========================================================================
# Comparative Empirical Evaluation of ML Algorithms for Software Defect Prediction (NASA KC1)
# ==========================================================================

# ==========================================================================
# 1. Imports y configuración
# ==========================================================================
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

IN_NOTEBOOK = "ipykernel" in sys.modules

import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # backend sin ventana cuando se ejecuta como script
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ------------------------- Configuración -------------------------
SEED = 42
N_SPLITS = 10
N_REPEATS = 10
ALPHA = 0.05
TARGET = "defects"

# Ruta al CSV crudo. En script: python kc1_defect_prediction.py [ruta_csv] [carpeta_salida]
if IN_NOTEBOOK:
    DATA_PATH = "kc1.csv"
    OUT_DIR = "resultados"
else:
    DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "kc1.csv"
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "resultados"

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

def finish_fig(filename):
    """Guarda la figura; en notebook además la muestra."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


# ==========================================================================
# 2. Carga y auditoría de datos
# ==========================================================================
df_raw = pd.read_csv(DATA_PATH)
df_raw[TARGET] = df_raw[TARGET].astype(int)   # True/False -> 1/0

n_feat = df_raw.shape[1] - 1
print(f"Dataset crudo: {df_raw.shape[0]} módulos, {n_feat} features + etiqueta '{TARGET}'")
print(f"Valores nulos (features + etiqueta): {int(df_raw.isnull().sum().sum())}")
print(f"Filas duplicadas exactas:            {int(df_raw.duplicated().sum())}")

vc = df_raw[TARGET].value_counts().sort_index()
print(f"Distribución cruda: no defectuosos = {vc[0]} ({vc[0]/len(df_raw):.2%}), "
      f"defectuosos = {vc[1]} ({vc[1]/len(df_raw):.2%})")


# ==========================================================================
# 3. Preprocesamiento: eliminación de duplicados
# ==========================================================================
assert df_raw.isnull().sum().sum() == 0, "Se esperaban 0 valores nulos"

df = df_raw.drop_duplicates(keep="first").reset_index(drop=True)

X = df.drop(columns=[TARGET])
y = df[TARGET].values

vc = df[TARGET].value_counts().sort_index()
print(f"Filas eliminadas: {len(df_raw) - len(df)}")
print(f"Dataset limpio:   {len(df)} módulos "
      f"({vc[0]} no defectuosos, {vc[1]} defectuosos; {vc[1]/len(df):.1%} defectuosos)")

# Verificación contra las cifras reportadas en el paper
assert len(df) == 1212 and vc[0] == 897 and vc[1] == 315, "No coincide con el paper"
print("OK: coincide con el paper (1,212 módulos; 897 / 315; 26.0%).")


# ==========================================================================
# 4. Modelos y pipelines
# ==========================================================================
models = {
    "LR":  LogisticRegression(solver="lbfgs", penalty="l2", C=1.0,
                              max_iter=1000, random_state=SEED),
    "DT":  DecisionTreeClassifier(criterion="gini", random_state=SEED),
    "RF":  RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  random_state=SEED, n_jobs=-1),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale",
               probability=True, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "GB":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      random_state=SEED),
}

pipelines = {
    name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    for name, clf in models.items()
}
print("Modelos:", list(pipelines))


# ==========================================================================
# 5. Validación: 10×10-fold estratificada
# ==========================================================================
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
splits = list(rskf.split(X, y))          # lista fija -> mismos folds para todos los modelos
n_folds = len(splits)

# Proporción de defectuosos por fold (verifica la estratificación)
val_rates = np.array([y[te].mean() for _, te in splits])
print(f"Folds totales: {n_folds}")
print(f"% defectuosos en validación: min={val_rates.min():.3f}  max={val_rates.max():.3f}  "
      f"(global={y.mean():.3f})")

scoring = {
    "roc_auc":   "roc_auc",
    "pr_auc":    "average_precision",
    "f1":        "f1",
    "recall":    "recall",
    "precision": "precision",
}


# ==========================================================================
# 6. Entrenamiento y evaluación
# ==========================================================================
fold_scores = {}   # nombre -> DataFrame (100 filas x 5 métricas)
fit_time = {}

for name, pipe in pipelines.items():
    t0 = time.time()
    res = cross_validate(pipe, X, y, cv=splits, scoring=scoring,
                         n_jobs=1, error_score="raise")
    fold_scores[name] = pd.DataFrame({m: res[f"test_{m}"] for m in scoring})
    fit_time[name] = time.time() - t0
    print(f"{name:>3}: ROC-AUC = {fold_scores[name]['roc_auc'].mean():.4f}   "
          f"({fit_time[name]:.1f} s)")

# Tabla larga con todos los resultados por fold
long = pd.concat(
    [d.assign(model=n, fold=np.arange(n_folds),
              repeat=np.arange(n_folds) // N_SPLITS) for n, d in fold_scores.items()],
    ignore_index=True,
)
long.to_csv(os.path.join(OUT_DIR, "scores_por_fold.csv"), index=False)


# ==========================================================================
# 7. RQ1: comparación de rendimiento
# ==========================================================================
rows = []
for name, d in fold_scores.items():
    rows.append({
        "Modelo": name,
        "ROC-AUC": d["roc_auc"].mean(),
        "ROC-AUC SD": d["roc_auc"].std(ddof=1),
        "PR-AUC": d["pr_auc"].mean(),
        "F1": d["f1"].mean(),
        "Recall": d["recall"].mean(),
        "Precision": d["precision"].mean(),
    })
summary = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
summary.to_csv(os.path.join(OUT_DIR, "tabla_rq1_resumen.csv"), index=False)

print(summary.round(4).to_string(index=False))
summary.round(4)


# ==========================================================================
# 8. RQ2: consistencia entre folds
# ==========================================================================
auc = pd.DataFrame({n: d["roc_auc"].values for n, d in fold_scores.items()})  # 100 x 6

# --- Dispersión ---
disp = pd.DataFrame({
    "Media": auc.mean(),
    "SD": auc.std(ddof=1),
    "IQR": auc.quantile(0.75) - auc.quantile(0.25),
    "Min": auc.min(),
    "Max": auc.max(),
}).sort_values("Media", ascending=False)
disp.to_csv(os.path.join(OUT_DIR, "tabla_rq2_dispersion.csv"))
print("Dispersión del ROC-AUC (100 folds):")
print(disp.round(4).to_string())

# --- Frecuencia de ranking por fold ---
ranks = auc.rank(axis=1, ascending=False, method="average")   # 1 = mejor
rank_freq = pd.DataFrame(
    {r: (ranks.round().astype(int) == r).mean() for r in range(1, auc.shape[1] + 1)}
)
rank_freq.columns = [f"Rank {r}" for r in rank_freq.columns]
rank_freq["Rank medio"] = ranks.mean()
rank_freq["% Top-1"] = (ranks == 1).mean() * 100
rank_freq = rank_freq.loc[disp.index]
rank_freq.to_csv(os.path.join(OUT_DIR, "tabla_rq2_frecuencia_ranking.csv"))
print("\nFrecuencia (proporción de folds) con la que cada modelo ocupa cada posición:")
print(rank_freq.round(3).to_string())


# ==========================================================================
# 9. Test t corregido de Nadeau–Bengio (comparaciones pareadas)
# ==========================================================================
n_train = np.mean([len(tr) for tr, _ in splits])
n_test = np.mean([len(te) for _, te in splits])
ratio = n_test / n_train
J = n_folds

def nadeau_bengio(a, b):
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0, d.mean()
    t = d.mean() / np.sqrt((1.0 / J + ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return t, p, d.mean()

names = list(auc.columns)
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        t, p, md_ = nadeau_bengio(auc[names[i]], auc[names[j]])
        pairs.append({"Modelo A": names[i], "Modelo B": names[j],
                      "Dif. media (A-B)": md_, "t": t, "p-valor": p})

pw = pd.DataFrame(pairs)

# Corrección de Holm (complementaria)
order = np.argsort(pw["p-valor"].values)
m = len(pw)
holm = np.empty(m)
running = 0.0
for k, idx in enumerate(order):
    running = max(running, (m - k) * pw["p-valor"].values[idx])
    holm[idx] = min(1.0, running)
pw["p Holm"] = holm
pw["Signif. (α=0.05, sin corr.)"] = np.where(pw["p-valor"] < ALPHA, "Sí", "No")
pw["Signif. (Holm)"] = np.where(pw["p Holm"] < ALPHA, "Sí", "No")
pw = pw.sort_values("p-valor").reset_index(drop=True)
pw.to_csv(os.path.join(OUT_DIR, "tabla_nadeau_bengio.csv"), index=False)

print(f"n_train medio = {n_train:.1f}, n_test medio = {n_test:.1f}, n_test/n_train = {ratio:.4f}, J = {J}")
print(pw.round(4).to_string(index=False))
n_sig = int((pw["p-valor"] < ALPHA).sum())
n_sig_h = int((pw["p Holm"] < ALPHA).sum())
print(f"\nPares significativos: {n_sig}/{m} (sin corrección), {n_sig_h}/{m} (Holm)")
pw.round(4)


# ==========================================================================
# 10. Gráficas
# ==========================================================================
order_names = list(disp.index)

# (a) Distribución del ROC-AUC por modelo
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.boxplot([auc[n] for n in order_names], showmeans=True)
ax.set_xticks(range(1, len(order_names) + 1))
ax.set_xticklabels(order_names)
ax.set_ylabel("ROC-AUC (100 folds)")
ax.set_title("Distribución del ROC-AUC por modelo — NASA KC1 (10×10 CV)")
ax.grid(axis="y", alpha=0.3)
finish_fig("fig_boxplot_roc_auc.png")

# (b) Frecuencia de ranking por modelo
cols = [c for c in rank_freq.columns if c.startswith("Rank ") and c != "Rank medio"]
fig, ax = plt.subplots(figsize=(8, 4.5))
rank_freq.loc[order_names, cols].plot(kind="bar", stacked=True, ax=ax, colormap="viridis_r")
ax.set_ylabel("Proporción de folds")
ax.set_xlabel("")
ax.set_title("Frecuencia de ranking por fold (ROC-AUC)")
ax.legend(title="Posición", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.setp(ax.get_xticklabels(), rotation=0)
finish_fig("fig_frecuencia_ranking.png")

# (c) Heatmap de p-valores de Nadeau–Bengio
P = pd.DataFrame(np.nan, index=order_names, columns=order_names)
for _, r in pw.iterrows():
    P.loc[r["Modelo A"], r["Modelo B"]] = r["p-valor"]
    P.loc[r["Modelo B"], r["Modelo A"]] = r["p-valor"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(P.values.astype(float), cmap="RdYlGn", vmin=0, vmax=0.2)
ax.set_xticks(range(len(order_names))); ax.set_xticklabels(order_names)
ax.set_yticks(range(len(order_names))); ax.set_yticklabels(order_names)
for i in range(len(order_names)):
    for j in range(len(order_names)):
        v = P.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
ax.set_title("p-valores Nadeau–Bengio (ROC-AUC)\nverde = más evidencia de diferencia")
fig.colorbar(im, ax=ax, label="p-valor (escala truncada a 0.2)")
finish_fig("fig_heatmap_pvalores.png")


# ==========================================================================
# 11. Resumen final
# ==========================================================================
best = summary.iloc[0]
print("=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"Dataset limpio: {len(df)} módulos ({y.mean():.1%} defectuosos) | "
      f"{J} folds ({N_SPLITS}×{N_REPEATS}) | seed={SEED}")
print(f"\nRQ1 — Mejor ROC-AUC medio: {best['Modelo']} = {best['ROC-AUC']:.4f} "
      f"(SD {best['ROC-AUC SD']:.4f})")
print("Ranking por ROC-AUC medio: " + " > ".join(summary["Modelo"]))
print(f"\nRQ2 — Pares con diferencia significativa (Nadeau–Bengio, α={ALPHA}): "
      f"{n_sig}/{m} sin corrección; {n_sig_h}/{m} con Holm")
print(f"Modelo con mayor % de folds en 1.ª posición: "
      f"{rank_freq['% Top-1'].idxmax()} ({rank_freq['% Top-1'].max():.0f}% de los folds)")
print(f"\nArchivos guardados en: {os.path.abspath(OUT_DIR)}")
