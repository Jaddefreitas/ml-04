# epithelium_clustering.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import os

# ------------- CONFIG -------------
FILE_PATH = "RTVue_20221110_MLClass.xlsx"   # ajuste se necessário
OUTPUT_CSV = "epithelial_clusters.csv"
FEATURES = ['C','S','ST','T','IT','I','IN','N','SN']
K_MIN = 2
K_MAX = 6   # reduzido para velocidade; ajuste para 8-10 se quiser mais testes
SILH_SAMPLE_SIZE = 1000  # silhouette em amostra para performance
DENDRO_SAMPLE = 400      # se quiser dendrograma (opcional)
RANDOM_STATE = 42
# ----------------------------------

# 1) Ler os dados
df = pd.read_excel(FILE_PATH, sheet_name=0)

# 2) Resumo inicial
print("Registros:", len(df))
print("Colunas presentes:", df.columns.tolist())
print("\nMissing counts por região:")
print(df[FEATURES].isna().sum())

# 3) Imputar (mediana) e padronizar os dados
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(df[FEATURES])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4) Testar o k (Elbow + Silhouette)
Ks = list(range(K_MIN, K_MAX+1))
inertia = []
silhs = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    try:
        sil = silhouette_score(X_scaled, labels, sample_size=min(SILH_SAMPLE_SIZE, X_scaled.shape[0]), random_state=RANDOM_STATE)
    except Exception:
        sil = silhouette_score(X_scaled, labels)
    silhs.append(sil)
    print(f"k={k} -> inertia={km.inertia_:.1f}, silhouette_sampled={sil:.4f}")

# 5) Escolher o k pelo maior silhouette
best_k = Ks[int(np.argmax(silhs))]
print("\nMelhor k por silhouette:", best_k)

# 6) Ajuste final KMeans
