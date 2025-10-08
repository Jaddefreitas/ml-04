# ==========================
# 1. Importação das bibliotecas
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# ==========================
# 2. Leitura dos dados
# ==========================
arquivo = "RTVue_20221110_MLClass.xlsx"  # Substitua pelo caminho do arquivo
df = pd.read_excel(arquivo)

print("Dimensão original do dataset:", df.shape)
print("Colunas originais:", list(df.columns))

# ==========================
# 3. Seleção e limpeza das colunas relevantes
# ==========================
# Colunas descritivas
colunas_descritivas = ['Index', 'pID', 'Age', 'Gender', 'Eye']

# Colunas de medidas epiteliais (variáveis para o clustering)
colunas_epiteliais = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

# Garantir que existam no dataframe
colunas_existentes = [c for c in colunas_epiteliais if c in df.columns]

# Criar novo dataframe apenas com as colunas de interesse
dados = df[colunas_descritivas + colunas_existentes].copy()

# Tratar valores ausentes apenas nas medidas
dados[colunas_existentes] = dados[colunas_existentes].fillna(dados[colunas_existentes].mean())

print(f"\nColunas epiteliais consideradas ({len(colunas_existentes)}): {colunas_existentes}")

# ==========================
# 4. Normalização dos dados (somente medidas epiteliais)
# ==========================
scaler = StandardScaler()
dados_norm = scaler.fit_transform(dados[colunas_existentes])

# ==========================
# 5. Determinação do número ideal de clusters (método do cotovelo)
# ==========================
inertia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_norm)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

# ==========================
# 6. Aplicar o K-Means (ajuste k com base no gráfico)
# ==========================
k_opt = 3  # Pode ajustar após ver o gráfico
kmeans = KMeans(n_clusters=k_opt, random_state=42)
labels = kmeans.fit_predict(dados_norm)

dados['Cluster'] = labels

# ==========================
# 7. Redução de dimensionalidade (PCA)
# ==========================
pca = PCA(n_components=2)
componentes = pca.fit_transform(dados_norm)

dados['PCA1'] = componentes[:, 0]
dados['PCA2'] = componentes[:, 1]

# ==========================
# 8. Visualização dos clusters
# ==========================
plt.figure(figsize=(7,5))
sns.scatterplot(data=dados, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=80)
plt.title(f'Clusters Epiteliais - K-Means (k={k_opt})')
plt.show()

# ==========================
# 9. Estatísticas médias por cluster
# ==========================
print("\nMédias das variáveis epiteliais por cluster:")
print(dados.groupby('Cluster')[colunas_existentes].mean().round(2))

# ==========================
# 10. (Opcional) Perfil dos pacientes por cluster
# ==========================
perfil_cluster = dados.groupby('Cluster')[['Age']].agg(['mean', 'min', 'max'])
print("\nPerfil etário por cluster:")
print(perfil_cluster)
