import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("features_2021.csv")
# Split the data into offense and defense
offense_data = data[['Points_mean', 'Poss_mean', 'OE_mean', 'FGM_mean', 'FGA_mean', 'FGM3_mean', 'FGA3_mean', 'FTM_mean', 'FTA_mean', 'OR_mean', 'Ast_mean', 'TO_mean', 'Stl_mean', 'Blk_mean', 'Fouls_mean']]
defense_data = data[['OppPoints_mean', 'OppFGM_mean', 'OppFGA_mean', 'OppFGM3_mean', 'OppFGA3_mean', 'OppFTM_mean', 'OppFTA_mean', 'OppOR_mean', 'OppAst_mean', 'OppTO_mean', 'OppStl_mean', 'OppBlk_mean', 'OppFouls_mean']]

# Perform PCA on offense data
pca_offense = PCA(n_components=2)
offense_pca_result = pca_offense.fit_transform(offense_data)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
offense_clusters = kmeans.fit_predict(offense_pca_result)
data['OffenseCluster'] = offense_clusters

# Perform PCA on defense data
pca_defense = PCA(n_components=2)
defense_pca_result = pca_defense.fit_transform(defense_data)
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
defense_clusters = kmeans.fit_predict(defense_pca_result)
data['DefenseCluster'] = defense_clusters

# Visualize the clustered offense PCA
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(offense_pca_result[:, 0], offense_pca_result[:, 1], c=offense_clusters)
plt.title('Offense PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Visualize defense PCA
plt.subplot(1, 2, 2)
plt.scatter(defense_pca_result[:, 0], defense_pca_result[:, 1], c=defense_clusters)
plt.title('Defense PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.savefig('clusters.png')