# DCISprogression

Package versions:
- Python 3.10.6
- pytorch 1.9.1
- scanpy 1.9.1
- scipy 1.9.1
- numpy 1.23.3
- scikit-learn 1.1.2
- matplotlib-base 3.6.0
- matplotlib 3.6.0
- seaborn 0.12.0
- pandas 1.5.3
- umap-learn 0.5.3
- anndata 0.8.0

The script for training the autoencoder is train_cnnvae.ipynb.

Notebooks for clustering and analyzing cluster statistics start with "cluster" in the file names.

Notebooks for pseudotime analysis start with "cluster" in the filenames.

The notebooks for training cluster classifier using NMCO scores are train_clusterClf_NMCO_balanced.ipynb (top-level clusters) and /train_clusterClf_NMCO_balanced_subclusters.ipynb (subclusters)

Notebooks for training classifiers of cells inside vs outside of breast ducts using either NMCO scores or CNN latent start with "train_ductClf" in the file names.

Notebooks for disease stage classification using different inputs start with "train_pathologyClf" in the file names.
