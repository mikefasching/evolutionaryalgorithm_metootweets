# Clustering #MeToo Twitter Discourse with Evolutionary Algorithms

This repository contains the full implementation of a research project comparing the performance of three evolutionary algorithms for semantic clustering of tweets from the #MeToo movement. The goal is to identify meaningful discourse clusters using Word2Vec embeddings and swarm intelligence.

---

## 📘 Paper Title

**Comparative Analysis of Evolutionary Algorithms for Clustering #MeToo Twitter Discourse**  
*If you’re reading this repo as part of the paper, welcome!*

---

## 📦 Algorithms Implemented

This project includes custom implementations of three advanced clustering algorithms with performance tracking and angular movement mechanisms:

- **Ant Colony Optimization (ACO)** → `modified_ant_colony.py`
- **Artificial Bee Colony (ABC)** → `modified_bee_colony.py`
- **Enhanced Fireworks Algorithm (EFWA)** → `modified_fireworks.py`

Each algorithm is adapted for semantic vector spaces using cosine similarity and tested for clustering performance using the Calinski-Harabasz (CH) Index and Silhouette Score.


---

## 🧪 How to Run the Project

1. **Install required libraries**

```bash
pip install numpy matplotlib scikit-learn pandas tqdm joblib progressbar2
```

2. **Prepare your tweet embedding data**
- Input should be an array of tweet vectors (e.g., Word2Vec).
- Make sure embeddings are normalized for cosine similarity.

3. **Run comparisons via notebook**
- Open `comparison.ipynb` to run and visualize clustering experiments across all three algorithms.

---

## 📈 Evaluation Metrics

- **Calinski-Harabasz Score** (primary fitness function)
- **Silhouette Score** (secondary validation)
- **Centroid Angular Movement**
- **Fitness vs. Evaluations and Iteration Time**
- **Statistical Significance via Paired t-tests**

---

## 📊 Sample Output

- Performance plots over time
- Cluster silhouette visualizations
- Fitness difference plots comparing ACO, ABC, and EFWA

---

## 🔁 Reproducibility

All algorithms load shared `initial_centroids_*.npy` files to ensure **identical starting conditions** for a fair comparative analysis.

---

## 📄 Citation

If you use this code in your work, please consider citing:
```
Michael Fasching (2025). Comparative Analysis of Evolutionary Algorithms for Clustering #MeToo Twitter Discourse.
```

---

## 📬 Contact

Questions or feedback? Reach out to [Michael Fasching](mailto:fasmim0@gmail.com).

---
