# Segmentation IQ



**Interactive clustering & interpretability playground, deployed on Streamlit Cloud**

This repository contains:

- `ml.segmentation.iq.py` — the Streamlit app entrypoint
- `requirements.txt`       — Python dependencies
- `customers_500.csv`       — sample dataset (500 rows) for quick testing
- `README.md`              — this documentation
- `.streamlit/config.toml`  — optional theme/config settings

---

## 🚀 Live Demo

Launch the app directly on Streamlit Cloud:

[**https://segmenation-unsupervised-ml-v1.streamlit.app/file**](https://segmenation-unsupervised-ml-v1.streamlit.app/file)

Use the **customers\_500.csv** file (included below) to test the full clustering flow immediately.

---

## 📂 Repository Structure

```txt
segmentation-iq/
├── ml.segmentation.iq.py    # Streamlit app
├── requirements.txt         # Python dependencies
├── customers_500.csv        # Sample dataset (500 rows)
├── README.md                # Project documentation (this file)
└── .streamlit/
    └── config.toml          # (optional) theme & server config
```

---

## 🎯 Quick Start

1. **Clone** this repo:

   ```bash
   git clone https://github.com/your-username/segmentation-iq.git
   cd segmentation-iq
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run** locally:

   ```bash
   streamlit run ml.segmentation.iq.py
   ```

4. **Test** with the provided sample:

   - On the app, upload `customers_500.csv` under “Configure & Run”.

---

## 🏃‍♂️ Usage

1. **Upload** any CSV with:
   - `Candidate_ID` (unique identifier)
   - Numeric feature columns
2. **Preprocess**: imputation, log‐transform, scaling, correlation & variance filtering
3. **PCA**: inspect explained variance; automatic component selection
4. **Clustering**: choose K-Means, GMM, Hierarchical, or DBSCAN
5. **Evaluation**: elbow/silhouette, AIC/BIC, dendrogram, k‑distance
6. **Visualize**: PCA & t‑SNE scatter, donut charts (count & %)
7. **Interpret**: per‑candidate distance & feature‑contribution metrics, ranking
8. **Scenarios**: save named runs and compare metrics side‑by‑side

---

## ☁️ Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Sign in at [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app, select `ml.segmentation.iq.py`
4. Deploy—future `main` pushes auto‑update

### Heroku (optional)

1. Create Heroku app & set `HEROKU_API_KEY` in GitHub Secrets
2. Enable auto‑deploy via GitHub Actions (see `.github/workflows`)

---

## 📄 Configuration

Optional `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4a90e2"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
font = "sans serif"
```

---

## 📦 Requirements

```txt
streamlit>=1.18.0
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=1.0.0
scipy>=1.5.0
plotly>=5.0.0
matplotlib>=3.0.0
ace_tools>=0.1.0
```

---

## 🤝 Contributing

1. Fork & clone
2. Create a feature branch
3. Make changes & open a PR
4. Ensure code quality & compatibility

---

*Segmentation IQ* • v1.1

