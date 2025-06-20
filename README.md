# Segmentation IQ



**Clustering & Interpretability Playground**

This repository contains a Streamlit application (`ml.segmentation.iq.py`) for end-to-end clustering and interpretability of customer/store feature datasets.

## 📂 Repository Structure

```txt
segmentation-iq/
├── ml.segmentation.iq.py  # Streamlit application entrypoint
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
└── .streamlit/
    └── config.toml        # (optional) Streamlit theme and server config
```

## 🚀 Quick Start

1. **Clone** this repository:

   ```bash
   git clone https://github.com/your-username/segmentation-iq.git
   cd segmentation-iq
   ```

2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run** the app locally:

   ```bash
   streamlit run ml.segmentation.iq.py
   ```

4. Open the URL shown in your terminal (e.g., [http://localhost:8501](http://localhost:8501)).

## 🏃‍♂️ Usage

1. **Upload** a CSV file with:
   - `Candidate_ID` column (unique identifier)
   - Numeric feature columns
2. **Configure** preprocessing options: imputation, log-transform, scaling, correlation and variance filtering.
3. **Explore** PCA results and choose a clustering model (K-Means, GMM, Hierarchical, DBSCAN).
4. **Visualize** clusters via PCA/t-SNE scatter plots and donut charts showing counts & percentages.
5. **Interpret** each candidate using distance & feature-contribution metrics.
6. **Save** multiple scenarios and **compare** them side-by-side.

## ☁️ Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Connect your repo and set `ml.segmentation.iq.py` as the entrypoint.
4. Click **Deploy**. Future pushes to `main` auto-update your app.

### Heroku (optional)

1. Create a Heroku app and add `HEROKU_API_KEY` to GitHub Secrets.
2. Use the provided GitHub Actions workflow for auto-deployment.

## 📄 Configuration

Optional `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4a90e2"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
font = "sans serif"
```

## 📦 Requirements

```txt
streamlit>=1.18.0
pandas
numpy
scikit-learn
scipy
plotly
matplotlib
ace_tools
```

> **Note:** `ace_tools` is required for local data previews in development.

## 🤝 Contributing

1. Fork this repo and create a feature branch.
2. Make your changes and submit a pull request.
3. Ensure compatibility and clean code standards.

---

*Segmentation IQ* • v1.0

