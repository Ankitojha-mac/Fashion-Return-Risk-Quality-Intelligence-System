# Fashion-Return-Risk-Quality-Intelligence-System

Yahan aapke GitHub repository ke liye ek professional, clean aur complete `README.md` file ready hai. Isme saare technical details, architecture, metrics aur setup steps shamil hain.

Bas neeche diye gaye text ko copy karke apne project folder mein `README.md` naam se save kar lein (aur zaroorat ho toh screenshots add kar sakte hain):

```markdown
# 🛍️ Fashion Return-Risk & Quality Intelligence System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-MiniLM%20%2B%20KMeans-orange?style=for-the-badge)

An end-to-end data science and business intelligence application designed to proactively identify high return-risk fashion products and uncover underlying customer quality friction. By combining **unsupervised NLP complaint clustering** with **supervised gradient-boosted risk scoring**, this tool empowers merchandising and QA teams to make data-driven catalog decisions.

---

## 🎯 Problem Statement

In fashion e-commerce, high return rates and post-purchase dissatisfaction severely impact profit margins and increase reverse-logistics costs. Sizing inconsistencies, fit mismatches, fabric flaws, and aggressive clearance discounting are major drivers of return friction. This system bridges the gap between customer review sentiment and catalog-level merchandising attributes to flag risky SKUs before they cause large-scale return bottlenecks.

---

## 🏗️ System Architecture & Pipeline

```text
  [1. DATA INGESTION & CLEANING]
   ├── Myntra Catalog Data (52,120 SKUs) — pricing, discounts, ratings & attributes
   └── Women's E-Commerce Reviews (23,486 rows) — text reviews & sentiment filtering
          │                                                │
          ▼                                                ▼
  [2. NLP COMPLAINT PIPELINE]                   [3. TABULAR FEATURE ENGINEERING]
   ├── Sentence Embeddings (MiniLM)                  ├── Price spread & clean discount %
   ├── Unsupervised Clustering (KMeans, k=6)         ├── Fit & silhouette flags (Slim, Relaxed, Denim)
   └── TF-IDF Root-Cause Labeling                    └── Brand catalog volume & non-leaky risk proxy
          │                                                │
          └───────────────────────┬────────────────────────┘
                                  ▼
  [4. HYBRID RISK MODELING & SEGMENTATION]
   ├── XGBoost Risk Classifier (0.7936 Accuracy | 0.8438 ROC-AUC)
   ├── Risk Tier Segmentation (Low Risk | Medium Risk | High Risk)
   └── Root-Cause Complaint Tagging per SKU
          │
          ▼
  [5. INTERACTIVE STREAMLIT DASHBOARD]
   └── Live filtering, KPI cards, risk charts, brand leaderboard & deduplicated SKU explorer

```

---

## 📊 Key Results & Business Insights

* **Model Performance:** Trained on non-leaky catalog and NLP-mapped features, the XGBoost model achieved **79.36% Accuracy** and **0.8438 ROC-AUC** on held-out test data.
* **Risk Segmentation:** Across **52,120 catalog SKUs**, roughly **4,963 items (~9.5%)** were flagged as **High Risk**, with an overall catalog average risk score of **29.9%** and an average discount of **46.7%**.
* **Primary Risk Drivers:** Heavy discounting (`discount_pct_clean`), relaxed/slim silhouette flags (`is_relaxed_loose`, `is_slim_fit`), price spreads, brand volume, and NLP sizing friction signals emerged as the strongest predictors of return risk.
* **Root-Cause Review Clusters:** Unsupervised clustering identified 6 distinct customer friction areas, led by *Dress Fit & Expectation Mismatch*, *Tops/Sweaters Fit*, *Size Scaling*, and *Bottoms Quality/Proportion Fit*.

---

## ✨ Dashboard Features

* **Myntra-Inspired UI:** Clean white (`#FFFFFF`) canvas with Myntra Pink (`#FF3F6C`) highlights and Charcoal (`#282C3F`) typography for an authentic retail analytics experience.
* **Interactive Sidebar Filters:** Instant case-insensitive **Brand Search**, Brand Dropdown, Risk Tier filter (`High`, `Medium`, `Low`), and Price Range (`₹`) slider.
* **Executive KPIs:** Real-time metrics for *Filtered SKUs*, *High Risk Items*, *Avg Risk Score*, and *Avg Discount*.
* **Visual Analytics:** Side-by-side distribution charts for Risk Tiers and NLP Complaint Categories.
* **Actionable Tables:** A **Brand Risk Leaderboard** (filtered by minimum catalog size) and a **Deduplicated Top High-Risk Catalog Items** table mapped to primary complaint tags.

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml                    # Custom Myntra light theme configuration
├── app.py                             # Main Streamlit dashboard application
├── scored_catalog_risk.csv            # Enriched catalog dataset with risk probabilities & tags
├── complaint_clusters_summary.csv     # NLP review cluster assignments & category labels
├── fashion_risk_xgb_model.joblib      # Serialized XGBoost risk classification model
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation

```

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository

```bash
git clone [https://github.com/Ankitojha-mac/fashion-return-risk-intelligence.git](https://github.com/Ankitojha-mac/fashion-return-risk-intelligence.git)
cd fashion-return-risk-intelligence

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Streamlit App

```bash
streamlit run app.py

```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📦 `requirements.txt`

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib

```

---

## 👨‍💻 Author & Contact

**Ankit Kumar Ojha**

* **Email:** [ankitojha118@gmail.com](https://www.google.com/search?q=mailto%3Aankitojha118%40gmail.com)
* **LinkedIn:** [linkedin.com/in/ankit-kumar-ojha-94835b323](https://www.google.com/search?q=https://www.linkedin.com/in/ankit-kumar-ojha-94835b323)
* **GitHub:** [github.com/Ankitojha-mac](https://www.google.com/search?q=https://github.com/Ankitojha-mac)

```

---



```
