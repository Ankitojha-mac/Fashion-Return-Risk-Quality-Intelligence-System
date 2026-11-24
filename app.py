import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fashion Return-Risk & Quality Intelligence",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #282C3F;
        }
        [data-testid="stSidebar"] {
            background-color: #F5F5F6;
            border-right: 1px solid #EAEAEC;
        }
        h1, h2, h3, .stMarkdown, label, p {
            color: #282C3F !important;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #EAEAEC;
            border-left: 4px solid #FF3F6C;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricValue"] {
            color: #FF3F6C !important;
        }
        [data-testid="stMetricLabel"] {
            color: #535766 !important;
        }
        .stSlider [data-baseweb="slider"] div[role="slider"] {
            background-color: #FF3F6C;
        }
        .dev-panel {
            background-color: #FFFFFF;
            border: 1px solid #EAEAEC;
            border-left: 4px solid #FF3F6C;
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.9rem;
        }
        .dev-panel a {
            color: #FF3F6C !important;
            text-decoration: none;
            font-weight: 600;
        }
        .dev-panel a:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
  catalog = pd.read_csv("scored_catalog_risk.csv")
  complaints = pd.read_csv("complaint_clusters_summary.csv")
  return catalog, complaints


@st.cache_resource
def load_model():
  return joblib.load("fashion_risk_xgb_model.joblib")


catalog_df, complaints_df = load_data()
model = load_model()

st.title("Fashion Return-Risk & Quality Intelligence System")
st.caption(
    "Hybrid NLP complaint clustering + tabular risk scoring for catalog quality"
    " & return-risk prioritization"
)

st.sidebar.header("Filter Catalog")

brand_search = st.sidebar.text_input(
    "Search Brand by Name",
    placeholder="e.g., Mufti, FUGAZEE, Kotty...",
).strip()

brands = ["All"] + sorted(
    catalog_df["brand_name"].dropna().unique().tolist()
)
selected_brand = st.sidebar.selectbox("Or Select Brand", brands)

tiers = ["All"] + ["High Risk", "Medium Risk", "Low Risk"]
selected_tier = st.sidebar.selectbox("Risk Tier", tiers)

max_price = float(catalog_df["price"].dropna().max())
min_price = float(catalog_df["price"].dropna().min())
price_range = st.sidebar.slider(
    "Price Range (₹)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="dev-panel">
        <div style="font-weight: 700; margin-bottom: 6px;">Developer</div>
        <div><strong>Ankit Kumar Ojha</strong></div>
        <div>Email: <a href="mailto:ankitojha118@gmail.com">ankitojha118@gmail.com</a></div>
        <div style="margin-top: 4px;">
            <a href="https://www.linkedin.com/in/ankit-kumar-ojha-94835b323?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank">
                LinkedIn Profile ↗
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

filtered_df = catalog_df[
    (catalog_df["price"].fillna(0) >= price_range[0])
    & (catalog_df["price"].fillna(0) <= price_range[1])
]

if brand_search:
  filtered_df = filtered_df[
      filtered_df["brand_name"]
      .fillna("")
      .str.contains(brand_search, case=False, na=False)
  ]
elif selected_brand != "All":
  filtered_df = filtered_df[filtered_df["brand_name"] == selected_brand]

if selected_tier != "All":
  filtered_df = filtered_df[filtered_df["risk_tier"] == selected_tier]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Items ", f"{len(filtered_df):,}")
col2.metric(
    "High Risk Items",
    f"{(filtered_df['risk_tier'] == 'High Risk').sum():,}",
)
col3.metric(
    "Avg Risk Score",
    f"{filtered_df['predicted_risk_prob'].mean() * 100:.1f}%"
    if len(filtered_df) > 0
    else "0.0%",
)
col4.metric(
    "Avg Discount",
    f"{filtered_df['discount_pct_clean'].mean():.1f}%"
    if len(filtered_df) > 0
    else "0.0%",
)

st.subheader("Risk Tier & Complaint Breakdown")
left_col, right_col = st.columns(2)

with left_col:
  st.markdown("**Catalog Risk Tier Counts**")
  tier_counts = (
      filtered_df["risk_tier"]
      .value_counts()
      .reindex(["High Risk", "Medium Risk", "Low Risk"])
      .fillna(0)
  )
  st.bar_chart(tier_counts, color="#FF3F6C")

with right_col:
  st.markdown("**Root-Cause Review Complaints**")
  comp_counts = complaints_df["complaint_category"].value_counts()
  st.bar_chart(comp_counts, color="#FF3F6C")

st.subheader("Brand Risk Leaderboard (Min 30 Items)")
brand_table = (
    filtered_df.groupby("brand_name")
    .agg(
        items=("predicted_risk_prob", "count"),
        avg_risk_pct=("predicted_risk_prob", lambda s: round(s.mean() * 100, 2)),
        avg_discount_pct=("discount_pct_clean", lambda s: round(s.mean(), 1)),
        avg_rating=("ratings", lambda s: round(s.mean(), 2)),
    )
    .query("items >= 30")
    .sort_values("avg_risk_pct", ascending=False)
    .head(15)
)
st.dataframe(brand_table, use_container_width=True)

st.subheader("Top High-Risk Catalog Items")
display_cols = [
    "brand_name",
    "pants_description",
    "price",
    "MRP",
    "discount_pct_clean",
    "ratings",
    "predicted_risk_prob",
    "risk_tier",
    "primary_complaint_tag",
]
top_items = (
    filtered_df.sort_values("predicted_risk_prob", ascending=False)
    .drop_duplicates(subset=["brand_name", "pants_description", "price", "MRP"])[
        display_cols
    ]
    .head(25)
)
st.dataframe(top_items, use_container_width=True)
