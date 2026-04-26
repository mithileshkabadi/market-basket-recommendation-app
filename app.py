import streamlit as st
import pandas as pd
import ast

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Smart Recommendation System", layout="wide")

# -------------------------
# Safe Parsing
# -------------------------
def safe_eval(val):
    try:
        if isinstance(val, str):
            return list(ast.literal_eval(val))
        return val
    except:
        return []

# -------------------------
# Load Rules
# -------------------------
@st.cache_data
def load_rules():
    rules = pd.read_csv("rules.csv")
    rules = rules.dropna(subset=['antecedents', 'consequents'])

    rules['antecedents'] = rules['antecedents'].apply(safe_eval)
    rules['consequents'] = rules['consequents'].apply(safe_eval)

    return rules

rules = load_rules()

# -------------------------
# Extract Products
# -------------------------
def get_all_products(rules):
    products = set()
    for sublist in rules['antecedents']:
        if isinstance(sublist, list):
            for item in sublist:
                if isinstance(item, str) and item.strip():
                    products.add(item)
    return sorted(products)

all_products = get_all_products(rules)

# -------------------------
# Trending Products
# -------------------------
def get_trending_products(rules, top_n=5):
    freq = {}
    for sublist in rules['antecedents']:
        for item in sublist:
            freq[item] = freq.get(item, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]

trending = get_trending_products(rules)

# -------------------------
# Recommendation Logic
# -------------------------
def recommend_products(input_items, rules, top_n=5):
    recommendations = []

    for _, row in rules.iterrows():
        if set(input_items).issubset(set(row['antecedents'])):
            for item in row['consequents']:
                recommendations.append((item, row['confidence'], row['lift'], row['antecedents']))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    seen = set(input_items)
    final = []

    for item, conf, lift, source in recommendations:
        if item not in seen:
            final.append((item, conf, lift, source))
            seen.add(item)

    return final[:top_n]

# -------------------------
# UI Layout
# -------------------------
st.title("🛒 Smart Product Recommendation System")
st.markdown("Get **AI-powered recommendations** based on shopping patterns")

col1, col2 = st.columns([2, 1])

# LEFT SIDE (MAIN)
with col1:
    selected_items = st.multiselect("🔍 Search & Select Products", all_products)

    if st.button("🚀 Get Recommendations"):
        if not selected_items:
            st.warning("⚠️ Please select at least one product")
        else:
            recs = recommend_products(selected_items, rules)

            if recs:
                st.success("✅ Recommended Products")

                for item, conf, lift, source in recs:
                    st.markdown(f"### 👉 {item}")

                    # Confidence bar
                    st.progress(min(conf, 1.0))

                    st.write(f"Confidence: **{round(conf,2)}** | Lift: **{round(lift,2)}**")

                    # Explanation
                    st.caption(f"💡 Because users who bought {', '.join(source)} also bought this")

                    st.markdown("---")

            else:
                st.error("❌ No strong recommendations found")

# RIGHT SIDE (INSIGHTS)
with col2:
    st.subheader("🔥 Trending Products")
    for item in trending:
        st.write(f"• {item}")

    st.markdown("---")

    st.subheader("ℹ️ About")
    st.write("""
    This system uses **Association Rule Mining (Apriori)** to:
    - Discover product relationships  
    - Recommend based on shopping behavior  
    - Optimize cross-selling strategies  
    """)

