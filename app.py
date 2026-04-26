import streamlit as st
import pandas as pd
import ast

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Smart Recommendation System", layout="centered")

# -------------------------
# Safe Parsing Function
# -------------------------
def safe_eval(val):
    try:
        if isinstance(val, str):
            return list(ast.literal_eval(val))
        return val
    except:
        return []

# -------------------------
# Load Rules (Model)
# -------------------------
@st.cache_data
def load_rules():
    rules = pd.read_csv("rules.csv")

    # Drop bad rows
    rules = rules.dropna(subset=['antecedents', 'consequents'])

    # Convert string → list safely
    rules['antecedents'] = rules['antecedents'].apply(safe_eval)
    rules['consequents'] = rules['consequents'].apply(safe_eval)

    return rules

rules = load_rules()

# -------------------------
# Extract Product List
# -------------------------
def get_all_products(rules):
    products = set()

    for sublist in rules['antecedents']:
        if isinstance(sublist, list):
            for item in sublist:
                if isinstance(item, str) and item.strip() != "":
                    products.add(item)

    return sorted(products)

all_products = get_all_products(rules)

# -------------------------
# Recommendation Function
# -------------------------
def recommend_products(input_items, rules, top_n=5):
    recommendations = []

    for _, row in rules.iterrows():
        if set(input_items).issubset(set(row['antecedents'])):
            for item in row['consequents']:
                recommendations.append((item, row['confidence'], row['lift']))

    # Sort by confidence
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # Remove duplicates + already selected
    seen = set(input_items)
    final_recommendations = []

    for item, conf, lift in recommendations:
        if item not in seen:
            final_recommendations.append((item, conf, lift))
            seen.add(item)

    return final_recommendations[:top_n]

# -------------------------
# UI
# -------------------------
st.title("🛒 Smart Product Recommendation System")
st.markdown("Select products to get **AI-powered recommendations** based on shopping patterns.")

# Debug (REMOVE later if needed)
# st.write("Sample Data:", rules.head())
# st.write("Products:", all_products[:20])

# Dropdown
selected_items = st.multiselect("Select Products", all_products)

# Button
if st.button("Get Recommendations"):
    if not selected_items:
        st.warning("⚠️ Please select at least one product")
    else:
        recs = recommend_products(selected_items, rules)

        if recs:
            st.success("✅ Recommended Products:")
            for item, conf, lift in recs:
                st.write(f"👉 **{item}** | Confidence: {round(conf,2)} | Lift: {round(lift,2)}")
        else:
            st.error("❌ No strong recommendations found")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("Built using Association Rule Mining (Apriori) + Streamlit")
