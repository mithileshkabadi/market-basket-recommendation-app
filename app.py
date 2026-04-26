import streamlit as st
import pandas as pd
import ast

rules['antecedents'] = rules['antecedents'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

rules['consequents'] = rules['consequents'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

@st.cache_data
def load_rules():
    rules = pd.read_csv("rules.csv")
    return rules

    # Drop bad rows first
    rules = rules.dropna(subset=['antecedents', 'consequents'])

    # Safe conversion function
    def safe_eval(x):
        try:
            return list(ast.literal_eval(x))
        except:
            return []

    rules['antecedents'] = rules['antecedents'].apply(safe_eval)
    rules['consequents'] = rules['consequents'].apply(safe_eval)

    return rules

rules = load_rules()

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

# Extract all unique products
all_products = sorted(list(set([item for sublist in rules['antecedents'] for item in sublist])))

# User selection
selected_items = st.multiselect("Select Products", all_products)

# Button click
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
# Footer (Nice Touch)
# -------------------------
st.markdown("---")
st.markdown("Built using Association Rule Mining (Apriori) + Streamlit")
