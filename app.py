import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Tech Stack Recommender", page_icon="🤖", layout="wide")

st.title("🎯 Project 3: AI Matchmaker Engine")
st.caption("DecodeLabs Industrial Training Kit — Content-Based Filtering Framework")
st.markdown("---")

# Load dataset safely
@st.cache_data
def load_data():
    try:
        return pd.read_csv('raw_skills.csv')
    except FileNotFoundError:
        st.error("❌ 'raw_skills.csv' not found. Please run the data generator first.")
        return None

df = load_data()

if df is not None:
    # Layout Split
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("### 📥 1. Ingestion Window (User State)")
        st.write("Enter your skills to discover your matching career path:")
        
        # Mandatory minimum 3 inputs to ensure vector data density
        skill_1 = st.text_input("Primary Technical Skill", placeholder="e.g., Python")
        skill_2 = st.text_input("Secondary Framework/Tool", placeholder="e.g., AWS")
        skill_3 = st.text_input("Operational Competency", placeholder="e.g., Automation")
        
        num_recommendations = st.slider("Select Top-N Recommendations", min_value=1, max_value=3, value=3)
        compute_trigger = st.button("🚀 Match My Profile")

    with col_out:
        st.markdown("### 📊 Predictive Engine Outputs")
        
        if compute_trigger:
            # Validate input density
            if not skill_1.strip() or not skill_2.strip() or not skill_3.strip():
                st.warning("⚠️ Ingestion rejected. You must provide at least 3 distinct skills to bypass a cold start.")
            else:
                with st.spinner("Executing similarity mathematics engine..."):
                    # Combine inputs into a single profile string
                    user_profile = f"{skill_1} {skill_2} {skill_3}"
                    
                    # 2. Vector Mapping & Feature Extraction via TF-IDF
                    vectorizer = TfidfVectorizer(stop_words='english')
                    
                    # Fit on item properties + user profile to ensure a shared vocabulary
                    all_texts = df['skills'].tolist() + [user_profile]
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    # Separate vectors
                    items_vectors = tfidf_matrix[:-1]
                    user_vector = tfidf_matrix[-1]
                    
                    # 3. Scoring via Angled Cosine Similarity
                    scores = cosine_similarity(user_vector, items_vectors).flatten()
                    
                    # Add scores to a copy of the dataframe
                    results_df = df.copy()
                    results_df['similarity_score'] = scores
                    
                    # 4. Sorting & Truncating (Filtering Top-N)
                    results_df = results_df.sort_values(by='similarity_score', ascending=False)
                    top_n_results = results_df.head(num_recommendations)
                    
                    # Display results cleanly
                    st.success("✨ Analysis complete! Your closest alignments:")
                    
                    for idx, row in top_n_results.iterrows():
                        match_percentage = row['similarity_score'] * 100
                        st.markdown(f"""
                        <div style="border-left: 4px solid #6366F1; padding: 10px; margin-bottom: 10px; background-color: rgba(99,102,241,0.05);">
                            <h4 style="margin:0;">🥇 Match: {row['role']}</h4>
                            <p style="margin:5px 0 0 0; font-size:14px; color:#818CF8;"><b>Match Metrics:</b> {match_percentage:.1f}% Angular Alignment</p>
                            <p style="margin:2px 0 0 0; font-size:12px; color:#A1A1AA;">Mapped Requirements: {row['skills']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("💡 Standby: Provide profile elements on the left to activate algorithmic content mapping.")
