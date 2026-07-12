# 🤖 Project 3: AI Recommendation Logic — Tech Stack Recommender

## 📌 Project Overview
Welcome to the implementation node for **Project 3: AI Recommendation Logic**, powered by the DecodeLabs Advanced Analytics Stack. 

This project marks a fundamental engineering shift from **Passive Classification** (used in Phase 2) to **Active Prediction**. Instead of sorting structured metadata after the fact, this digital matchmaker processes incoming raw user states and bridges the gap to high-value relevant items before needs are explicitly articulated. This content-based filtering system bypasses standard information overload by converting qualitative engineering skill paths into objective, ranked mathematical vector alignments.

---

## 🏗️ Architecture: The 4-Step Assembly Line
The recommendation engine enforces a strict **Input-Process-Output (IPO)** framework designed to combat choice overload:

1. **Ingestion (User State):** Accepts a minimum vector density of three structural skill inputs from the user interface to safeguard the pipeline against the classic "User Cold Start" bottleneck[cite: 1].
2. **Scoring (Similarity Logic):** Transforms the qualitative user parameters alongside database items (`raw_skills.csv`) into weighted vector elements using a statistical **TF-IDF Weighting Matrix**[cite: 1]. It then evaluates their orientation in multidimensional space utilizing magnitude-invariant **Cosine Similarity**[cite: 1].
3. **Sorting:** Organizes the resulting database nodes sequentially in descending order based on their angular similarity percentage[cite: 1].
4. **Filtering (Top-N List):** Truncates the live processing stream to display only the Top 3 highest-aligned career path outcomes[cite: 1].

---

## 🛠️ Tech Stack & Dependencies
The framework utilizes pure Python engineering logic coupled with lean, production-grade statistical mapping tools. The configuration relies on the exact versions outlined in `requirements.txt`:

* **Streamlit** (v1.35.0) — Frontend Architecture & UI Hub
* **Pandas** (v2.2.2) — Structural Data Ingestion & Manipulation
* **Numpy** (v1.26.4) — Array Element Math Operations
* **Scikit-Learn** (v1.5.0) — TF-IDF Matrix Vectorization & Cosine Similarity Engines

---

## 🚀 Execution Guide (Google Colab Setup)

Follow these step-by-step terminal and cell instructions inside your notebook node to initiate the live server telemetry:
### Live Demo Website
https://raw-skills-data-project3-decodes-lab-i5qshdwxzyjzdafb65vdmp.streamlit.app/
### 1. Install Workspace Dependencies
Execute the pip pipeline command to clear requirements block setups:
```bash
!pip install streamlit pandas numpy scikit-learn


