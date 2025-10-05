import streamlit as st
import json
import numpy as np
import faiss
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load data and model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_data():
    with open("faiss_index/embedded_dataset_aligned.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array([d["embedding"] for d in data]).astype("float32")
    return data, embeddings

@st.cache_resource
def load_index():
    return faiss.read_index("faiss_index/equinet_faiss.index")

model = load_model()
data, embeddings = load_data()
index = load_index()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="EquiNet",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #1a1d24;
        color: white;
        border-radius: 10px;
    }
    .result-card {
        background-color: #161a22;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Filters")
source_filter = st.sidebar.multiselect(
    "Source Type", 
    options=list(set(d["source"] for d in data)),
    default=[]
)
group_filter = st.sidebar.radio(
    "Voice Type", ["All", "Underrepresented", "Mainstream"]
)
vis_toggle = st.sidebar.checkbox("Show Embedding Space Visualization", value=True)

# -------------------------------
# Main UI
# -------------------------------
st.title("🌍 EquiNet — Equalizing the Knowledge Graph")
st.write("Search across global and marginalized perspectives in sustainability, policy, and climate research.")

query = st.text_input("🔎 Enter your query:", placeholder="e.g., indigenous climate adaptation strategies in Asia")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), 5)
    results = [data[i] for i in I[0]]

    # Apply filters
    if source_filter:
        results = [r for r in results if r["source"] in source_filter]
    if group_filter != "All":
        results = [r for r in results if r["group"].lower() == group_filter.lower()]

    st.subheader("🧾 Top Results")
    for r in results:
        st.markdown(f"""
        <div class="result-card">
            <b>Source:</b> {r["source"]}<br>
            <b>Voice:</b> {r["group"].capitalize()}<br>
            <b>Excerpt:</b> {r["text"][:500]}...
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Visualization
    # -------------------------------
    if vis_toggle:
        st.subheader("🌐 Embedding Space Visualization")

        # Sample a small subset for plotting
        subset_size = 300
        idxs = np.random.choice(len(data), min(subset_size, len(data)), replace=False)
        subset_emb = embeddings[idxs]
        subset_labels = [data[i]["group"] for i in idxs]

        # Add query embedding
        emb_scaled = StandardScaler().fit_transform(np.vstack([subset_emb, query_vec]))
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(emb_scaled)
        
        subset_x, subset_y = reduced[:-1, 0], reduced[:-1, 1]
        query_x, query_y = reduced[-1, 0], reduced[-1, 1]

        df = {
            "x": subset_x,
            "y": subset_y,
            "Group": subset_labels
        }

        fig = px.scatter(df, x="x", y="y", color="Group", 
                         title="Embedding Space — Query vs Knowledge Voices",
                         color_discrete_map={
                             "underrepresented": "#00CC96",
                             "mainstream": "#636EFA"
                         },
                         opacity=0.7)
        fig.add_scatter(
            x=[query_x],
            y=[query_y],
            mode="markers+text",
            name="Query",
            marker=dict(size=16, color="#FFD700", symbol="star"),
            text=["Your Query"],
            textposition="top center"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter a query above to start exploring global knowledge diversity.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Yakshith • EquiNet (2025)")
