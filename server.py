import streamlit as st
import time
import json
import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Setup & Page Config ---
st.set_page_config(page_title="WWII Hybrid RAG Explorer", layout="wide")
st.title("🪖 WWII Hybrid RAG Explorer")
st.markdown("Retrieval-Augmented Generation using **Dense (FAISS)** + **Sparse (BM25)** with **RRF**.")

# --- Cache Models and Indices ---
@st.cache_resource
def load_resources():
    # Load Metadata
    with open('ww2_chunks.json', 'r') as f:
        chunks = json.load(f)
    
    # Load Dense Search
    dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    dense_index = faiss.read_index("ww2_index.faiss")
    
    # Load Sparse Search
    with open('ww2_bm25.pkl', 'rb') as f:
        bm25_model = pickle.load(f)
        
    # Load LLM (Flan-T5)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return chunks, dense_model, dense_index, bm25_model, tokenizer, llm

chunks_data, dense_model, dense_index, bm25_index, tokenizer, llm = load_resources()

# --- Retrieval & RRF Logic ---
def hybrid_search(query, k=50, rrf_k=60):
    # 1. Dense (FAISS)
    q_vec = dense_model.encode([query])
    faiss.normalize_L2(q_vec)
    d_scores, d_indices = dense_index.search(q_vec, k)
    
    # 2. Sparse (BM25)
    tokenized_query = query.lower().split()
    s_scores = bm25_index.get_scores(tokenized_query)
    s_indices = np.argsort(s_scores)[::-1][:k]
    
    # Build RRF Rankings
    dense_ranks = {idx: r + 1 for r, idx in enumerate(d_indices[0])}
    sparse_ranks = {idx: r + 1 for r, idx in enumerate(s_indices)}
    
    all_indices = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    rrf_results = []
    
    for idx in all_indices:
        d_rank = dense_ranks.get(idx, 1e6)
        s_rank = sparse_ranks.get(idx, 1e6)
        
        # Calculate RRF Score
        rrf_score = (1 / (rrf_k + d_rank)) + (1 / (rrf_k + s_rank))
        
        rrf_results.append({
            "idx": idx,
            "rrf_score": rrf_score,
            "dense_rank": d_rank if d_rank < 1e5 else "N/A",
            "sparse_rank": s_rank if s_rank < 1e5 else "N/A",
            "dense_score": float(d_scores[0][list(d_indices[0]).index(idx)]) if idx in d_indices[0] else 0.0,
            "sparse_score": float(s_scores[idx])
        })
    
    # Sort by RRF
    rrf_results.sort(key=lambda x: x['rrf_score'], reverse=True)
    return rrf_results[:5]

# --- UI Components ---
query = st.text_input("Enter your WWII question:", placeholder="e.g., What was the significance of the Battle of Midway?")

if query:
    start_time = time.time()
    
    # 1. Retrieve
    with st.spinner("Searching and Generating..."):
        retrieved_meta = hybrid_search(query)
        context = "\n".join([chunks_data[r['idx']]['content'] for r in retrieved_meta])
        
        # 2. Generate
        prompt = f"Answer using context and provide your chain of thought: \nQuestion: {query}\nContext: {context}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = llm.generate(**inputs, max_new_tokens=200,temperature=0.7, top_p=0.9, do_sample=True )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    elapsed = end_time - start_time

    # --- Display Results ---
    st.subheader("💡 Generated Answer")
    st.success(answer)
    st.caption(f"⏱️ Response Time: {elapsed:.2f} seconds")

    st.divider()

    st.subheader("🔍 Retrieval Metrics & Sources")
    
    # Create a table for the scores
    table_data = []
    for r in retrieved_meta:
        chunk = chunks_data[r['idx']]
        table_data.append({
            "Title": chunk['title'],
            "RRF Score": f"{r['rrf_score']:.5f}",
            "Dense Score (Cos)": f"{r['dense_score']:.3f}",
            "Sparse Score (BM25)": f"{r['sparse_score']:.2f}",
            "Dense Rank": r['dense_rank'],
            "Sparse Rank": r['sparse_rank']
        })
    
    st.table(table_data)

    # Show actual content in expanders
    for i, r in enumerate(retrieved_meta):
        chunk = chunks_data[r['idx']]
        with st.expander(f"Source {i+1}: {chunk['title']}"):
            st.write(chunk['content'])
            st.markdown(f"**URL:** [{chunk['url']}]({chunk['url']})")