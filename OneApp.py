# app_hybrid_rag.py
import os
import streamlit as st
import re
from typing import List, Dict

# LangChain + vectorstore + loaders
# 🔹 Loaders are now in langchain_community
from langchain_community.document_loaders import PyPDFLoader

# 🔹 Text splitters have their own package
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 🔹 Embeddings are in langchain_community (Hugging Face, Ollama, OpenAI, etc.)
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔹 Vector stores (like FAISS, Chroma, etc.)
from langchain_community.vectorstores import FAISS

# 🔹 Ollama and all LLMs are now in langchain_community.llms
from langchain_community.llms import Ollama


# web search fallback
from serpapi import GoogleSearch

# ---------------------------
# CONFIG
# ---------------------------
PDF_DIR = os.path.join(os.getcwd(), "pdfs")  # put your PDFs here
INDEX_DIR = os.path.join(os.getcwd(), "index_faiss")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMMA_MODEL = "gemma2:2b"   # Ollama model name
RETRIEVE_K = 7  # Increased default k value for more relevant chunks
CHUNK_SIZE = 1500  # Increased chunk size for more context per chunk
CHUNK_OVERLAP = 300  # Increased overlap to avoid splitting answers
RESTRICTED_KEYWORDS = [
    "suicide", "self harm", "kill myself", "illegal drug", "terror", "bomb", 
    "politics", "vote", "election", "religion", "sex", "child sexual"
]
# ---------------------------

st.set_page_config(page_title="Hybrid RAG Chat (Gemma2 + Ollama)", layout="wide")

st.title("DocuMind - By Rahul Gaba")
st.markdown(
    """
    This app indexes PDF files from a `pdfs/` folder, retrieves relevant chunks via FAISS,
    optionally searches the web (DuckDuckGo) for fresh info, and asks Gemma2 (Ollama) to answer.
    It also performs pre-filtering and LLM-based moderation to block restricted topics.
    """
)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

@st.cache_resource
def get_llm():
    # Uses Ollama local runtime — ensure ollama is installed and model is pulled
    return Ollama(model=GEMMA_MODEL)

def ensure_index(pdf_dir: str, index_dir: str):
    """
    Build or load FAISS index from pdf_dir. Rebuilds if no index exists.
    """
    if os.path.exists(index_dir) and os.path.isdir(index_dir) and os.listdir(index_dir):
        try:
            vectordb = FAISS.load_local(index_dir, get_embeddings(), allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            st.warning("Could not load existing index; rebuilding. Error: " + str(e))

    #Build index
    docs = []
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                path = os.path.join(root, fn)
                loader = PyPDFLoader(path)
                try:
                    loaded = loader.load()
                    # attach source metadata
                    for d in loaded:
                        d.metadata = d.metadata or {}
                        d.metadata["source"] = path
                    docs.extend(loaded)
                except Exception as e:
                    st.error(f"Failed to load {path}: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Remove non-printable / weird characters
        text = re.sub(r'[^A-Za-z0-9.,;:%\-\n() ]+', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Clean and filter chunks
    valid_chunks = []
    for c in chunks:
        if hasattr(c, 'page_content') and isinstance(c.page_content, str) and c.page_content.strip():
            c.page_content = clean_text(c.page_content)
            if c.page_content:
                valid_chunks.append(c)

    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(valid_chunks, embeddings)
    # persist
    vectordb.save_local(index_dir)
    return vectordb

def keyword_filter(q: str) -> bool:
    ql = q.lower()
    for kw in RESTRICTED_KEYWORDS:
        if kw in ql:
            return True
    return False

def llm_moderator(q: str, llm) -> bool:
    """
    Ask the LLM to classify whether the question should be blocked.
    Returns True if blocked, False if allowed.
    """
    mod_prompt = f"""
You are a safety classifier. If the user's question requests or solicits disallowed content,
such as instructions for illegal/violent activity, explicit sexual content involving minors,
medical diagnosis requests that could lead to harm, or targeted political persuasion,
reply with exactly: BLOCK
Otherwise reply with exactly: ALLOW

User question: \"\"\"{q}\"\"\"
"""
    try:
        resp = llm.invoke(mod_prompt)
        text = str(resp).strip().upper()
        if "BLOCK" in text and "ALLOW" not in text:
            return True
        # if uncertain, check for keywords fallback
        return False
    except Exception as e:
        # on any LLM failure, be conservative: block only if keyword filter triggered earlier
        st.warning(f"Moderator failure: {e}")
        return False

def web_search_snippets(query: str) -> list[dict]:
    try:
        # Initialize parameters
        params = {
            "engine": "google",           # search engine (google, bing, youtube, etc.)
            "q": query,                   # query entered by user
            "api_key": "28b6d15e47f8f82b6ac4c2a4a7027d89efeac940a3cb11685b0ed3afdcdafc71" # replace with your actual key
        }
        # Perform the search
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = []
        for r in results.get("organic_results", []):
            snippets.append({
                "title": r.get("title"),
                "snippet": r.get("snippet"),
                "url": r.get("link")
            })
        
        return snippets
    except Exception as e:
        # On failure, return empty list
        print(f"Web search failed: {e}")
        return []

def build_prompt(user_q: str, local_docs: List, web_snippets: List) -> str:
    """
    Create a system-like prompt instructing Gemma2 to use only the provided context.
    """
    ctx_parts = []
    if local_docs:
        ctx_parts.append("LOCAL DOCUMENT SNIPPETS (from your PDFs):\n")
        for i, d in enumerate(local_docs):
            source = d.metadata.get("source", "unknown")
            excerpt = d.page_content.strip().replace("\n", " ")[:1000]
            ctx_parts.append(f"[LOCAL {i+1}] Source: {source}\n{excerpt}\n")
    if web_snippets:
        ctx_parts.append("WEB SNIPPETS:\n")
        for i, s in enumerate(web_snippets):
            title = s.get("title") or ""
            snippet = (s.get("snippet") or "").replace("\n", " ")
            url = s.get("url") or ""
            ctx_parts.append(f"[WEB {i+1}] {title}\n{snippet}\nURL: {url}\n")

    context_text = "\n\n".join(ctx_parts) if ctx_parts else "No additional context available."

    prompt = f"""
You are a helpful assistant. Prefer to answer using the provided context below if it contains the answer.
If the answer is not found in the context, use your own general knowledge and say 'Based on my general knowledge:' before your answer.

Context:
{context_text}

User question: {user_q}

Answer:
"""
    return prompt

# ---------------------------
# UI: Left sidebar: indexing & settings
# ---------------------------
with st.sidebar:
    st.header("Index & Settings")
    st.write("Put your PDFs inside the `pdfs/` folder in this app directory.")
    st.write(f"PDF folder: `{PDF_DIR}`")
    if st.button("(Re)build index from PDF folder"):
        with st.spinner("Building index..."):
            os.makedirs(PDF_DIR, exist_ok=True)
            vect = ensure_index(PDF_DIR, INDEX_DIR)
            if vect:
                st.success("Index built and saved.")
            else:
                st.error("No PDFs found or index build failed.")
    st.markdown("---")
    st.write("Search settings")
    k = st.slider("Local retrieval (k)", min_value=1, max_value=20, value=RETRIEVE_K)
    use_web = st.checkbox("Enable web search (DuckDuckGo)", value=True)
    st.markdown("---")
    st.write("Safety")
    st.write("Restricted keywords (basic):")
    st.write(", ".join(RESTRICTED_KEYWORDS))
    st.markdown("---")
    st.write("Ollama model:")
    st.write(GEMMA_MODEL)
    st.markdown("---")
    if st.button("Test Ollama (model health)"):
        try:
            llm = get_llm()
            test = llm.invoke("Say hello")
            st.success("Ollama responded.")
            st.write(test)
        except Exception as e:
            st.error("Ollama connection failed: " + str(e))

# ---------------------------
# Main UI
# ---------------------------
st.subheader("Chat")
if "history" not in st.session_state:
    st.session_state.history = []

# Ensure index exists
vectordb = ensure_index(PDF_DIR, INDEX_DIR)
if vectordb is None:
    st.info("No indexed PDFs found. Put PDF files inside ./pdfs and click (Re)build index.")
    st.stop()

col1, col2 = st.columns([3,1])
with col1:
    user_q = st.text_input("Ask a question (will use local PDFs):")
with col2:
    if st.button("Send"):
        if not user_q or user_q.strip()=="":
            st.warning("Type a question first.")
        else:
            # 1) quick keyword filter
            if keyword_filter(user_q):
                st.error("This question matches restricted keywords and is blocked.")
            else:
                llm = get_llm()
                # 2) LLM-based moderation
                blocked = llm_moderator(user_q, llm)
                if blocked:
                    st.error("This question is blocked by the safety moderator.")
                else:
                    # 3) Retrieval: local
                    retriever = vectordb.as_retriever(search_kwargs={"k": k})
                    local_docs = retriever.invoke(user_q)

                    # # 4) Web search (optional)
                    web_snips = []
                    if use_web:
                      web_snips = web_search_snippets(user_q)

                    # 5) Compose prompt and call LLM
                    prompt = build_prompt(user_q, local_docs, web_snips)
                    with st.spinner("Thinking..."):
                        try:
                            resp = llm.invoke(prompt)
                            answer = str(resp)
                        except Exception as e:
                            answer = f"LLM error: {e}"

                    # 6) record and display
                    st.session_state.history.append({
                        "question": user_q,
                        "answer": answer,
                        "local_docs": local_docs,
                        "web_snips": web_snips
                    })

# show chat history
for entry in reversed(st.session_state.history[-10:]):
    st.markdown(f"**Q:** {entry['question']}")
    st.markdown(f"**A:** {entry['answer']}")
    # show top sources
    if entry["local_docs"]:
        st.markdown("**Local sources used:**")
        for i, d in enumerate(entry["local_docs"][:3], 1):
            src = d.metadata.get("source","unknown")
            snippet = d.page_content[:300].replace("\n"," ")
            st.markdown(f"- [LOCAL {i}] `{src}` — {snippet}...")
    if entry["web_snips"]:
        st.markdown("**Web results used:**")
        for i, s in enumerate(entry["web_snips"][:3], 1):
            st.markdown(f"- [WEB {i}] {s.get('title')} — {s.get('url')}")

st.markdown("---")
st.caption("Hybrid RAG demo — Model runs via Ollama local server. All local PDFs never leave your machine.")
