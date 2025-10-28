from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1. Load files
loaders = [
    PyPDFLoader("data/clinical_guidelines.pdf"),
    TextLoader("data/finance_notes.txt"),
    JSONLoader("data/enzyme_data.json", jq_schema=".records[]")  # optional jq filter
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# Step 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Step 3. Create embeddings
embeddings = OllamaEmbeddings(model="gemma2:2b")

# Step 4. Build FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("db/faiss_index")
