import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google import genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()


# --- PDF Text Extraction ---
def extract_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# --- Chunking ---
def get_chunks(text):
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = textsplitter.split_text(text=text)
    return chunks

# --- Embeddings + FAISS Vector Store ---
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"])
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("vectorstore")

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",google_api_key=os.environ["GEMINI_API_KEY"])
    vector_store = FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )
    return vector_store

# --- QA Chain with Gemini ---
def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context below. 
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",google_api_key=os.environ["GEMINI_API_KEY"])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return model, prompt

# --- Streamlit App ---
st.set_page_config(page_title="PDF QA with Gemini", layout="wide")
st.title("📄 Chat with your PDF (Gemini + FAISS)")

# Sidebar - Upload PDFs
with st.sidebar:
    st.header("Upload your PDFs")
    pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs") and pdfs:
        with st.spinner("Processing PDFs..."):
            text = extract_text(pdfs)
            chunks = get_chunks(text)
            get_vector_store(chunks)
            st.success("✅ PDFs processed and vector store created!")

# Main chat section
query = st.text_input("Ask a question about your documents:")

if query:
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(query, k=3)

    # Join context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build QA chain
    model, prompt = get_conversational_chain()
    formatted_prompt = prompt.format(context=context, question=query)

    # Get Gemini response
    response = model.invoke(formatted_prompt)

    # Show retrieved docs
    with st.expander("📖 Retrieved Context"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content}")

    # Show final answer
    st.subheader("💡 Answer:")
    st.write(response.content)
