Here’s a **clean and beginner-friendly README** for your project 🚀

---

# 📄 Chat with Your PDFs (Gemini + LangChain + FAISS)

This project is a **PDF Question Answering App** built with **Streamlit**, **Google Gemini API**, and **LangChain**.
It lets you upload PDF documents, process them into searchable chunks, and then ask natural language questions.
The app retrieves relevant sections from your documents and uses **Gemini LLM** to provide accurate answers.

---

## ⚡ Features

* 📤 **Upload PDFs** via a simple Streamlit interface
* ✂️ **Automatic text extraction & chunking** from PDFs
* 🧠 **Embeddings with Gemini** (`gemini-embedding-001`)
* 📚 **Vector search with FAISS** to find relevant chunks
* 🤖 **Gemini chat model** (`gemini-2.0-flash-001`) for answering questions
* 🔍 View the retrieved context before the answer (transparency)

---

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) – UI framework
* [PyPDF2](https://pypi.org/project/pypdf2/) – PDF text extraction
* [LangChain](https://www.langchain.com/) – text splitting, vector stores, prompt templates
* [Google Gemini API](https://ai.google.dev/) – embeddings + chat LLM
* [FAISS](https://faiss.ai/) – vector similarity search

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/chat-with-pdf-gemini.git
cd chat-with-pdf-gemini
```

### 2. Create a virtual environment

Using conda:

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key
```

> 🔑 You can get your API key from the [Google AI Studio](https://aistudio.google.com/).

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📖 Usage

1. Open the app in your browser (Streamlit will give you a local URL).
2. Upload one or more PDF files from the sidebar.
3. Click **Process PDFs** → the app extracts text, chunks it, and stores embeddings in FAISS.
4. Ask a question in the text box.
5. The app retrieves the top relevant chunks and sends them to **Gemini** for answering.
6. Expand the **📖 Retrieved Context** section to see the exact passages used.

---

## 🖼️ Example

* Upload: `machine_learning.pdf`
* Question: *"What is overfitting?"*
* Retrieved Context: shows chunks from the PDF explaining overfitting.
* Answer: *"Overfitting occurs when a model learns the training data too well, including noise, and fails to generalize to new data."*

---

## 🧩 Project Structure

```
.
├── app.py          # Main Streamlit app
├── requirements.txt
├── .env            # API key (not committed to git)
└── vectorstore/    # Saved FAISS database
```

---

## ⚠️ Notes

* Free Gemini API tier has **rate limits** → heavy use may hit quota.
* `FAISS` saves embeddings locally → delete `vectorstore/` to reset.
* This app answers **based only on your PDFs** (RAG pipeline).

---

## 📌 Roadmap

* ✅ Basic single-turn QA
* ⏳ Add **chat history** for multi-turn conversations
* ⏳ Support **other file types** (DOCX, TXT)
* ⏳ Deploy online (e.g., Streamlit Cloud, Render, or with ngrok)

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Google Gemini](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)

---

👉 With this, you’ve got a working **Chat with PDF using Gemini** app.

---
