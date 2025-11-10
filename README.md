# Local RAG Agent For Luxembourg Pillar 2 and ATAD Legislation.

A local Retrieval-Augmented Generation (RAG) agent that answers questions using your own PDF documents as its knowledge base. I tried on  Luxembourg Pillar 2 and ATAD Legislation.  
It provides **source citations**, generates **Mermaid.js flowcharts**, and ensures all data remains **fully local and secure**.

---

## ✨ Features

- **Citations**  
  Every answer includes the exact source document and page number.

- **Flowchart Generation**  
  Automatically generates **Mermaid.js** diagrams (useful for company structures or process flows).

- **Local & Secure**  
  All your PDFs, embeddings, and API keys are stored **locally** and never sent anywhere else.

---

## ⚙️ How It Works

This agent uses a **RAG (Retrieval-Augmented Generation)** pipeline:

1. **Load**  
   On startup, the app reads and processes all PDF documents located in the `/data` folder.

2. **Embed & Index**  
   The text from PDFs is chunked and converted into numerical embeddings using the **Google Gemini API**.  
   These embeddings are stored in a local **FAISS** vector database.

3. **Retrieve**  
   When a user asks a question, the system searches the FAISS database for the most relevant text chunks.

4. **Generate**  
   The question and retrieved text are sent to the **Gemini LLM**, which generates an answer strictly based on the provided sources.

---

## 🧩 Prerequisites

Before running the app, make sure you have these installed:

- Python **3.10+**
- Git

---

## 🚀 Local Setup & Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPONAME.git
cd YOUR_REPONAME
```

---

### 2. Create and Activate a Virtual Environment

It’s best to use a virtual environment to isolate dependencies.

```bash
# Create the environment
python3 -m venv venv

# Activate the environment (Mac/Linux)
source venv/bin/activate

# (or) Activate the environment (Windows)
.env\Scripts\activate
```

---

### 3. Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

### 4. Set Up Your API Key

You’ll need a **Google Gemini API key** for embeddings and responses.

Create a folder named `.streamlit` in the project root:

```bash
mkdir .streamlit
```

Then create a new file named `secrets.toml` inside it:

```bash
touch .streamlit/secrets.toml
```

Open `secrets.toml` and add your API key:

```toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

---

### 5. Add Your Knowledge Base (The “Brain”)

The agent learns from whatever PDFs you provide.

1. Create a folder named `data`:
   ```bash
   mkdir data
   ```
2. Add all your PDF documents (like OECD commentary, tax firm reports, etc.) into the `/data` folder.

The app automatically detects and reads every `.pdf` file placed here.

---

## ▶️ Running the Application

Once setup is complete, start the app:

```bash
streamlit run app.py
```

This will open the app automatically at:  
👉 **http://localhost:8501**

> ⚡ Note: On first launch, it may take 30–60 seconds to process and embed all documents.  
> Once done, the “brain” is cached and responses become nearly instant.

---

## 🧠 Summary

- Local RAG pipeline with **Google Gemini**  
- **PDF-based** knowledge base  
- **FAISS** for fast semantic retrieval  
- **Streamlit** frontend for easy interaction  
- 100% local data control  

---

## 🪄 Example Use Cases

- Legal and tax research assistants  
- Internal policy Q&A systems  
- Private company document search  
- Knowledge management tools  

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **FAISS**
- **Google Gemini API**
- **PyPDF**
- **LangChain**

---

## 📜 License

This project is open-source.  
Feel free to use, modify, and extend it for your own local RAG workflows.

---
