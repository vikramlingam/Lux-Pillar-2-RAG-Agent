import streamlit as st
import sympy as sp
import os
import fitz  # for reading PDFs
import faiss
import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

# 1. Page Config
st.set_page_config(page_title="Lux Pillar 2 RAG Agent", layout="wide")

# 2. Custom CSS for a professional look
st.markdown("""
    <style>
    /* Make the main title smaller and cleaner */
    h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    /* Make the subheader cleaner */
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
    }
    /* Make the base text a bit larger for readability */
    .stApp {
        font-size: 1.05rem;
    }
    /* Clean up the chat input box */
    .stChatInputContainer {
        border-top: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Title and Intro
st.title("Lux Pillar 2 RAG Agent")
st.write("""
This tool is an AI-powered advisor trained on Luxembourg's Pillar 2 and ATAD legislation. 
Ask it technical questions to get cited answers from the source documents.
""")

# 4. "How this works" expander
with st.expander("How this works"):
    st.write("""
    This agent uses a **Retrieval-Augmented Generation (RAG)** process:
    1.  **Load:** When the app starts, it reads and processes 8 expert documents (PDFs) from the `data` folder.
    2.  **Embed & Index:** The text from these documents is converted into numerical vectors (embeddings) and stored in a local FAISS vector database. This is the agent's "brain".
    3.  **Retrieve:** When you ask a question, the agent embeds your query and searches the database to find the most relevant text chunks from the documents.
    4.  **Generate:** It then feeds your question *and* the relevant text chunks to the Gemini model, instructing it to answer *only* using the provided information and to cite its sources.
    """)

# 5. The RAG "Brain"

# 5.1. Configure API Key
try:
    # Get the API key from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("GEMINI_API_KEY not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

# Define our embedding model
embedding_model = "models/embedding-001"

# 5.2. PDF Text Extraction Function
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF, page by page, with metadata."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text:  # Only add pages with text
            chunks.append({
                "source": os.path.basename(pdf_path),
                "page": page_num + 1,
                "text": text
            })
    return chunks

# 5.3. Master RAG Loader Function (Cached)
@st.cache_resource
def load_rag_core():
    """
    Loads all data, embeds it, and creates a FAISS index.
    This function is cached to run only once.
    """
    data_folder = "data"
    all_chunks = []

    # 1. Load and Chunk
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        all_chunks.extend(extract_text_from_pdf(pdf_path))

    # 2. Embed
    texts_to_embed = [chunk['text'] for chunk in all_chunks]

    # Embed in batches to respect API limits
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        try:
            # Use RETRIEVAL_DOCUMENT for indexing
            result = genai.embed_content(
                model=embedding_model,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT"
            )
            all_embeddings.extend(result['embedding'])
        except (GoogleAPIError, ValueError) as e:
            st.error(f"Error embedding batch {i//batch_size}: {e}")
            continue # Skip bad batches

    # 3. Index (FAISS)
    embeddings_np = np.array(all_embeddings).astype("float32")
    d = embeddings_np.shape[1] 
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)

    # Return the index and the data chunks
    return index, all_chunks

# 5.4. Run the Loader and Display Status
st.subheader("AI Tax Advisor")

with st.spinner("Processing documents, embedding text, and building vector index... (This runs only once on first load)"):
    try:
        vector_index, text_chunks = load_rag_core()
        st.success(f"Success! Processed {len(text_chunks)} text chunks from {len(os.listdir('data'))} documents. The agent is ready.")
    except Exception as e:
        st.error(f"An error occurred while building the RAG core: {e}")

# 6. The Chat Interface

# model for chat
chat_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

def find_relevant_chunks(query, index, chunks, k=3):
    """
    Finds the top-k most relevant text chunks from the vector index.
    """
    try:
        # embed the user query
        query_embedding_result = genai.embed_content(
            model=embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array([query_embedding_result['embedding']]).astype("float32")

        # search the FAISS index
        distances, indices = index.search(query_embedding, k)
        
        # get the chunks
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks

    except (GoogleAPIError, ValueError) as e:
        st.error(f"Error finding relevant chunks: {e}")
        return []

def generate_response(query, context_chunks):
    """
    Generates a response from the LLM based on the query and context.
    """
    # Create the augmented prompt
    context_text = "\n\n".join([f"Source: {chunk['source']} (Page {chunk['page']})\nText: {chunk['text']}" for chunk in context_chunks])

    system_prompt = f"""
    You are an expert Tax Advisor for Luxembourg. Your answers must be professional, accurate, and based *only* on the provided sources.
    
    **CRITICAL RULE 1 (Flowcharts):** If the user asks for a company structure, flowchart, or ownership diagram, you MUST respond *only* with valid Mermaid.js syntax. For example:
    graph TD;
        A[Parent Co] --> B[Lux HoldCo];
        B --> C[OpCo 1];
    
    **CRITICAL RULE 2 (RAG):** For all other questions, you MUST use ONLY the following context to answer the user's question. 
    * Cite your sources clearly using the (Source: [filename], Page [number]) format. 
    * If the answer is not found in the provided context, you MUST state: "I cannot answer that based on the provided documents."
    * Do not make up information.

    --- BEGIN CONTEXT ---
    {context_text}
    --- END CONTEXT ---
    
    **User's Question:** {query}
    """

    try:
        # send to gemini
        response = chat_model.generate_content(system_prompt)
        return response.text

    except (GoogleAPIError, ValueError) as e:
        st.error(f"Error generating response: {e}")
        return "An error occurred while generating the response."

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # check if it's a mermaid chart
        if message["content"].strip().startswith("graph TD") or message["content"].strip().startswith("graph LR"):
            st.markdown(f"```mermaid\n{message['content']}\n```")
        else:
            st.markdown(message["content"])


# --- Handle New Chat Input ---
if prompt := st.chat_input("Ask a question about Pillar 2, ATAD, or a company structure..."):

    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show a spinner while thinking
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # 3. Find relevant documents (RAG)
            if "structure" in prompt.lower() or "flowchart" in prompt.lower() or "diagram" in prompt.lower():
                # mermaid request, skip RAG
                relevant_chunks = [] 
            else:
                # normal RAG query
                relevant_chunks = find_relevant_chunks(prompt, vector_index, text_chunks)

            # 4. Generate the response
            response_text = generate_response(prompt, relevant_chunks)

            # 5. Display the response
            if response_text.strip().startswith("graph TD") or response_text.strip().startswith("graph LR"):
                # it's a flowchart, wrap it to render
                st.markdown(f"```mermaid\n{response_text}\n```")
            else:
                # normal RAG answer
                st.markdown(response_text)

    # 6. Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})