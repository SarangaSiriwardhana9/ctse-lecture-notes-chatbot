import os
import requests
import glob
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

# Set page config
st.set_page_config(page_title="CTSE Lecture Notes Chatbot", page_icon="📚", layout="wide")

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBzS_RgPK9r-ZAWFndoDkm6TunuIpRRSlA"

# Step 1: Load documents from the 'data' folder
@st.cache_resource
def load_documents(directory='./data'):
    documents = []
    
    # Process text files
    for file_path in glob.glob(f"{directory}/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append({
                "content": content,
                "source": file_path
            })
    
    # Process PDF files
    for file_path in glob.glob(f"{directory}/*.pdf"):
        try:
            content = extract_text_from_pdf(file_path)
            documents.append({
                "content": content,
                "source": file_path
            })
        except Exception as e:
            st.error(f"Error extracting text from {file_path}: {str(e)}")
    
    return documents

# Helper function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    # Clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 2: Split documents into chunks
@st.cache_resource
def split_into_chunks(documents, chunk_size=800, overlap=150):
    chunks = []
    for doc in documents:
        content = doc["content"]
        source = doc["source"]
        
        # Simple sliding window approach
        for i in range(0, len(content), chunk_size - overlap):
            chunk_text = content[i:i + chunk_size]
            if len(chunk_text) < 100:  # Skip very small chunks
                continue
            chunks.append({
                "content": chunk_text,
                "source": source
            })
    
    return chunks

# Step 3: Create embeddings
@st.cache_resource
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk["content"])
        embeddings.append(embedding)
    
    return model, embeddings

# Step 4: Simple retrieval function
def retrieve_relevant_chunks(query, model, chunks, embeddings, k=3):
    # Get query embedding
    query_embedding = model.encode(query)
    
    # Calculate similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    return [chunks[i] for i in top_indices]

# Step 5: Function to call Google Gemini API
def ask_gemini(prompt):
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts":[{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            if "candidates" in response_json and len(response_json["candidates"]) > 0:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "No valid response found in API response"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

# Step 6: Main QA function
def answer_question(question, model, chunks, embeddings):
    # Get relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, model, chunks, embeddings)
    
    # Create prompt for Gemini
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    prompt = f"""Answer the following question based ONLY on the information provided in the context below.
    If the answer is not found in the context, say "I don't have enough information to answer this question."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:"""
    
    # Get answer from Gemini
    answer = ask_gemini(prompt)
    
    return answer, relevant_chunks

# Main Streamlit App
def main():
    st.title("📚 CTSE Lecture Notes Chatbot")
    st.markdown("---")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Loading data with progress indicator
    if "data_loaded" not in st.session_state:
        with st.spinner("Loading documents and creating embeddings..."):
            # Load and process documents
            documents = load_documents()
            st.session_state.document_count = len(documents)
            
            chunks = split_into_chunks(documents, chunk_size=800)
            st.session_state.chunk_count = len(chunks)
            
            model, embeddings = create_embeddings(chunks)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.data_loaded = True
    
    # Show document stats in sidebar
    st.sidebar.title("📊 Document Stats")
    st.sidebar.markdown(f"📄 Documents loaded: **{st.session_state.document_count}**")
    st.sidebar.markdown(f"🧩 Text chunks created: **{st.session_state.chunk_count}**")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Instructions")
    st.sidebar.markdown("Ask questions about your CTSE lecture notes.")
    st.sidebar.markdown("The chatbot will find relevant information from your documents and provide answers.")
    
    # Chat input
    if question := st.chat_input("Ask a question about your lecture notes"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                response, sources = answer_question(
                    question, 
                    st.session_state.model, 
                    st.session_state.chunks, 
                    st.session_state.embeddings
                )
            
            message_placeholder.markdown(response)
            
            # Display sources
            with st.expander("View Sources"):
                for i, chunk in enumerate(sources):
                    source_file = os.path.basename(chunk["source"])
                    st.markdown(f"**Source {i+1} from {source_file}:**")
                    st.text(chunk["content"][:200] + "...")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()