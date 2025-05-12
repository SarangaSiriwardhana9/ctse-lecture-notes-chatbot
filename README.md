# CTSE Lecture Notes Chatbot

A question-answering system for Current Trends in Software Engineering lecture notes, developed as part of Assignment 2 for SE4010 - Current Trends in Software Engineering, Semester 1, 2025.
 
## Project Overview

This project implements a chatbot that can answer questions based on CTSE lecture notes using Retrieval Augmented Generation (RAG). The system:

1. Loads lecture notes from text and PDF files  
2. Splits them into manageable chunks  
3. Creates vector embeddings using Sentence Transformers  
4. Retrieves relevant content when asked questions  
5. Generates natural language answers using Google's Gemini API  

## Repository

GitHub Repository URL: [https://github.com/SarangaSiriwardhana9/ctse-lecture-notes-chatbot](https://github.com/SarangaSiriwardhana9/ctse-lecture-notes-chatbot)

## Repository Structure

- `chatbot.ipynb` - Main Jupyter Notebook implementation of the RAG system  
- `app.py` - Streamlit web interface for the chatbot  
- `data/` - Directory for storing lecture notes (add your .txt or .pdf files here)  
- `.gitignore` - Configuration for Git to exclude unnecessary files  

## Features

- **Document Loading**: Imports lecture notes from text and PDF files  
- **Text Processing**: Splits documents into manageable chunks  
- **Semantic Search**: Uses embeddings to find relevant content  
- **AI-Powered Responses**: Generates conversational answers using Google Gemini  
- **Interactive Interface**: Simple Streamlit web interface and Jupyter Notebook  

## Requirements

- Python 3.8+  
- Required libraries:
  - sentence-transformers  
  - requests  
  - numpy  
  - scikit-learn  
  - PyPDF2  
  - streamlit (for web interface)  

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SarangaSiriwardhana9/ctse-lecture-notes-chatbot.git
   cd ctse-lecture-notes-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install requests sentence-transformers numpy scikit-learn PyPDF2 streamlit
   ```

4. Add your lecture notes:

   - Place your lecture notes (.txt or .pdf files) in the `data/` folder

5. Set up your Google Gemini API key:

   - Get an API key from Google AI Studio  
   - Replace the placeholder API key in the code with your actual key  

## Usage

### Jupyter Notebook

Run the Jupyter Notebook to explore the implementation details:
```bash
jupyter notebook chatbot.ipynb
```

### Streamlit Interface

Launch the web interface:
```bash
streamlit run app.py
```

## How It Works

### Architecture Overview

The system implements a Retrieval Augmented Generation (RAG) architecture:

- **Document Processing**:
  - Loads text documents from the data folder  
  - Splits documents into smaller chunks for efficient processing  

- **Embedding Creation**:
  - Converts each text chunk into vector embeddings  
  - Embeddings capture semantic meaning of the text  

- **Retrieval**:
  - Converts a question into an embedding  
  - Finds chunks most similar to the question using cosine similarity  

- **Answer Generation**:
  - Provides retrieved chunks as context to the Gemini model  
  - Prompts Gemini to generate an answer based solely on the provided context  

## Development Approach

This project uses a simple, modular approach to RAG implementation:

- Design prioritizes simplicity and clarity over optimization  
- Functions are separated based on their role in the RAG pipeline  
- No complex frameworks are used to ensure transparency of implementation  
- The modular structure makes it easy to modify or extend individual components  

## Justification of LLM Choice

Google Gemini was chosen as the LLM for this project because:

- Provides high-quality responses with strong contextual understanding  
- API accessible with a free tier for academic projects  
- Handles context-based responses well, making it suitable for RAG applications  
- Documentation and support are robust for beginning developers  
 
## References

- Google Generative AI Documentation  
- RAG Tutorial from LangChain  
- Sentence Transformers Documentation  

## License

This project is created for educational purposes as part of a university assignment.  
Created for SE4010 - Current Trends in Software Engineering, Assignment 2 - AI/ML (Semester 1, 2025)
