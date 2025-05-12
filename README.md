# CTSE Lecture Notes Chatbot

A simple Question-Answering system for Current Trends in Software Engineering lecture notes, developed as part of Assignment 2 for SE4010 - Current Trends in Software Engineering, Semester 1, 2025.

## Project Overview

This project implements a chatbot that can answer questions based on CTSE lecture notes using Retrieval Augmented Generation (RAG). The system:

1. Loads lecture notes from text files
2. Splits them into manageable chunks
3. Creates vector embeddings using Sentence Transformers
4. Retrieves relevant content when asked questions
5. Generates natural language answers using Google's Gemini API

## Features

- **Document Loading**: Imports lecture notes from text files
- **Text Processing**: Splits documents into manageable chunks
- **Semantic Search**: Uses embeddings to find relevant content
- **AI-Powered Responses**: Generates conversational answers using Google Gemini
- **Interactive Interface**: Simple command-line chat interface

## Requirements

- Python 3.8+
- Required libraries:
  - sentence-transformers
  - requests
  - numpy
  - scikit-learn

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ctse-lecture-notes-chatbot.git
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
   pip install requests sentence-transformers numpy scikit-learn
   ```

4. Add your lecture notes:
   - Create a `data` folder in the project directory
   - Add your lecture notes as text files (`.txt`) in this folder

5. Set up your Google Gemini API key:
   - Get an API key from Google AI Studio
   - Update the API key in the code

## Usage

Run the chatbot:
```bash
python chatbot.py
```

- Ask questions about your lecture notes
- Type `exit` to quit the application

## How It Works

### Architecture Overview

The system implements a Retrieval Augmented Generation (RAG) architecture:

**Document Processing**:
- Loads text documents from the `data` folder
- Splits documents into smaller chunks for efficient processing

**Embedding Creation**:
- Converts each text chunk into vector embeddings
- Embeddings capture semantic meaning of the text

**Retrieval**:
- Converts a question into an embedding
- Finds chunks most similar to the question using cosine similarity

**Answer Generation**:
- Provides retrieved chunks as context to the Gemini model
- Prompts Gemini to generate an answer based solely on the provided context

## Components

- **Document Loader**: Imports text files and organizes their content  
- **Text Chunker**: Splits large documents into manageable pieces  
- **Embedding Model**: Converts text to vector representations  
- **Retriever**: Finds relevant chunks based on question similarity  
- **LLM Interface**: Connects to Google Gemini API for answer generation  
- **Chat Interface**: Provides a simple command-line interface for interaction  

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

## Future Improvements

Potential enhancements for the system:

- Implement a web-based user interface  
- Add support for multiple document formats (PDF, DOCX)  
- Improve chunking strategy with semantic-aware splitting  
- Add conversation history for more natural interactions  
- Implement evaluation metrics to measure answer quality  

## References

- Google Generative AI Documentation  
- RAG Tutorial from LangChain  
- Sentence Transformers Documentation  

## License

This project is created for educational purposes as part of a university assignment.

Created for SE4010 - Current Trends in Software Engineering, Assignment 2 - AI/ML (Semester 1, 2025)
