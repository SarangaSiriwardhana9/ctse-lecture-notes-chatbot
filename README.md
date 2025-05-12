# CTSE Lecture Notes Chatbot

A Question-Answering system for Current Trends in Software Engineering lecture notes, developed as part of Assignment 2 for SE4010 - Current Trends in Software Engineering, Semester 1, 2025.


## Project Overview

This project implements a chatbot that answers questions about CTSE lecture notes using Retrieval Augmented Generation (RAG). The system:

1. Loads lecture notes from text and PDF files  
2. Splits them into semantically meaningful chunks  
3. Creates vector embeddings using Sentence Transformers  
4. Retrieves relevant content based on question similarity  
5. Generates accurate answers using Google's Gemini API  
6. Provides source citations for transparency  

## Features

- **Multi-format Document Loading**: Imports lecture notes from both text and PDF files  
- **Intelligent Text Chunking**: Splits documents with appropriate overlap to maintain context  
- **Semantic Search**: Uses dense vector embeddings to find conceptually related content  
- **Context-Aware Responses**: Generates answers grounded in the provided lecture materials  
- **Source Attribution**: Shows which parts of the lecture notes were used for each answer  
- **User-Friendly Interface**: Simple interactive chat experience in Jupyter or via Streamlit  

## Requirements

- Python 3.8+
- Required libraries:
  - sentence-transformers
  - requests
  - numpy
  - scikit-learn
  - PyPDF2
  - streamlit (optional for UI)

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
   pip install -r requirements.txt
   ```

4. Add your lecture notes:
   - Create a `data` folder in the project directory
   - Add your lecture notes as text files (`.txt`) or PDFs (`.pdf`) in this folder

5. Set up your Google Gemini API key:
   - Get an API key from Google AI Studio  
   - Add your API key to the environment variables or update it in the notebook

## Usage

### Jupyter Notebook
Open and run the chatbot.ipynb notebook:
```bash
jupyter notebook chatbot.ipynb
```

### Command Line Interface
Run the chatbot from the command line:
```bash
python chatbot.py
```

### Streamlit Interface (Optional)
Launch the Streamlit web interface:
```bash
streamlit run app.py
```

## How It Works

### RAG Architecture

The system implements a Retrieval Augmented Generation (RAG) architecture:

#### Document Processing:
- Loads and parses text/PDF documents from the data folder  
- Splits documents into smaller chunks (800 tokens with 150 token overlap)  

#### Embedding Creation:
- Uses Sentence Transformers (`all-MiniLM-L6-v2` model)  
- Generates 384-dimensional vector embeddings for each text chunk  
- Creates a vector store for efficient similarity search  

#### Query Processing:
- Converts user questions into the same embedding space  
- Uses cosine similarity to find the most relevant document chunks  
- Retrieves top-k chunks based on similarity scores  

#### Answer Generation:
- Constructs an instructional prompt incorporating retrieved contexts  
- Sends the prompt to Google's Gemini API  
- Specifies that answers must be based only on provided context  
- Returns the generated answer with source attributions  

## System Components

- **Document Loader**: Handles different file formats and text extraction  
- **Text Chunker**: Implements intelligent document splitting algorithms  
- **Embedding Model**: Converts text to dense vector representations  
- **Vector Store**: Enables efficient similarity search in the embedding space  
- **Retriever**: Finds semantically relevant document chunks for each query  
- **Prompt Template**: Structures context and instructions for the LLM  
- **LLM Interface**: Manages API communication with Google Gemini  
- **Response Formatter**: Presents answers with clear source attribution  

## Performance Evaluation

The system was evaluated using:

- **Answer Relevance**: Manual assessment of response pertinence to questions  
- **Source Accuracy**: Verification that answers cite appropriate lecture material  
- **Response Time**: Measurement of end-to-end query processing time  
- **Hallucination Rate**: Analysis of unfounded claims not present in source material  

## Justification of LLM Choice

Google Gemini was selected as the LLM for this project because:

- **Context Handling**: Effectively processes multiple chunks of lecture content  
- **Instruction Following**: Consistently adheres to constraints about using only provided context  
- **Free Academic Tier**: Accessible API with reasonable free quota for student projects  
- **Response Quality**: Generates well-structured, natural language explanations  
- **Documentation**: Comprehensive API documentation and examples for developers  

## Development Approach

This project uses a modular, component-based approach to RAG implementation:

- **Simplicity First**: Prioritizes clean, understandable code over complex optimizations  
- **Functional Separation**: Organizes code by distinct RAG pipeline stages  
- **Framework Independence**: Avoids excessive dependencies on specific ML frameworks  
- **Extensibility**: Designed for easy modification and enhancement  

## Future Improvements

Potential enhancements for the system:

- Multi-modal Processing: Add support for images, diagrams, and charts from lecture slides  
- Document Format Expansion: Handle additional file types (DOCX, PPT, etc.)  
- Conversation Memory: Implement short-term memory for follow-up questions  
- Adaptive Chunking: Use semantic boundaries for more intelligent text splitting  
- Performance Optimization: Implement caching and vector store persistence  
- User Feedback Loop: Add mechanisms to improve responses based on user ratings  
- Multi-course Support: Scale to handle materials from multiple courses  

## Project Resources

The complete implementation of the CTSE Lecture Notes Chatbot is available for review through the following resources:

### GitHub Repository

The full source code, including the Jupyter Notebook implementation, data processing modules, and documentation, is available in the public GitHub repository:

Repository URL: [https://github.com/SarangaSiriwardhana9/ctse-lecture-notes-chatbot](https://github.com/SarangaSiriwardhana9/ctse-lecture-notes-chatbot)

The repository contains:

- `chatbot.ipynb`: The main Jupyter Notebook implementation  
- `data/`: Sample lecture notes for demonstration  
- `requirements.txt`: All required dependencies  
- Documentation and license information  


The demonstration provides a practical overview of how the RAG architecture effectively retrieves and generates responses based on the course lecture notes.

## License

This project is created for educational purposes as part of a university assignment.

Created for SE4010 - Current Trends in Software Engineering, Assignment 2 - AI/ML (Semester 1, 2025)
