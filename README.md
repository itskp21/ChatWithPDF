# Chat with PDF using Gemini

This Streamlit application allows users to upload PDF files, process their content using the FAISS vector store, and ask questions about the extracted information using Google Generative AI's Gemini model.

## Features

- **PDF Upload**: Easily upload multiple PDF files.
- **Text Extraction**: Extracts text content from the uploaded PDFs.
- **FAISS Vector Store**: Processes the text and stores it in a FAISS index for efficient similarity search.
- **Conversational AI**: Utilize Google Generative AI's Gemini model to answer questions based on the PDF content.

## Technologies Used

- Streamlit
- PyPDF2
- LangChain
- Google Generative AI
- FAISS
- dotenv

## Prerequisites

- Python 3.7 or higher
- Google Cloud account with access to Google Generative AI
- A valid API key for Google Generative AI
