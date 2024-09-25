import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit app title and description
st.set_page_config(page_title="Chat with PDF using Gemini 🤖", page_icon=":books:")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Chat with PDF using Gemini 📚</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload your PDF, process it with FAISS, and ask questions using Google Generative AI's Gemini model.</h4>", unsafe_allow_html=True)

# Sidebar for PDF upload and actions
st.sidebar.title("Upload and Process PDF")
st.sidebar.markdown("Upload your PDF files here:")

# Function to extract text from PDF files
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_file = io.BytesIO(pdf.read())
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Ensure extracted text is not None
                    text += extracted_text
        except Exception as e:
            return f"Error reading PDF file: {e}"
    return text

# Split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings_model)
    
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")  # Save the index locally

# Generate conversational chain with Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say "answer is not available in the context." Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

# Function to handle the user query
def handle_user_input(user_question):
    embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# File uploader widget for PDF files
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process PDF files and create FAISS index
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        raw_text = get_pdf_text(uploaded_files)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success(f"Processed {len(text_chunks)} text chunks.")
        else:
            st.sidebar.error("Failed to extract text from PDFs.")

# Ask a question after processing the PDFs
st.markdown("## Ask a Question 💬")
user_question = st.text_input("Ask a question based on the content of the uploaded PDFs:")

# Display answer to the user's question
if user_question:
    if os.path.exists("faiss_index/index.faiss"):
        with st.spinner("Generating answer..."):
            response = handle_user_input(user_question)
            st.success(f"**Answer:** {response}")
    else:
        st.error("FAISS index not found. Please upload and process PDFs first.")

# Footer
st.markdown("""<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: #6c757d;
}
</style>
<div class="footer">
</div>
""", unsafe_allow_html=True)