import streamlit as st
import fitz  # PyMuPDF for PDF processing
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings  #paid
from sentence_transformers import SentenceTransformer #free
from langchain.vectorstores import FAISS

# Set Google Gemini API key
google_key ='key'
genai.configure(api_key=google_key)
model = genai.GenerativeModel('gemini-pro')

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective model

# Define embedding function for FAISS
def embedding_function(text):
    """Embedding function for FAISS."""
    # Ensure the input is a list of strings
    if isinstance(text, str):
        text = [text]
    # Encode the text and return the first embedding
    return embedding_model.encode(text)[0]

# Streamlit UI
st.title("Chat with your PDF 📄💬")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
    # Extract text from the PDF
    def extract_text(pdf_file):
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None

    text = extract_text(uploaded_file)

    if text:
        # Split text into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Generate embeddings using Sentence Transformers
        embeddings = embedding_model.encode(chunks)  # Convert text chunks to embeddings

        # Store embeddings in FAISS
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(chunks, embeddings)),  # Pair text chunks with embeddings
            embedding=embedding_function  # Pass the embedding function
        )

        st.write("✅ PDF processed! Start chatting below.")

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            st.write(f"{message['role']}: {message['text']}")

        # Chat Interface
        user_input = st.text_input("Ask a question about the PDF:")

        if user_input:
            # Add user input to chat history
            st.session_state.messages.append({"role": "user", "text": user_input})

            # Perform similarity search
            docs = vector_store.similarity_search(user_input, k=5)
            context = " ".join([doc.page_content for doc in docs])

            # Query Gemini Pro with context
            try:
                response = model.generate_content(
                    f"Context: {context}\n\nQuestion: {user_input}"
                )
                answer = response.text
                st.session_state.messages.append({"role": "AI", "text": answer})
                st.write("🤖 AI Response:", answer)
            except Exception as e:
                st.error(f"Error querying Gemini Pro: {e}")


