import streamlit as st
import os
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

def main():
    st.set_page_config(
        page_title="PDF Knowledge Assistant",
        page_icon=":orange_book:",
        layout="wide"
    )
    
    st.title("📚 PDF Knowledge Assistant")
    st.markdown("Query PDF documents using Groq and PGVector")

    # User Inputs Section
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("GROQ API Key", type="password")
        db_url = st.text_input(
            "Database URL",
            placeholder="postgresql+psycopg://user:password@host:port/database"
        )
        pdf_url = st.text_input(
            "PDF URL", 
            value="https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
        )
    
    # Main Chat Interface
    user_query = st.text_area(
        "Enter your question:",
        placeholder="Ask about the PDF content...",
        height=150
    )

    if st.button("Submit"):
        if not all([groq_api_key, db_url, pdf_url, user_query]):
            st.error("Please fill all fields")
            return

        # Set environment variables
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize components
        knowledge_base = PDFUrlKnowledgeBase(
            urls=[pdf_url],
            vector_db=PgVector2(
                collection="pdf_documents",
                db_url=db_url
            )
        )
        
        storage = PgAssistantStorage(
            table_name="pdf_assistants",
            db_url=db_url
        )

        assistant = Assistant(
            knowledge_base=knowledge_base,
            storage=storage,
            show_tool_calls=True,
            search_knowledge=True,
            read_chat_history=True,
            # Use mixtral for faster responses
            model="mixtral-8x7b-32768"
        )

        # Load knowledge base and process query
        with st.spinner("Loading knowledge base..."):
            knowledge_base.load(recreate=False)
            
        with st.spinner("Processing your query..."):
            response = assistant.run(user_query, stream=False)
            
        st.subheader("Answer")
        st.markdown(response)

if __name__ == "__main__":
    main()
