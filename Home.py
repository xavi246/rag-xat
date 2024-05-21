import client
import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(./rag-icon.png);
                background-repeat: no-repeat;
                background-size: content;
                background-position: center top;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app() -> None:
    """RAG frontend"""
    
    # config
    st.set_page_config(
        page_title="Retrieval Augmented Generation",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )
    add_logo()

    st.title("📚 Retrieval Augmented Generation")

    if client.get_fastapi_status() is False:
        st.warning("❗ FastAPI is not ready. Make sure your backend is running")
        st.stop()  # exit application after displaying warning if FastAPI is not available
    else:
        st.warning("FastAPI backend is running 👌")
        st.session_state['embed_model'] = "sentence-transformers/all-MiniLM-L12-v2"
        st.session_state['ia_model'] = "mistralai/Mistral-7B-Instruct-v0.2"


    st.header("Home")
    st.markdown(
        """
    This application allows you to import PDF files and search over them using LLMs.
    
    For each file, the text is divided in chunks that are embedded with the embedding model selected.
    The embedded files are stored in VectorStoreIndex.
    
    When you query the system, the most relevant chunks are retrieved and a summary answer is generated using the selected model.

    The ingestion and retrieval steps are exposed via FastAPI endpoints.
    
    The frontend is built with Streamlit and exposes the different functionalities via a simple web UI.
    """
    )

    st.subheader("Hello from TFM-RAG 👋")
    st.markdown(
        """
        TFM - Màster en Ciència de dades - UOC
    """
    )


if __name__ == "__main__":
    # run as a script to test streamlit app locally
    app()
