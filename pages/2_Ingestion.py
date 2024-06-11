import client
import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(rag-icon.png);
                background-repeat: no-repeat;
                background-size: 70%;
                background-position: 20px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



def pdf_upload_container():
    """Container to uploader arbitrary PDF files and send /store_pdfs POST request"""
    uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Upload"):
        with st.status("Storing PDFs"):
            client.post_store_pdfs(uploaded_files)
  


def stored_documents_container():
    """Container showing stored PDF documents, results of /documents GET request"""
    response = client.get_all_documents_file_name()
    documents = response.json()
    docs = [d['metadata']['file_name'] for d in documents if d['metadata']['page_label'] == '1']
    st.table({"PDF file name": docs})
    


def app() -> None:
    """Streamlit entrypoint for PDF Summarize frontend"""
    # config
    st.set_page_config(
        page_title="Ingestion",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )
    add_logo()

    st.title("📥 Ingestion")

    if client.get_fastapi_status() is False:
        st.warning("❗ FastAPI is not ready. Make sure your backend is running")
        st.stop()  # exit application after displaying warning if FastAPI is not available

    left, right = st.columns(2)

    with left:
        st.subheader("Upload PDF files")
        pdf_upload_container()

    with right:
        st.subheader("Selected models")
        st.write(st.session_state['embed_model'])
        st.write(st.session_state['ia_model'])

    st.header("Stored documents")
    stored_documents_container()


if __name__ == "__main__":
    # run as a script to test streamlit app locally
    app()
