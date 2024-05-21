import client
import streamlit as st

# def show_pdf(base64_pdf: str) -> str:
#     """Show a base64 encoded PDF in the browser using an HTML tag"""
#     return f'<embed src="data:application/pdf;base64,{base64_pdf}" width=100% height=800 type="application/pdf">'


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(./rag-icon.png);
                background-repeat: no-repeat;
                background-size: contain;
                background-position: 20px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def retrieval_form_container() -> None:
    """Container to enter RAG query and sent /rag_summary GET request"""

    with st.container():
        form = st.form(key="retrieval_query")
        rag_query = form.text_area(
            "Retrieval Query", value="¿Está vigente el producto PRAKTIS en España?"
        )

    if form.form_submit_button("Search"):
        with st.status("Running"):
            response = client.get_rag_answer(rag_query)
            st.write(response.json())
        
        st.session_state["history"].append(dict(query=rag_query, response=response.json()))


def history_display_container(history):
    if len(history) > 1:
        st.header("History")
        max_idx = len(history) - 1
        history_idx = st.slider("History", 0, max_idx, value=max_idx, label_visibility="collapsed")
        entry = history[history_idx]
    else:
        entry = history[0]

    st.subheader("Query")
    st.write(entry["query"])

    st.subheader("Response")
    st.write(entry["response"])
 
#    with st.expander("Sources"):
#        st.write(entry["response"])


def app() -> None:
    """Streamlit entrypoint for PDF Summarize frontend"""
    # config
    st.set_page_config(
        page_title="Retrieval",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )
    add_logo()

    st.title("📤 Retrieval")

    if client.get_fastapi_status() is False:
        st.warning("❗ FastAPI is not ready. Make sure your backend is running")
        st.stop()  # exit application after displaying warning if FastAPI is not available

    retrieval_form_container()

    if history := st.session_state.get("history"):
        history_display_container(history)
    else:
        st.session_state["history"] = list()


if __name__ == "__main__":
    # run as a script to test streamlit app locally
    app()
