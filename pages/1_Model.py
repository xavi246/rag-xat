import client
import streamlit as st

# def show_pdf(base64_pdf: str) -> str:
#     """Show a base64 encoded PDF in the browser using an HTML tag"""
#     return f'<embed src="data:application/pdf;base64,{base64_pdf}" width=100% height=800 type="application/pdf">'

# selected models
embed_model = None
ia_model = None

# time for loading models
temps = None


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(../rag-icon.png);
                background-repeat: no-repeat;
                background-size: 30%;
                background-position: 20px 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def aply(c):
    temps = client.set_models(embed_model, ia_model)
    c.write(temps)


def models_containers() -> None:
    """Container to select models and sent /rag_summary GET request"""
 
    with st.container(border=True):
        st.subheader('Embdding model', divider='rainbow')
        
        embed_model = st.selectbox("Select the embeding model:",
                                   ("sentence-transformers/all-MiniLM-L12-v2", "OpenAI"))
        st.session_state['embed_model'] = embed_model
    
    with st.container(border=True):
        st.subheader('Generative model', divider='rainbow')
        
        ia_model = st.selectbox("Select the generative model:",
                                ("mistralai/Mistral-7B-Instruct-v0.2", "OpenAI"))
        st.session_state['ia_model'] = ia_model
    
    c = st.container()
    c.button('Aply', on_click=aply(c))



def app() -> None:
    """Model selection"""
    # config
    st.set_page_config(
        page_title="Model selection",
        page_icon="📚",
        layout="centered",
        menu_items={"Get help": None, "Report a bug": None},
    )

    add_logo()
    st.title("✨ Model selection")

    if client.get_fastapi_status() is False:
        st.warning("❗ FastAPI is not ready. Make sure your backend is running")
        st.stop()  # exit application after displaying warning if FastAPI is not available

    models_containers()



if __name__ == "__main__":
    # run as a script to test streamlit app locally
    app()
