HUGGING_FACE_API_KEY = "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"

import transformers
import torch
import re
from time import time
from huggingface_hub import login

torch.cuda.empty_cache()

login(HUGGING_FACE_API_KEY)  # Nos autentificamos en HuggingFace

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    # The quantization line
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True, "cache_dir": "/mnt/huggingface"},
    max_new_tokens=256
 )  # Cargamos nuestro modelo en memoria


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


def get_documents_from_directory(directory: str):
    loader = PyPDFDirectoryLoader(directory)
    docs_loaded = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_splitter = splitter.split_documents(docs_loaded)

    return docs_splitter


def create_db_from_documents(documents: list):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGING_FACE_API_KEY,
        model_name="BAAI/bge-m3"
    )
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store


documents = get_documents_from_directory('documentos_fitosanitarios')  # Para que funcione crea una carpeta y sube los documentos a dicha carpetam
vector_store = create_db_from_documents(documents)


# **LangChain** funciona con cadenas, para esta demonstración cargamos una cadena muy básica de Retrieval para que pueda obtener info del documento.

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline(pipeline=pipeline)


from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.prompt import PromptTemplate


def create_chain(vector_store, model_loaded):

    SYS_TEMPLATE = """
    <s>[INST] Responde a la siguiente pregunta sobre productos fitosanitarios. Debes contestar las preguntas en ESPAÑOL, en base al contexto.
    <context>
    {context}
    </context>
    Pregunta: {input}
    [/INST]
    """

    prompt = PromptTemplate(
        input_variables=['context', 'input'],
        template=SYS_TEMPLATE
    )


    chain = create_stuff_documents_chain(
        model_loaded,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=chain
    )

    return retrieval_chain


chain = create_chain(vector_store, hf)

t1 = time()
response = chain.invoke({'input': '¿Está vigente el producto PRAKTIS en España?'})
t2 = time()

resposta = response['answer']
print(re.split(r'INST\]\n', resposta)[-1].lstrip())
print(t2 - t1)
