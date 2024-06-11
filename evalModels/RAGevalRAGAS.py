from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from huggingface_hub import login
import torch
import re

torch.cuda.empty_cache()

HUGGING_FACE_API_KEY = "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"
login(HUGGING_FACE_API_KEY)  # Nos autentificamos en HuggingFace


# Load and split documents
loader = PyPDFDirectoryLoader("documentos_fitosanitarios")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

documents = loader.load()
chuncks = splitter.split_documents(documents)


# Create db from documents
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGING_FACE_API_KEY,
    model_name="BAAI/bge-m3"
)
vectorstore = FAISS.from_documents(chuncks, embeddings)


# Define vectorstore as retriever to enable semantic search
retriever = vectorstore.as_retriever()


# Define LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True, "cache_dir": "/mnt/huggingface"},
    pipeline_kwargs={"max_new_tokens": 256}
)


# Define prompt
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

chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=chain
    )

## Preparing the Evaluation Data
from datasets import Dataset

questions = ["¿Está vigente el producto PRAKTIS en España?", 
             "¿Me puedes decir los cultivos con los que se puede usar PRAKTIS?",
             "¿Cual es la dosis recomendada para tratar la peste Roya, Puccinia spp. en la cebada?"]

ground_truth = ["Sí",
                "Avena, Cebada, Centeno, Trigo, Triticale.",
                "La dosis recomendada es 0,8 l/ha."]
answers = []
contexts = []

# Inference
for query in questions:
    resposta = rag_chain.invoke({'input': query})['answer']
    answer = re.split(r'INST\]\n', resposta)[-1].lstrip()
    answers.append(answer)
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truth
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

## Evaluating the RAG application
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=llm,
    embeddings=embeddings
)

df = result.to_pandas()
