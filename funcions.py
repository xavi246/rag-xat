from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate


class RAG:
   
    def __init__(self):
        self.model_embedding = self.definir_model_embedding()
        self.model_llm = self.definir_model_llm()
        self.documents = self.carregar_fitxers()
        self.index = self.crear_index()
        self.xat = self.crear_xat()

        
    def definir_model_embedding(self, nom_model = None):
        # loads sentence-transformers/all-MiniLM-L12-v2
        if nom_model is None:
            nom_model = "sentence-transformers/all-MiniLM-L12-v2"

        embed_model = HuggingFaceEmbedding(model_name=nom_model)
        Settings.embed_model = embed_model
        
        return(embed_model)



    def definir_model_llm(self, nom_model = None):
        # llmintfloat/e5-large-v2
        #llm = HuggingFaceLLM(model_name="intfloat/e5-large-v2",
        #                     tokenizer_name="intfloat/e5-large-v2")
        token = "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"
    
        if nom_model is None:
            nom_model = "mistralai/Mistral-7B-Instruct-v0.2"

        llm = HuggingFaceInferenceAPI(model_name=nom_model, token=token)
        Settings.llm = llm
        
        return(llm)



    def carregar_fitxers(self, cami: str = None):
        # load data
        if cami is None:
            cami = "./rag-documents"
        
        documents = SimpleDirectoryReader(cami).load_data()
    
        return(documents)



    def crear_index(self, documents = None):
        # create index
        if documents is None:
            documents = self.documents
        
        index = VectorStoreIndex.from_documents(documents=documents)

        return(index)


    def crear_xat(self):
        xat = self.index.as_query_engine()

        return(xat)



    def fer_pregunta(self, pregunta: str):
        query = '<s>[INST]Responde a la siguiente pregunta sobre productos fitosanitarios. Debes contestar las preguntas en ESPAÑOL, en base al contexto. Pregunta: ' + pregunta + '[/INST]'

        resposta = self.xat.query(query)
    
        return(resposta.response)
