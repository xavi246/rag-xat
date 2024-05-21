from fastapi  import FastAPI, HTTPException
from funcions import RAG


app = FastAPI()
rag = RAG()

@app.get("/")
def arrel():
    return {"message": "Servidor actiu"}


@app.get("/models")
async def defineix(embed_model: str, llm_model: str):

#    moment0 = time()
    rag.model_embedding = rag.definir_model_embedding(embed_model)
#    moment1 = time()
    rag.model_llm = rag.definir_model_llm(llm_model)
#    moment2 = time()
#    temps1 = (moment1 - moment0) * 10**3
#    temps2 = (moment2 - moment1) * 10**3
#    missatge = f"Temps per carregar embeddings: {temps1:.03f}ms<br>"
#    missatge += f"Temps per carregar embeddings: {temps2:.03f}ms"
    missatge = "Models"
    return {"message:": missatge}



@app.post("/documents")
async def documents():

    try:
        rag.documents = rag.carregar_fitxers('./upload_docs')
        rag.index = rag.crear_index(rag.documents)
        rag.xat = rag.crear_xat(rag.index)
    except ValueError as e:
        raise HTTPException(status_code=418, detail=str(e))

    missatge = "Documents"
    return {"message:": missatge}



@app.get("/documents_stored")
async def documents_guardats():

    try:
        documents = rag.documents
    except ValueError as e:
        raise HTTPException(status_code=418, detail=str(e))

    return documents



@app.get("/xat")
async def xat():

    try:
        rag.xat = rag.crear_xat()
    except ValueError as e:
        raise HTTPException(status_code=418, detail=str(e))

    missatge = "Xat"
    return {"message:": missatge}



@app.get("/pregunta")
async def pregunta(rag_query: str):

    try:
        resposta = rag.fer_pregunta(rag_query)
    except ValueError as e:
        raise HTTPException(status_code=418, detail=str(e))

    return resposta

        