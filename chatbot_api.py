from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import faiss 
import uvicorn
import os
import threading
from sentence_transformers import SentenceTransformer
import gradio as gr
from starlette.middleware.wsgi import WSGIMiddleware

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app=FastAPI()
#load embedding model
embedding_model=SentenceTransformer('all-MiniLM-L6-v2')

documents=["Outstanding stock is not a form of corporate debt, in particular, stocks are not \
obligations, they have no maturity, and the corporation does not promise to pay back \
the stockholders their originally invested amount any time later." ,
"Instead, stockholders or shareholders are owners of the corporation in the proportion to the total stock \
issued by the corporation.",
"Some of this stock, which is called treasury stock, can be \
owned by the corporation itself.",
"In fact, the total value of the corporation, which is referred to as its market capitalization, is determined by the number of issued shares \
multiplied by the current market price of one share."]

doc_embeddings=embedding_model.encode(documents)

dim=doc_embeddings.shape[1] #embedding size
nlist=5 #number of clusters
#q=faiss.IndexFlatL2(dim) #quantizer
#idx=faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_L2)
idx=faiss.IndexFlatL2(dim)

# num_embeddings=5000
# embeddings=torch.randn(num_embeddings, dim).to(torch.float32).numpy()
# min_train_size=max(39*nlist, 2000)
# train_size=min(min_train_size, len(embeddings))
# idx.train(embeddings[:train_size])
idx.add(doc_embeddings)

#prevent crashes
batch_size=100
for i in range(0, len(doc_embeddings), batch_size):
    idx.add(doc_embeddings[i:i+batch_size])

class QueryRequest(BaseModel):
    query: str #list[float]=Field(..., min_items=256, max_items=256) 

@app.get("/")
async def root():
    return {"message": "FAISS Document Retrieval API is running!"}

@app.post("/search")
async def search_faiss(request: QueryRequest):
    # if len(request.query_vector)!=dim:
    #     raise HTTPException(status_code=400, detail=f'Query_vector must have exactly {dim} values.')
    if not request.query.strip():
        return {"error":"Query cannot be empty."}
    #query=torch.tensor(request.query_vector).to(torch.float32).numpy().reshape(1,-1)
    query_embedding=embedding_model.encode([request.query])
    D,I=idx.search(query_embedding, k=3)
    results=[documents[i] for i in I[0] if i<len(documents)]
    return {"results": results}
    #return {"neighbors": I.tolist(), "distances":D.tolist()}

#Gradio UI
def chatbot(query):
    if not query.strip():
        return "Please enter a valid query."
    query_embedding=embedding_model.encode([query])
    D,I=idx.search(query_embedding, k=3)
    return '\n'.join([documents[i] for i in I[0] if i<len(documents)])

def run_gradio():
    iface=gr.Interface(
        fn=chatbot, 
        inputs="text", 
        outputs="text",
        title="Document Search Chatbot",
        description="Ask a question and find the most relevant documents")
    iface.launch(server_name="127.0.0.1", server_port=7861, share=False)
    
    app.mount("/gradio", WSGIMiddleware(iface))

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

class AddDocumentsRequest(BaseModel):
    new_documents: list[str]

@app.post("/add-documents")
async def add_documents(request: AddDocumentsRequest):
    global documents, doc_embeddings, IndexError
    new_embeddings = embedding_model.encode(request.new_documents)
    documents.extend(request.new_documents)
    idx.add(new_embeddings)
    return {"message": f"Added {len(request.new_documents)} documents!"}

if __name__=="__main__":
    gradio_thread=threading.Thread(target=run_gradio, daemon=True)
    gradio_thread.start()
    run_fastapi()

#start /B python chatbot_api.py
#uvicorn chatbot_api:app --reload
#uvicorn chatbot_api:app --host 127.0.0.1 --port 8000 --reload
#curl -X POST "http://127.0.0.1:8000/search" -H "Content-Type: application/json" -d "{\"query\": \"What is the value of a corporation?\"}"
#netstat -ano | findstr :8000
#taskkill /PID # /F
