from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import faiss 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app=FastAPI()
dim=256 #embedding size
nlist=50 #number of clusters
q=faiss.IndexFlatL2(dim) #quantizer
idx=faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_L2)

num_embeddings=5000
embeddings=torch.randn(num_embeddings, dim).to(torch.float32).numpy()
min_train_size=max(39*nlist, 2000)
train_size=min(min_train_size, len(embeddings))
idx.train(embeddings[:train_size])
idx.add(embeddings)

#prevent crashes
batch_size=100
for i in range(0, len(embeddings), batch_size):
    idx.add(embeddings[i:i+batch_size])

class QueryRequest(BaseModel):
    query_vector: list[float]=Field(..., min_items=256, max_items=256) 

@app.post("/search")
async def search_faiss(request: QueryRequest):
    if len(request.query_vector)!=dim:
        raise HTTPException(status_code=400, detail=f'Query_vector must have exactly {dim} values.')
    query=torch.tensor(request.query_vector).to(torch.float32).numpy().reshape(1,-1)
    D,I=idx.search(query, k=3)
    return {"neighbors": I.tolist(), "distances":D.tolist()}

#uvicorn chatbot_api:app --reload
#uvicorn chatbot_api:app --host 127.0.0.1 --port 8000 --reload
#netstat -ano | findstr :8000
#taskkill /PID <PID> /F
