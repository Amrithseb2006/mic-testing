from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from predictor import predict_new_data   

# 👇 ADD THIS
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 👇 ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuditInput(BaseModel):
    audit_at: str
    DB_username: str
    OS_username: str
    Source_programname: str
    Src_ip: str
    full_query: str
    full_qry_hv: str


@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_api(data: AuditInput):
    df = pd.DataFrame([data.dict()])
    result = predict_new_data(df)
    return result