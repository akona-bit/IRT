from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from pydantic import BaseModel
from typing import List, Dict, Any

from irt import mmle, chi_square

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IRTData(BaseModel):
    items: List[str]
    responses: List[List[float]] # N students x J items

@app.post("/api/calibrate-irt-json")
async def calibrate_irt_json(data: IRTData):
    try:
        U = np.array(data.responses)
        a_est, b_est = mmle(U, name="MMLE", verbose=False)
        
        results = []
        for i, col in enumerate(data.items):
            # Cần tính fitness parameter (p-value của chi-square) 
            # Nhưng chi-square cần theta (ước lượng năng lực). Để đơn giản trả về tham số trước
            results.append({
                "id": col,
                "a": float(a_est[i]),
                "b": float(b_est[i]),
                "fit": 1.0 # default fit for now if not computed
            })
            
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/calibrate-irt")
async def calibrate_irt(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        item_cols = [c for c in df.columns if c.startswith('Cau')]
        if not item_cols:
            return {"status": "error", "message": "No item columns found (must start with 'Cau')"}
            
        U = df[item_cols].fillna(-1).values
        a_est, b_est = mmle(U, name="MMLE", verbose=False)
        
        results = []
        for i, col in enumerate(item_cols):
            results.append({
                "id": col,
                "a": float(a_est[i]),
                "b": float(b_est[i]),
                "fit": 1.0 # placeholder
            })
        
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run with: uvicorn main:app --reload
