from fastapi import FastAPI, Depends, HTTPException
import uvicorn
from api.routes import risk

app = FastAPI(title="ABRB: Autonomous Business Risk Brain", version="1.0.0")

# Include routes
app.include_router(risk.router, prefix="/risk", tags=["Risk"])

@app.get("/")
async def root():
    return {"message": "ABRB System is Online", "status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
