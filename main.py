from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.openapi.utils import get_openapi
from app import RAGAssistant
import uvicorn
import os

app = FastAPI(
    title="Document Analyzer API",
    description="API for analyzing documents and providing JSON summaries",
    version="1.0.0",
)

rag_assistant = RAGAssistant()


@app.post("/analyze", summary="Analyze Document", description="Upload a document for analysis and receive a JSON summary.")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze an uploaded document and return a JSON summary of its key information.

    - **file**: The document file to be analyzed (PDF, TXT, CSV, XLSX, or DOCX)

    Returns a JSON object containing the analysis result.
    """
    try:
        # Save the uploaded file temporarily
        with open(file.filename, "wb") as f:
            f.write(await file.read())

        # Process the document
        rag_assistant.process_document(file.filename)

        # Analyze the document
        json_output = rag_assistant.analyze_document()

        # Remove the temporary file
        os.remove(file.filename)

        return {"result": json_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Document Analyzer API",
        version="1.0.0",
        description="API for analyzing documents and providing JSON summaries",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
