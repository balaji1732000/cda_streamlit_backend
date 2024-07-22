from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import os

app = FastAPI()

UPLOAD_FOLDER = "uploaded_files"


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    streamlit_url = os.getenv("STREAMLIT_URL", "http://localhost:8501")
    return RedirectResponse(url=f"{streamlit_url}?file={file.filename}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
