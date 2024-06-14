from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from keras.models import load_model
from efficientnet.keras import EfficientNetB5
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
loaded_model = load_model('../saved_models/final_model.h5')

def process_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    return image

def predict_disease(image):
    resim_array = image.reshape((1, 224, 224, 3))
    prediction = loaded_model.predict(resim_array, verbose = 0)
    return np.argmax(prediction)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    processed_image = process_image(image)
    prediction = predict_disease(processed_image)
    
    print(prediction)
    prediction = str(prediction)
    return {prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
