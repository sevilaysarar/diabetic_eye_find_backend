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
loaded_model = load_model('diabetic_eye_model_GPU_trained_5_epoch.h5')

def process_image(image):
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kopya = image.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(kopya, (5, 5), 0)
    thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)[1]

    kontur = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    kontur = kontur[:, 0, :]
    x1 = tuple(kontur[kontur[:, 0].argmin()])[0]
    y1 = tuple(kontur[kontur[:, 1].argmin()])[1]
    x2 = tuple(kontur[kontur[:, 0].argmax()])[0]
    y2 = tuple(kontur[kontur[:, 1].argmax()])[1]

    x = int(x2 - x1) * 4 // 50
    y = int(y2 - y1) * 5 // 50
    kopya2 = image.copy()
    if x2 - x1 > 100 and y2 - y1 > 100:
        kopya2 = kopya2[y1 + y: y2 - y, x1 + x: x2 - x]
        kopya2 = cv2.resize(kopya2, (400, 400))

    lab = cv2.cvtColor(kopya2, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=((8, 8)))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    med_son = cv2.medianBlur(son, 3)
    arka_plan = cv2.medianBlur(son, 37)
    maske = cv2.addWeighted(med_son, 1, arka_plan, -1, 255)
    son_img = cv2.bitwise_and(maske, med_son)
    return son_img

def predict_disease(image):
    resim_array = image.reshape((1, 400, 400, 3))
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
