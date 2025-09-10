# views.py
import json
from pathlib import Path
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from tensorflow import keras

# compute model path relative to project directory
BASE = Path(settings.BASE_DIR).parent  # settings.BASE_DIR -> ml_api_project, parent -> project root
MODEL_PATH = BASE / 'models' / 'kaggle_cnn_model.h5'
CLASS_PATH = BASE / 'models' / 'class_names.json'
IMG_SIZE = (64, 64)

# load once
model = keras.models.load_model(str(MODEL_PATH))
with open(CLASS_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

def preprocess(img: Image.Image):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@csrf_exempt
def predict(request):
    if request.method != 'POST':
        return JsonResponse({'detail': 'POST only'}, status=405)
    if 'image' not in request.FILES:
        return JsonResponse({'detail': 'Send file with key \"image\"'}, status=400)
    try:
        f = request.FILES['image']
        img = Image.open(f)
        x = preprocess(img)
        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        return JsonResponse({
            'predicted_class': CLASS_NAMES[idx],
            'class_index': idx,
            'confidence': float(round(float(preds[idx]), 4))
        })
    except Exception as e:
        return JsonResponse({'detail': str(e)}, status=500)
