from flask import Flask, request, render_template
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load class names from classes.txt
with open("classes.txt", "r") as f:
    class_names = f.read().splitlines()

# Load YOLOv8 model
model = YOLO('yolov8_model.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            # Run prediction
            results = model.predict(source=filepath, save=True, project='static/runs', name='predict', exist_ok=True)
            result_img_name = os.path.basename(filepath)
            result_path = f"static/runs/predict/{result_img_name}"

            # Extract predicted labels using classes.txt
            detected_objects = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                detected_objects.append(label)

            return render_template('result.html', result_image=result_path, labels=detected_objects, time=time.time())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)