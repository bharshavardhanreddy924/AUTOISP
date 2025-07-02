import streamlit as st
import tempfile
from PIL import Image
import os
from ultralytics import YOLO
import streamlit.components.v1 as components
import shutil
import io
import cv2
import numpy as np

st.set_page_config(page_title="3D Panorama Detection", layout="wide")
st.title("🧠 3D Panorama Object Detection Viewer")
st.markdown("Upload a 360° stitched JPEG panorama and apply YOLOv8 object detection.")

uploaded_file = st.file_uploader("📷 Upload JPEG Panorama", type=["jpg", "jpeg"])

model_choice = st.selectbox("🎯 Select YOLOv8 Model", [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8n-seg.pt",
    "yolov8s-seg.pt"
])

model_path = os.path.join(os.getcwd(), model_choice)

if uploaded_file and model_choice:
    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    with open(temp_image_path, 'wb') as f:
        f.write(uploaded_file.read())

    original_image = Image.open(temp_image_path).convert("RGB")
    original_width, original_height = original_image.size
    st.image(original_image, caption="Uploaded Panorama", use_container_width=True)

    model = YOLO(model_path)
    st.subheader("🔍 Running Object Detection (no resizing)...")

    results = model.predict(source=temp_image_path, imgsz=(original_height, original_width), conf=0.3)[0]

    # Convert PIL image to NumPy array (BGR for OpenCV)
    image_np = np.array(original_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    boxes = results.boxes
    class_names = model.names

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().numpy())
        label = class_names[cls]
        
        # Small clean bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Small label tag
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.5
        label_thickness = 1
        label_size, _ = cv2.getTextSize(label, label_font, label_scale, label_thickness)
        cv2.rectangle(image_np, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image_np, label, (x1, y1 - 2), label_font, label_scale, (0, 0, 0), label_thickness, cv2.LINE_AA)

    # Save high-quality JPEG
    detected_path = temp_image_path.replace(".jpg", "_det.jpg")
    cv2.imwrite(detected_path, image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Copy to static folder for viewer
    static_folder = os.path.join(os.getcwd(), "static")
    os.makedirs(static_folder, exist_ok=True)
    static_img_name = os.path.basename(detected_path)
    static_img_path = os.path.join(static_folder, static_img_name)
    shutil.copy(detected_path, static_img_path)

    # Show image
    st.image(detected_path, caption="Detected Objects (Clean)", use_container_width=True)

    # Download option
    with open(detected_path, "rb") as f:
        st.download_button("📥 Download Clean Detected Image", f, file_name="detected_panorama.jpg", mime="image/jpeg")

    # 360 Viewer
    st.subheader("🌀 360° Interactive Viewer")
    components.html(f"""
        <div id="panorama" style="width: 100%; height: 500px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
        <script>
          pannellum.viewer('panorama', {{
            type: 'equirectangular',
            panorama: 'static/{static_img_name}',
            autoLoad: true
          }});
        </script>
    """, height=520)
