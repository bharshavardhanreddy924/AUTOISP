import streamlit as st
import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import tempfile
import traceback
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

# --- Configuration ---
RAW_IMG_WIDTH = 4608
RAW_IMG_HEIGHT = 2592
JSON_BLACK_LEVEL_SCALE_FACTOR = 64.0
MAX_10_BIT_VAL = 1023.0

# Metadata normalization ranges
METADATA_RANGES = {
    'SensorBlackLevel': (0.0, 200.0),
    'AnalogueGain': (1.0, 16.0),
    'DigitalGain': (1.0, 16.0),
    'ColourGainR': (0.5, 4.0),
    'ColourGainB': (0.5, 4.0),
    'CCM_0': (-3.0, 3.0), 'CCM_1': (-3.0, 3.0), 'CCM_2': (-3.0, 3.0),
    'CCM_3': (-3.0, 3.0), 'CCM_4': (-3.0, 3.0), 'CCM_5': (-3.0, 3.0),
    'CCM_6': (-3.0, 3.0), 'CCM_7': (-3.0, 3.0), 'CCM_8': (-3.0, 3.0),
    'ExposureTime': (10.0, 1000000.0),
    'ColourTemperature': (1500.0, 10000.0)
}

METADATA_KEYS_ORDERED = [
    'SensorBlackLevel', 'AnalogueGain', 'DigitalGain', 'ColourGainR', 'ColourGainB',
    'CCM_0', 'CCM_1', 'CCM_2', 'CCM_3', 'CCM_4', 'CCM_5', 'CCM_6', 'CCM_7', 'CCM_8',
    'ExposureTime', 'ColourTemperature'
]

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

ENHANCED_CATEGORY_COLORS = {
    'person': (255, 0, 0), 'bicycle': (0, 255, 0), 'car': (0, 255, 0), 'motorcycle': (0, 255, 0),
    'airplane': (0, 255, 0), 'bus': (0, 255, 0), 'train': (0, 255, 0), 'truck': (0, 255, 0),
    'boat': (0, 255, 0), 'bird': (0, 0, 255), 'cat': (0, 0, 255), 'dog': (0, 0, 255), 'horse': (0, 0, 255),
    'sheep': (0, 0, 255), 'cow': (0, 0, 255), 'elephant': (0, 0, 255), 'bear': (0, 0, 255),
    'zebra': (0, 0, 255), 'giraffe': (0, 0, 255), 'traffic light': (255, 165, 0),
    'fire hydrant': (255, 165, 0), 'stop sign': (255, 0, 255), 'parking meter': (255, 165, 0),
    'bench': (128, 128, 128), 'backpack': (255, 255, 0), 'umbrella': (255, 255, 0),
    'handbag': (255, 255, 0), 'tie': (255, 255, 0), 'suitcase': (255, 255, 0), 'frisbee': (0, 255, 255),
    'skis': (0, 255, 255), 'snowboard': (0, 255, 255), 'sports ball': (0, 255, 255), 'kite': (0, 255, 255),
    'baseball bat': (0, 255, 255), 'baseball glove': (0, 255, 255), 'skateboard': (0, 255, 255),
    'surfboard': (0, 255, 255), 'tennis racket': (0, 255, 255), 'bottle': (128, 0, 128),
    'wine glass': (128, 0, 128), 'cup': (128, 0, 128), 'fork': (128, 0, 128), 'knife': (128, 0, 128),
    'spoon': (128, 0, 128), 'bowl': (128, 0, 128), 'banana': (255, 255, 0), 'apple': (255, 255, 0),
    'sandwich': (255, 255, 0), 'orange': (255, 165, 0), 'broccoli': (0, 128, 0), 'carrot': (255, 165, 0),
    'hot dog': (255, 255, 0), 'pizza': (255, 255, 0), 'donut': (255, 255, 0), 'cake': (255, 255, 0),
    'chair': (139, 69, 19), 'couch': (139, 69, 19), 'potted plant': (0, 128, 0), 'bed': (139, 69, 19),
    'dining table': (139, 69, 19), 'toilet': (192, 192, 192), 'tv': (64, 64, 64), 'laptop': (64, 64, 64),
    'mouse': (64, 64, 64), 'remote': (64, 64, 64), 'keyboard': (64, 64, 64), 'cell phone': (64, 64, 64),
    'microwave': (192, 192, 192), 'oven': (192, 192, 192), 'toaster': (192, 192, 192),
    'sink': (192, 192, 192), 'refrigerator': (192, 192, 192), 'book': (255, 255, 0), 'clock': (255, 255, 0),
    'vase': (128, 0, 128), 'scissors': (255, 255, 0), 'teddy bear': (255, 192, 203),
    'hair drier': (255, 255, 0), 'toothbrush': (255, 255, 0)
}

# --- Data Processing Functions ---

def unpack_raw10(raw_file_path, width, height):
    stride = (width * 10) // 8
    try:
        with open(raw_file_path, 'rb') as f:
            raw10_data = np.fromfile(f, dtype=np.uint8)
    except FileNotFoundError:
        st.error(f"Raw file not found at {raw_file_path}")
        return None
    except Exception as e:
        st.error(f"Error reading raw file: {e}")
        return None

    expected_size = stride * height
    if raw10_data.size < expected_size:
        st.error(f"File is smaller ({raw10_data.size} bytes) than expected ({expected_size} bytes)")
        return None

    raw10_data = raw10_data[:expected_size]
    raw10 = raw10_data.reshape((height, stride))
    unpacked = np.zeros((height, width), dtype=np.uint16)

    for i in range(0, width, 4):
        base_idx = (i // 4) * 5
        b0 = raw10[:, base_idx + 0].astype(np.uint16)
        b1 = raw10[:, base_idx + 1].astype(np.uint16)
        b2 = raw10[:, base_idx + 2].astype(np.uint16)
        b3 = raw10[:, base_idx + 3].astype(np.uint16)
        b4 = raw10[:, base_idx + 4].astype(np.uint16)

        unpacked[:, i + 0] = ((b0 << 2) | ((b4 >> 0) & 0x3)) & 0x3FF
        unpacked[:, i + 1] = ((b1 << 2) | ((b4 >> 2) & 0x3)) & 0x3FF
        unpacked[:, i + 2] = ((b2 << 2) | ((b4 >> 4) & 0x3)) & 0x3FF
        unpacked[:, i + 3] = ((b3 << 2) | ((b4 >> 6) & 0x3)) & 0x3FF
    return unpacked

def load_and_normalize_metadata(json_path):
    try:
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return None

    norm_meta_features = []
    try:
        bl_16bit = meta_data.get('SensorBlackLevels', [4096.0])[0]
        bl_10bit = bl_16bit / JSON_BLACK_LEVEL_SCALE_FACTOR
        min_val, max_val = METADATA_RANGES['SensorBlackLevel']
        norm_meta_features.append(np.clip((bl_10bit - min_val) / (max_val - min_val + 1e-6), 0, 1))

        ag = meta_data.get('AnalogueGain', 1.0)
        min_val, max_val = METADATA_RANGES['AnalogueGain']
        norm_meta_features.append(np.clip((ag - min_val) / (max_val - min_val + 1e-6), 0, 1))

        dg = meta_data.get('DigitalGain', 1.0)
        min_val, max_val = METADATA_RANGES['DigitalGain']
        norm_meta_features.append(np.clip((dg - min_val) / (max_val - min_val + 1e-6), 0, 1))

        cg = meta_data.get('ColourGains', [1.0, 1.0])
        min_val_r, max_val_r = METADATA_RANGES['ColourGainR']
        min_val_b, max_val_b = METADATA_RANGES['ColourGainB']
        norm_meta_features.append(np.clip((cg[0] - min_val_r) / (max_val_r - min_val_r + 1e-6), 0, 1))
        norm_meta_features.append(np.clip((cg[1] - min_val_b) / (max_val_b - min_val_b + 1e-6), 0, 1))

        ccm_data = meta_data.get('ColourCorrectionMatrix', [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
        for i in range(9):
            min_val, max_val = METADATA_RANGES[f'CCM_{i}']
            norm_meta_features.append(np.clip((ccm_data[i] - min_val) / (max_val - min_val + 1e-6), 0, 1))

        et = meta_data.get('ExposureTime', 33000.0)
        min_val, max_val = METADATA_RANGES['ExposureTime']
        norm_meta_features.append(np.clip((et - min_val) / (max_val - min_val + 1e-6), 0, 1))

        ct = meta_data.get('ColourTemperature', 5000.0)
        min_val, max_val = METADATA_RANGES['ColourTemperature']
        norm_meta_features.append(np.clip((ct - min_val) / (max_val - min_val + 1e-6), 0, 1))
    except KeyError as e:
        st.error(f"Metadata key error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error processing metadata: {e}")
        return None

    return np.array(norm_meta_features, dtype=np.float32)

def create_raw_preview(raw_image_data):
    if raw_image_data is None:
        return None

    normalized_8bit = np.clip(raw_image_data / MAX_10_BIT_VAL * 255, 0, 255).astype(np.uint8)

    try:
        bgr_image = cv2.demosaicing(normalized_8bit, cv2.COLOR_BAYER_RG2BGR)
    except Exception as e:
        st.warning(f"Could not demosaic RAW preview, showing grayscale instead. Error: {e}")
        return Image.fromarray(normalized_8bit, mode='L')
    
    bgr_float = bgr_image.astype(np.float32)
    b_avg, g_avg, r_avg = np.mean(bgr_float[:, :, 0]), np.mean(bgr_float[:, :, 1]), np.mean(bgr_float[:, :, 2])
    
    if g_avg > 1e-6:
        gray_avg = g_avg
        r_scale = gray_avg / (r_avg + 1e-6)
        b_scale = gray_avg / (b_avg + 1e-6)
        bgr_float[:, :, 0] *= b_scale
        bgr_float[:, :, 2] *= r_scale

    white_balanced_bgr = np.clip(bgr_float, 0, 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(white_balanced_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image, mode='RGB')

    return pil_image

# --- Model Architectures ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithMetadata(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, metadata_dim, bilinear=True, base_c=32):
        super(UNetWithMetadata, self).__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        factor = 2 if bilinear else 1
        bottleneck_ch = base_c * 16 // factor
        intermediate_dim_mlp = max(metadata_dim * 2, bottleneck_ch // 2, 64)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, intermediate_dim_mlp), nn.ReLU(),
            nn.Linear(intermediate_dim_mlp, bottleneck_ch)
        )
        self.inc = DoubleConv(n_channels_in * 4, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.down4 = Down(base_c*8, bottleneck_ch)
        self.up1 = Up(bottleneck_ch + base_c*8, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8 // factor + base_c*4, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4 // factor + base_c*2, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2 // factor + base_c, base_c, bilinear)
        self.pre_shuffle_conv = nn.Conv2d(base_c, n_channels_out * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.final_activation = nn.Sigmoid()

    def forward(self, x_raw, x_meta):
        x1_unshuffled = self.pixel_unshuffle(x_raw)
        x1_inc = self.inc(x1_unshuffled)
        x2_down = self.down1(x1_inc)
        x3_down = self.down2(x2_down)
        x4_down = self.down3(x3_down)
        x5_bottleneck = self.down4(x4_down)
        meta_embedding = self.metadata_mlp(x_meta).unsqueeze(-1).unsqueeze(-1)
        x5_fused = x5_bottleneck + meta_embedding.expand(-1, -1, x5_bottleneck.size(2), x5_bottleneck.size(3))
        x_up1 = self.up1(x5_fused, x4_down)
        x_up2 = self.up2(x_up1, x3_down)
        x_up3 = self.up3(x_up2, x2_down)
        x_up4 = self.up4(x_up3, x1_inc)
        x_preshuffle = self.pre_shuffle_conv(x_up4)
        x_shuffled = self.pixel_shuffle(x_preshuffle)
        return self.final_activation(x_shuffled)

# --- Model Loading Functions ---

@st.cache_resource
def load_isp_model(device):
    model_path = "isp_model.pth"
    if not os.path.exists(model_path):
        st.error(f"ISP Model file not found: {model_path}")
        return None
    model = UNetWithMetadata(n_channels_in=1, n_channels_out=3, metadata_dim=len(METADATA_KEYS_ORDERED), base_c=32).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ISP model: {e}")
        return None

@st.cache_resource
def download_and_load_yolo_models():
    models = {}
    model_configs = {'YOLOv8n (Fastest)': 'yolov8n.pt', 'YOLOv8s (Balanced)': 'yolov8s.pt', 'YOLOv8m (Accurate)': 'yolov8m.pt'}
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, (name, model_path) in enumerate(model_configs.items()):
        status_text.info(f"Checking for {name} model...")
        try:
            model = YOLO(model_path)
            models[name] = model
        except Exception as e:
            st.error(f"Error loading or downloading {name} model: {e}")
        progress_bar.progress((i + 1) / len(model_configs))
    status_text.success("Object detection models are ready!")
    progress_bar.empty()
    return models

# --- Core Inference and Detection Logic ---

def enhance_image_for_detection(image):
    enhancer = ImageEnhance.Contrast(image)
    enhanced = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(1.1)
    return enhanced

def advanced_object_detection(image, yolo_models, model_name='YOLOv8n (Fastest)', confidence_threshold=0.5, iou_threshold=0.4):
    try:
        yolo_model = yolo_models[model_name]
        enhanced_image = enhance_image_for_detection(image)
        results = yolo_model(np.array(enhanced_image), conf=confidence_threshold, iou=iou_threshold, verbose=False)
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.truetype("arial.ttf", max(15, min(image.width, image.height) // 100))
        except IOError:
            font = ImageFont.load_default()

        detection_stats = {'total_objects': 0, 'category_counts': {}, 'objects_detected': [], 'confidence_distribution': []}
        categories = {'Humans': ['person'], 'Animals': ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'], 'Vehicles': ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat'], 'Electronics': ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone'], 'Furniture': ['chair', 'couch', 'bed', 'dining table'], 'Food': ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake'], 'Sports': ['frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'], 'Traffic': ['traffic light', 'fire hydrant', 'stop sign'], 'Accessories': ['backpack', 'umbrella', 'handbag', 'tie', 'suitcase'], 'Kitchen': ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl'], 'Other': []}
        for category in categories:
            detection_stats['category_counts'][category] = 0

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = COCO_CLASSES[class_id]
                    detection_stats['total_objects'] += 1
                    detection_stats['confidence_distribution'].append(confidence)
                    detection_stats['objects_detected'].append({'class': class_name, 'confidence': confidence, 'bbox': [float(x) for x in [x1, y1, x2, y2]]})
                    
                    assigned_category = 'Other'
                    for category, class_list in categories.items():
                        if class_name in class_list:
                            assigned_category = category
                            break
                    detection_stats['category_counts'][assigned_category] += 1
                    
                    color = ENHANCED_CATEGORY_COLORS.get(class_name, (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, min(image.width, image.height) // 500))
                    label = f"{class_name}: {confidence:.2f}"
                    text_bbox = draw.textbbox((x1, y1), label, font=font)
                    draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
                    draw.text((x1, y1 - (text_bbox[3] - text_bbox[1])), label, fill=(0, 0, 0), font=font)

        return annotated_image, detection_stats
    except Exception as e:
        st.error(f"Error during object detection: {e}")
        traceback.print_exc()
        return image, {'total_objects': 0, 'category_counts': {}, 'objects_detected': []}

def run_pipeline(isp_model, yolo_models, raw_file, json_file, device, enable_detection=True,
                 model_name='YOLOv8n (Fastest)', confidence_threshold=0.5, iou_threshold=0.4):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as temp_raw, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as temp_json:
        temp_raw.write(raw_file.getvalue())
        temp_raw_path = temp_raw.name
        json_content = json.loads(json_file.getvalue())
        json.dump(json_content, temp_json)
        temp_json_path = temp_json.name

    try:
        raw_image_data = unpack_raw10(temp_raw_path, RAW_IMG_WIDTH, RAW_IMG_HEIGHT)
        if raw_image_data is None: return (None,) * 7
        raw_preview = create_raw_preview(raw_image_data)

        metadata_vec = load_and_normalize_metadata(temp_json_path)
        if metadata_vec is None: return raw_preview, None, None, None, None, None, None
        
        bl_16bit = json_content.get('SensorBlackLevels', [4096.0])[0]
        effective_black_level = bl_16bit / JSON_BLACK_LEVEL_SCALE_FACTOR
        raw_image_processed = np.clip(raw_image_data.astype(np.float32) - effective_black_level, 0, None)
        denominator = (MAX_10_BIT_VAL - effective_black_level)
        raw_image_normalized = np.clip(raw_image_processed / denominator, 0.0, 1.0)
        
        raw_tensor = ToTensor()(raw_image_normalized.astype(np.float32)).unsqueeze(0).to(device)
        metadata_tensor = torch.from_numpy(metadata_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_tensor = isp_model(raw_tensor, metadata_tensor)
        
        output_numpy = (predicted_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        processed_image = Image.fromarray(output_numpy)

        annotated_raw, stats_raw, annotated_isp, stats_isp = None, None, None, None
        if enable_detection and yolo_models:
            annotated_raw, stats_raw = advanced_object_detection(raw_preview, yolo_models, model_name, confidence_threshold, iou_threshold)
            annotated_isp, stats_isp = advanced_object_detection(processed_image, yolo_models, model_name, confidence_threshold, iou_threshold)
        
        return raw_preview, processed_image, annotated_raw, stats_raw, annotated_isp, stats_isp, metadata_vec

    except Exception as e:
        st.error(f"An error occurred in the processing pipeline: {e}")
        traceback.print_exc()
        return (None,) * 7
    finally:
        os.unlink(temp_raw_path)
        os.unlink(temp_json_path)

# --- UI Display Components ---

def display_image_comparison(raw_preview, processed_image):
    st.header("🖼️ Image Comparison: RAW vs. ISP")
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_preview, caption="RAW Preview (Demosaiced & White-Balanced)", use_container_width=True)
    with col2:
        st.image(processed_image, caption="ISP Processed RGB Image", use_container_width=True)

def display_detection_comparison(annotated_raw, annotated_isp):
    if annotated_raw is None or annotated_isp is None:
        return
    st.header("🎯 Object Detection Results")
    st.markdown("Detection performed on both the simple RAW preview and the final ISP-processed image.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated_raw, caption="Detections on RAW Preview", use_container_width=True)
    with col2:
        st.image(annotated_isp, caption="Detections on ISP Output", use_container_width=True)

def display_single_detection_analytics(detection_stats, title):
    st.subheader(title)
    if not detection_stats or detection_stats['total_objects'] == 0:
        st.info("No objects were detected for this image.")
        return

    tabs = st.tabs(["Summary", "Categories", "Details"])
    
    with tabs[0]:
        st.metric("Total Objects Detected", detection_stats['total_objects'])
        avg_conf = np.mean(detection_stats['confidence_distribution']) if detection_stats['confidence_distribution'] else 0
        st.metric("Average Confidence", f"{avg_conf:.3f}")
        if len(detection_stats['confidence_distribution']) > 1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(detection_stats['confidence_distribution'], bins=15, color="#4ECDC4", edgecolor='black')
            ax.set_title("Confidence Distribution", fontsize=10)
            st.pyplot(fig)
            
    with tabs[1]:
        category_data = {k: v for k, v in detection_stats['category_counts'].items() if v > 0}
        if category_data:
            df_cat = pd.DataFrame(list(category_data.items()), columns=['Category', 'Count']).sort_values('Count', ascending=False)
            st.dataframe(df_cat, use_container_width=True, hide_index=True)

    with tabs[2]:
        if detection_stats['objects_detected']:
            sorted_objects = sorted(detection_stats['objects_detected'], key=lambda x: x['confidence'], reverse=True)
            df_data = [{'Object': obj['class'].title(), 'Confidence': f"{obj['confidence']:.2%}"} for obj in sorted_objects]
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=300, hide_index=True)

def display_metadata_info(json_file):
    if json_file is None:
        st.warning("Metadata not available.")
        return
    st.header("📋 Camera & Scene Metadata")
    json_file.seek(0)
    json_content = json.load(json_file)
    tabs = st.tabs(["Key Parameters", "Color Correction Matrix (CCM)", "Full JSON"])
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Exposure Time (µs)", f"{json_content.get('ExposureTime', 'N/A'):,}")
            st.metric("Analogue Gain", f"{json_content.get('AnalogueGain', 'N/A'):.2f}x")
        with col2:
            st.metric("Color Temperature (K)", f"{json_content.get('ColourTemperature', 'N/A'):,}")
            bl_16bit = json_content.get('SensorBlackLevels', [0])[0]
            st.metric("Black Level (16-bit)", f"{bl_16bit:.0f}")
    with tabs[1]:
        ccm = np.array(json_content.get('ColourCorrectionMatrix', np.eye(3).flatten().tolist())).reshape(3, 3)
        st.table(pd.DataFrame(ccm, columns=['R', 'G', 'B'], index=['R_out', 'G_out', 'B_out']).round(4))
    with tabs[2]:
        st.json(json_content)

def create_download_options(processed_image, annotated_raw, annotated_isp, raw_filename=""):
    st.header("💾 Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        if processed_image:
            img_buffer = BytesIO()
            processed_image.save(img_buffer, format='PNG')
            st.download_button(label="📥 Download ISP Image", data=img_buffer, file_name=f"isp_{Path(raw_filename).stem}.png", mime="image/png", use_container_width=True)
    with col2:
        if annotated_raw:
            img_buffer = BytesIO()
            annotated_raw.save(img_buffer, format='PNG')
            st.download_button(label="📥 Download Annotated RAW", data=img_buffer, file_name=f"annotated_raw_{Path(raw_filename).stem}.png", mime="image/png", use_container_width=True)
    with col3:
        if annotated_isp:
            img_buffer = BytesIO()
            annotated_isp.save(img_buffer, format='PNG')
            st.download_button(label="📥 Download Annotated ISP", data=img_buffer, file_name=f"annotated_isp_{Path(raw_filename).stem}.png", mime="image/png", use_container_width=True)

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Advanced ISP & Detection Pipeline", page_icon="🤖", layout="wide")

    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: **{str(device).upper()}**")
        with st.expander("Model Loading Status", expanded=True):
            isp_model = load_isp_model(device)
            if isp_model is not None: st.success("ISP model loaded!")
            yolo_models = download_and_load_yolo_models()
            if not yolo_models: st.warning("Object detection models failed to load.")
        st.markdown("---")
        st.subheader("🎯 Detection Settings")
        enable_detection = st.checkbox("Enable Object Detection", value=True, disabled=(not yolo_models))
        model_name, confidence_threshold, iou_threshold = 'YOLOv8n (Fastest)', 0.5, 0.4
        if enable_detection and yolo_models:
            model_name = st.selectbox("YOLO Model", options=list(yolo_models.keys()), index=0)
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
            iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.4, 0.05)
        
        # --- MODIFIED SECTION: Quick Links with Buttons ---
        st.markdown("---")
        st.title("🔗 Other ISP Tools")
        st.link_button("🛠️ Manual ISP", "https://manual-isp.streamlit.app/", use_container_width=True)
        st.link_button("🔧 Denoising & Sharpness", "https://denoising-and-sharpness.streamlit.app/", use_container_width=True)
        st.link_button("🌅 HDR Imaging", "https://hdrimaging.streamlit.app/", use_container_width=True)
        # --- END OF MODIFIED SECTION ---

    st.title("📸 Advanced ISP & Object Detection Pipeline")
    st.markdown("This application transforms RAW sensor data into a high-quality RGB image using a deep learning ISP, then performs and compares object detection on both the RAW preview and the final ISP image.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        raw_file = st.file_uploader("1. Upload RAW Image", type=['raw'])
    with col2:
        json_file = st.file_uploader("2. Upload JSON Metadata", type=['json'])

    if raw_file is not None and json_file is not None:
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            if isp_model is None:
                st.error("Cannot process. The ISP model is not loaded.")
                return
            with st.spinner("Running the full pipeline... This may take a moment."):
                raw_preview, proc_img, ann_raw, stats_raw, ann_isp, stats_isp, meta_vec = run_pipeline(
                    isp_model, yolo_models, raw_file, json_file, device,
                    enable_detection, model_name, confidence_threshold, iou_threshold
                )

            if proc_img is None:
                st.error("Processing failed. Please check the logs and file formats.")
            else:
                st.success("✅ Pipeline executed successfully!")
                st.markdown("---")
                
                display_image_comparison(raw_preview, proc_img)
                st.markdown("---")
                
                if enable_detection:
                    display_detection_comparison(ann_raw, ann_isp)
                    st.markdown("---")
                
                create_download_options(proc_img, ann_raw, ann_isp, raw_file.name)
                st.markdown("---")
                
                if enable_detection and (stats_raw or stats_isp):
                    st.header("📊 Detection Analytics Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        display_single_detection_analytics(stats_raw, "On RAW Preview")
                    with col2:
                        display_single_detection_analytics(stats_isp, "On ISP Output")
                    st.markdown("---")

                display_metadata_info(json_file)
    else:
        st.info("👆 Please upload both a RAW file and a JSON file to begin.")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #888;'>Built with Streamlit, PyTorch, and YOLO</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()