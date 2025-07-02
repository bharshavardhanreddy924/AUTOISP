import os
import json
import glob # Not strictly needed for test_model.py but good to keep consistency if functions are shared
import numpy as np
import cv2
import torch
import torch.nn as nn
# No optimizer needed for inference
from torchvision.transforms import ToTensor # Only ToTensor needed for inference preprocessing
from PIL import Image # For saving the output image

# --- Configuration (Should match training script exactly for model architecture and preprocessing) ---
RAW_IMG_WIDTH = 4608
RAW_IMG_HEIGHT = 2592
JSON_BLACK_LEVEL_SCALE_FACTOR = 64.0 # Ensure float
MAX_10_BIT_VAL = 1023.0

# Metadata normalization ranges (MUST be identical to those used during training)
METADATA_RANGES = {
    'SensorBlackLevel': (0.0, 200.0),
    'AnalogueGain': (1.0, 16.0),
    'DigitalGain': (1.0, 16.0),
    'ColourGainR': (0.5, 4.0),
    'ColourGainB': (0.5, 4.0),
    'CCM_0': (-3.0, 3.0), 'CCM_1': (-3.0, 3.0), 'CCM_2': (-3.0, 3.0),
    'CCM_3': (-3.0, 3.0), 'CCM_4': (-3.0, 3.0), 'CCM_5': (-3.0, 3.0),
    'CCM_6': (-3.0, 3.0), 'CCM_7': (-3.0, 3.0), 'CCM_8': (-3.0, 3.0),
    'ExposureTime': (10.0, 1000000.0), # Use floats for consistency
    'ColourTemperature': (1500.0, 10000.0)
}
METADATA_KEYS_ORDERED = [
    'SensorBlackLevel', 'AnalogueGain', 'DigitalGain', 'ColourGainR', 'ColourGainB',
    'CCM_0', 'CCM_1', 'CCM_2', 'CCM_3', 'CCM_4', 'CCM_5', 'CCM_6', 'CCM_7', 'CCM_8',
    'ExposureTime', 'ColourTemperature'
]

# --- RAW10 Unpacking Function (Identical to training script) ---
def unpack_raw10(raw_file_path, width, height):
    stride = (width * 10) // 8
    try:
        with open(raw_file_path, 'rb') as f:
            raw10_data = np.fromfile(f, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Raw file not found at {raw_file_path}")
        return None
    except Exception as e:
        print(f"Error reading raw file {raw_file_path}: {e}")
        return None

    expected_size = stride * height
    if raw10_data.size < expected_size:
        # For inference, if the file is bad, we probably shouldn't pad.
        print(f"Error: File {raw_file_path} is smaller ({raw10_data.size} bytes) than expected ({expected_size} bytes). Cannot process.")
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

# --- Metadata Loading and Normalization (Identical to training script) ---
def load_and_normalize_metadata(json_path):
    try:
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing JSON {json_path}: {e}")
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
        print(f"Metadata key error in {json_path}: {e}. Check METADATA_RANGES and JSON content.")
        return None
    except Exception as e:
        print(f"Unexpected error processing metadata for {json_path}: {e}")
        return None
        
    return np.array(norm_meta_features, dtype=np.float32)

# --- Model Architecture (Identical to training script) ---
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
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.metadata_dim = metadata_dim

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2) 
        
        factor = 2 if bilinear else 1
        bottleneck_ch = base_c * 16 // factor
        intermediate_dim_mlp = max(metadata_dim * 2, bottleneck_ch // 2, 64)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, intermediate_dim_mlp),
            nn.ReLU(),
            nn.Linear(intermediate_dim_mlp, bottleneck_ch) 
        )

        self.inc = DoubleConv(n_channels_in * 4, base_c) 
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8) 
        self.down4 = Down(base_c*8, bottleneck_ch)

        # Corrected Up layer in_channels for concatenation logic within Up
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

        meta_embedding = self.metadata_mlp(x_meta) 
        meta_embedding_spatial = meta_embedding.unsqueeze(-1).unsqueeze(-1)
        
        if x5_bottleneck.shape[1] == meta_embedding_spatial.shape[1]:
            x5_fused = x5_bottleneck + meta_embedding_spatial.expand(-1, -1, x5_bottleneck.size(2), x5_bottleneck.size(3))
        else:
             print(f"Warning: Bottleneck features ({x5_bottleneck.shape[1]}) and metadata embedding ({meta_embedding_spatial.shape[1]}) channel mismatch. Check MLP and bottleneck channel calculations. Proceeding without metadata fusion.")
             x5_fused = x5_bottleneck

        x_up1 = self.up1(x5_fused, x4_down)
        x_up2 = self.up2(x_up1, x3_down)
        x_up3 = self.up3(x_up2, x2_down)
        x_up4 = self.up4(x_up3, x1_inc)
        
        x_preshuffle = self.pre_shuffle_conv(x_up4) 
        x_shuffled = self.pixel_shuffle(x_preshuffle)    
        
        return self.final_activation(x_shuffled)

# --- Inference Function ---
def run_inference_on_single_image(model_weights_path, raw_image_path, raw_json_path, output_image_path, device):
    print(f"Running inference for: {os.path.basename(raw_image_path)}")

    # 1. Instantiate the model (ensure parameters match the trained model)
    #    The `base_c` and `bilinear` settings MUST match what was used for training the loaded .pth file.
    #    If you trained with base_c=32 and bilinear=True, use those here.
    model = UNetWithMetadata(
        n_channels_in=1, 
        n_channels_out=3, 
        metadata_dim=len(METADATA_KEYS_ORDERED), 
        base_c=32,        # <<<< MAKE SURE THIS MATCHES THE TRAINED MODEL
        bilinear=True     # <<<< MAKE SURE THIS MATCHES THE TRAINED MODEL
    ).to(device)

    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        print(f"Model weights loaded successfully from: {model_weights_path}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_weights_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval() # Set the model to evaluation mode

    # 3. Load and preprocess the raw image
    raw_image_data = unpack_raw10(raw_image_path, RAW_IMG_WIDTH, RAW_IMG_HEIGHT)
    if raw_image_data is None:
        print(f"Failed to unpack raw image: {raw_image_path}")
        return

    # 4. Load and preprocess metadata
    metadata_vec = load_and_normalize_metadata(raw_json_path)
    if metadata_vec is None:
        print(f"Failed to load or normalize metadata: {raw_json_path}")
        return
    
    try:
        with open(raw_json_path, 'r') as f_json:
            raw_json_content = json.load(f_json)
        bl_16bit = raw_json_content.get('SensorBlackLevels', [4096.0])[0]
        effective_black_level = bl_16bit / JSON_BLACK_LEVEL_SCALE_FACTOR
    except Exception as e:
        print(f"Error reading black level from {raw_json_path}: {e}")
        return

    raw_image_processed = np.clip(raw_image_data.astype(np.float32) - effective_black_level, 0, None)
    denominator = (MAX_10_BIT_VAL - effective_black_level)
    if denominator <= 0:
        print(f"Error: Black level ({effective_black_level}) is too high relative to MAX_10_BIT_VAL ({MAX_10_BIT_VAL}). Cannot normalize.")
        return
    else:
        raw_image_normalized = raw_image_processed / denominator
    raw_image_normalized = np.clip(raw_image_normalized, 0.0, 1.0)
    
    to_tensor_transform = ToTensor()
    raw_tensor = to_tensor_transform(raw_image_normalized.astype(np.float32))
    metadata_tensor = torch.from_numpy(metadata_vec)

    raw_tensor = raw_tensor.unsqueeze(0).to(device)
    metadata_tensor = metadata_tensor.unsqueeze(0).to(device)

    # 5. Perform inference
    print("Performing inference on the model...")
    with torch.no_grad():
        predicted_output_tensor = model(raw_tensor, metadata_tensor)
    print("Inference completed.")

    # 6. Convert output tensor to PIL Image and save
    try:
        # The ToPILImage transform expects a (C, H, W) tensor or (H, W) if single channel.
        # Ensure it's on CPU before converting.
        pil_image_converter = Image.fromarray # Using Image.fromarray for more control if needed
        
        # Convert tensor to numpy array, scale to 0-255, and change order from CHW to HWC
        output_numpy = predicted_output_tensor[0].cpu().numpy() # Get first image in batch, move to CPU, convert to numpy
        output_numpy = np.transpose(output_numpy, (1, 2, 0)) # CHW to HWC
        output_numpy = (output_numpy * 255).astype(np.uint8) # Scale to 0-255 and convert to uint8
        
        predicted_pil_image = Image.fromarray(output_numpy) # Create PIL image from HWC numpy array

        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        predicted_pil_image.save(output_image_path)
        print(f"Output image saved to: {output_image_path}")
    except Exception as e:
        print(f"Error converting tensor to PIL image or saving: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution for Testing ---
if __name__ == "__main__":
    # --- Configuration for Testing ---
    MODEL_WEIGHTS_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\isp_model.pth" # Path to your trained model

    # --- Specify the RAW image and its JSON metadata you want to test ---
    # You MUST provide these paths
    # Example:
    # TEST_RAW_IMAGE_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\dataset\raw\capture_20250517_052619.raw"
    # TEST_RAW_JSON_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\dataset\raw\capture_20250517_052619.json"
    
    # Using a placeholder, REPLACE with your actual file paths:
    TEST_RAW_IMAGE_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\dataset\raw\capture_20250517_052619.raw"
    TEST_RAW_JSON_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\dataset\raw\capture_20250517_052619.json"
    
    # --- Specify where to save the output PNG image ---
    # Example:
    # OUTPUT_IMAGE_SAVE_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\inference_results\output_capture_20250517_052619.png"
    
    # Using a placeholder, REPLACE with your desired output path:
    OUTPUT_IMAGE_SAVE_PATH = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\op\output_capture_20250521_170918.png"


    # --- Setup Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Validate Paths ---
    if "REPLACE_WITH_YOUR" in TEST_RAW_IMAGE_PATH or \
       "REPLACE_WITH_YOUR" in TEST_RAW_JSON_PATH or \
       "REPLACE_WITH_YOUR" in OUTPUT_IMAGE_SAVE_PATH:
        print("-" * 50)
        print("!!! IMPORTANT: Please update the placeholder paths !!!")
        print(f"  TEST_RAW_IMAGE_PATH  : {TEST_RAW_IMAGE_PATH}")
        print(f"  TEST_RAW_JSON_PATH   : {TEST_RAW_JSON_PATH}")
        print(f"  OUTPUT_IMAGE_SAVE_PATH: {OUTPUT_IMAGE_SAVE_PATH}")
        print("-" * 50)
        exit()

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        exit()
    if not os.path.exists(TEST_RAW_IMAGE_PATH):
        print(f"Error: Test raw image not found at {TEST_RAW_IMAGE_PATH}")
        exit()
    if not os.path.exists(TEST_RAW_JSON_PATH):
        print(f"Error: Test raw JSON not found at {TEST_RAW_JSON_PATH}")
        exit()

    # --- Run Inference ---
    run_inference_on_single_image(
        model_weights_path=MODEL_WEIGHTS_PATH,
        raw_image_path=TEST_RAW_IMAGE_PATH,
        raw_json_path=TEST_RAW_JSON_PATH,
        output_image_path=OUTPUT_IMAGE_SAVE_PATH,
        device=DEVICE
    )

    print("\nTest script finished.")