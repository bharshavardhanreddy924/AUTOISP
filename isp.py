import os
import json
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, ToPILImage
from PIL import Image
import time # For timing epochs

# --- User-provided RAW10 Unpacking Function ---
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
        print(f"Warning: File {raw_file_path} is smaller ({raw10_data.size} bytes) than expected ({expected_size} bytes). Padding with zeros.")
        padding = np.zeros(expected_size - raw10_data.size, dtype=np.uint8)
        raw10_data = np.concatenate((raw10_data, padding))
    
    raw10_data = raw10_data[:expected_size] # Truncate if larger, or use up to padded size
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

# --- Configuration ---
RAW_IMG_WIDTH = 4608
RAW_IMG_HEIGHT = 2592
# CRITICAL ASSUMPTION: JSON Black Levels are 16-bit scale. For 10-bit data (0-1023), scale by 2^(16-10) = 64.
# So, a JSON value of 4096 becomes 4096/64 = 64 for 10-bit data.
JSON_BLACK_LEVEL_SCALE_FACTOR = 64
MAX_10_BIT_VAL = 1023.0

# Metadata normalization ranges (rough estimates, tune these based on your dataset statistics)
# Format: (min_val, max_val)
METADATA_RANGES = {
    'SensorBlackLevel': (0.0, 200.0),  # Effective 10-bit black level (e.g., 64 falls here)
    'AnalogueGain': (1.0, 16.0),
    'DigitalGain': (1.0, 16.0),
    'ColourGainR': (0.5, 4.0),
    'ColourGainB': (0.5, 4.0),
    'CCM_0': (-3.0, 3.0), 'CCM_1': (-3.0, 3.0), 'CCM_2': (-3.0, 3.0),
    'CCM_3': (-3.0, 3.0), 'CCM_4': (-3.0, 3.0), 'CCM_5': (-3.0, 3.0),
    'CCM_6': (-3.0, 3.0), 'CCM_7': (-3.0, 3.0), 'CCM_8': (-3.0, 3.0),
    'ExposureTime': (10, 1000000),  # In microseconds
    'ColourTemperature': (1500, 10000) # In Kelvin
}
METADATA_KEYS_ORDERED = [
    'SensorBlackLevel', 'AnalogueGain', 'DigitalGain', 'ColourGainR', 'ColourGainB',
    'CCM_0', 'CCM_1', 'CCM_2', 'CCM_3', 'CCM_4', 'CCM_5', 'CCM_6', 'CCM_7', 'CCM_8',
    'ExposureTime', 'ColourTemperature'
] # 16 features

# --- Helper Functions ---
def get_file_pairs(raw_dir, processed_dir):
    pairs = []
    raw_files = glob.glob(os.path.join(raw_dir, "*.raw"))
    
    if not raw_files:
        print(f"No .raw files found in {raw_dir}. Please check the path.")
        return []

    for raw_path in raw_files:
        basename = os.path.basename(raw_path)
        stem = os.path.splitext(basename)[0]
        
        # Corresponding files
        # Assuming processed images are JPG, change extension if PNG
        processed_img_path = os.path.join(processed_dir, stem + ".jpg") 
        raw_json_path = os.path.join(raw_dir, stem + ".json")
        # Processed JSON is not directly used for model input but good to check existence
        processed_json_path = os.path.join(processed_dir, stem + ".json") 

        if os.path.exists(processed_img_path) and \
           os.path.exists(raw_json_path) and \
           os.path.exists(processed_json_path):
            pairs.append({
                "raw_image": raw_path,
                "raw_json": raw_json_path,
                "processed_image": processed_img_path
            })
        else:
            print(f"Skipping {stem}: Missing one or more corresponding files.")
            if not os.path.exists(processed_img_path): print(f"  Missing: {processed_img_path}")
            if not os.path.exists(raw_json_path): print(f"  Missing: {raw_json_path}")
            if not os.path.exists(processed_json_path): print(f"  Missing: {processed_json_path}")
            
    return pairs

def load_and_normalize_metadata(json_path):
    try:
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return None

    # Extract and normalize features
    norm_meta_features = []
    
    # 1. SensorBlackLevel (use first, scale, then normalize)
    bl_16bit = meta_data.get('SensorBlackLevels', [4096])[0] # Default if missing
    bl_10bit = bl_16bit / JSON_BLACK_LEVEL_SCALE_FACTOR
    min_val, max_val = METADATA_RANGES['SensorBlackLevel']
    norm_meta_features.append(np.clip((bl_10bit - min_val) / (max_val - min_val), 0, 1))

    # 2. AnalogueGain
    ag = meta_data.get('AnalogueGain', 1.0)
    min_val, max_val = METADATA_RANGES['AnalogueGain']
    norm_meta_features.append(np.clip((ag - min_val) / (max_val - min_val), 0, 1))

    # 3. DigitalGain
    dg = meta_data.get('DigitalGain', 1.0)
    min_val, max_val = METADATA_RANGES['DigitalGain']
    norm_meta_features.append(np.clip((dg - min_val) / (max_val - min_val), 0, 1))

    # 4. ColourGains (R and B)
    cg = meta_data.get('ColourGains', [1.0, 1.0])
    min_val_r, max_val_r = METADATA_RANGES['ColourGainR']
    min_val_b, max_val_b = METADATA_RANGES['ColourGainB']
    norm_meta_features.append(np.clip((cg[0] - min_val_r) / (max_val_r - min_val_r), 0, 1)) # R Gain
    norm_meta_features.append(np.clip((cg[1] - min_val_b) / (max_val_b - min_val_b), 0, 1)) # B Gain
    
    # 5. ColourCorrectionMatrix (9 elements)
    ccm = meta_data.get('ColourCorrectionMatrix', [1,0,0,0,1,0,0,0,1]) # Default to identity
    for i in range(9):
        min_val, max_val = METADATA_RANGES[f'CCM_{i}']
        norm_meta_features.append(np.clip((ccm[i] - min_val) / (max_val - min_val), 0, 1))
        
    # 6. ExposureTime
    et = meta_data.get('ExposureTime', 33000)
    min_val, max_val = METADATA_RANGES['ExposureTime']
    norm_meta_features.append(np.clip((et - min_val) / (max_val - min_val), 0, 1))

    # 7. ColourTemperature
    ct = meta_data.get('ColourTemperature', 5000)
    min_val, max_val = METADATA_RANGES['ColourTemperature']
    norm_meta_features.append(np.clip((ct - min_val) / (max_val - min_val), 0, 1))
    
    return np.array(norm_meta_features, dtype=np.float32)


# --- Dataset Class ---
class ISPDataset(Dataset):
    def __init__(self, file_pairs, raw_img_width, raw_img_height, metadata_keys):
        self.file_pairs = file_pairs
        self.raw_img_width = raw_img_width
        self.raw_img_height = raw_img_height
        self.metadata_keys = metadata_keys # For ensuring order if needed later
        self.to_tensor = ToTensor() # Converts numpy HWC [0,255] or HW [0,1] to CHW tensor [0,1]

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        pair = self.file_pairs[idx]

        # Load RAW image
        raw_image_data = unpack_raw10(pair["raw_image"], self.raw_img_width, self.raw_img_height)
        if raw_image_data is None: # unpack_raw10 failed
             # Return a dummy sample or raise an error. For simplicity, skip by returning None for collate_fn to handle
            print(f"Warning: Could not load raw image {pair['raw_image']}, skipping sample {idx}")
            return None


        # Load metadata and get effective black level for raw image processing
        metadata_vec = load_and_normalize_metadata(pair["raw_json"])
        if metadata_vec is None: # load_and_normalize_metadata failed
            print(f"Warning: Could not load metadata {pair['raw_json']}, skipping sample {idx}")
            return None

        # Effective black level for 10-bit data (derived from first element of normalized metadata)
        # This needs to be un-normalized from the metadata_vec to get the actual 10-bit black level
        raw_json_data_for_bl = json.load(open(pair["raw_json"], 'r'))
        bl_16bit = raw_json_data_for_bl.get('SensorBlackLevels', [4096])[0]
        effective_black_level = bl_16bit / JSON_BLACK_LEVEL_SCALE_FACTOR
        
        # Preprocess RAW image:
        # 1. Subtract black level
        # 2. Clip to ensure non-negative (though ideally BL is less than min signal)
        # 3. Normalize to [0, 1]
        raw_image_processed = np.clip(raw_image_data.astype(np.float32) - effective_black_level, 0, None)
        raw_image_normalized = raw_image_processed / (MAX_10_BIT_VAL - effective_black_level)
        raw_image_normalized = np.clip(raw_image_normalized, 0, 1) # Ensure it's in [0,1]

        # Add channel dimension: (H, W) -> (H, W, 1) for ToTensor, or handle in ToTensor
        raw_tensor = self.to_tensor(raw_image_normalized) # Becomes (1, H, W)

        # Load processed image (JPG/PNG)
        # cv2.imread loads as BGR by default
        processed_image = cv2.imread(pair["processed_image"]) 
        if processed_image is None:
            print(f"Warning: Could not load processed image {pair['processed_image']}, skipping sample {idx}")
            return None
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert HWC to CHW for PyTorch
        processed_tensor = self.to_tensor(processed_image_rgb) # Becomes (3, H, W)

        return raw_tensor, torch.from_numpy(metadata_vec), processed_tensor

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] # Filter out None samples
    if not batch:
        return None # Or raise error, or return empty tensors
    return torch.utils.data.dataloader.default_collate(batch)

# --- Model Architecture (U-Net with metadata injection) ---
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
            self.conv = DoubleConv(in_channels, out_channels) # in_channels because of concatenation

    def forward(self, x1, x2): # x1 from upsample, x2 from skip connection
        x1 = self.up(x1)
        # Pad x1 if x2 is larger (due to odd dimensions and integer division in pooling)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetWithMetadata(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, metadata_dim, bilinear=True, base_c=32): # Reduced base_c
        super(UNetWithMetadata, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.metadata_dim = metadata_dim

        # Initial Bayer processing (Space to Depth)
        # Input: (B, 1, H, W) Bayer -> (B, 4, H/2, W/2) for RGGB
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2) 
        
        # Metadata MLP
        # Reduced size for MLP
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, base_c*4), # base_c*4 = 128
            nn.ReLU(),
            nn.Linear(base_c*4, base_c*4) # Output matches bottleneck feature depth roughly
        )

        # U-Net Encoder
        self.inc = DoubleConv(n_channels_in * 4, base_c) # *4 from PixelUnshuffle
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8) 
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor) # Bottleneck

        # U-Net Decoder
        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear)
        self.up4 = Up(base_c*2, base_c, bilinear)
        
        # Final processing to get RGB
        # Output of U-Net is (B, base_c, H/2, W/2)
        # We need (B, 3*4, H/2, W/2) for PixelShuffle to (B, 3, H, W)
        self.pre_shuffle_conv = nn.Conv2d(base_c, n_channels_out * 4, kernel_size=1) # 12 channels for 3*2^2
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.final_activation = nn.Sigmoid() # To ensure output is [0,1]

    def forward(self, x_raw, x_meta):
        # x_raw: (B, 1, H, W)
        # x_meta: (B, metadata_dim)

        x1 = self.pixel_unshuffle(x_raw) # (B, 4, H/2, W/2)
        x1 = self.inc(x1) # (B, base_c, H/2, W/2)
        x2 = self.down1(x1) # (B, base_c*2, H/4, W/4)
        x3 = self.down2(x2) # (B, base_c*4, H/8, W/8)
        x4 = self.down3(x3) # (B, base_c*8, H/16, W/16)
        x5 = self.down4(x4) # (B, base_c*16 // factor, H/32, W/32) - Bottleneck

        # Process and inject metadata at bottleneck
        meta_embedding = self.metadata_mlp(x_meta) # (B, base_c*4)
        # Reshape embedding to be (B, base_c*4, 1, 1) and expand to bottleneck spatial size
        # This is a simple addition; FiLM layers or concatenation are more common
        meta_embedding_spatial = meta_embedding.unsqueeze(-1).unsqueeze(-1)
        meta_embedding_spatial = meta_embedding_spatial.expand_as(x5) # If x5 is (B, base_c*4, H_b, W_b)

        # x5 = x5 + meta_embedding_spatial # Additive injection
        # Concatenation is often more robust for bottleneck
        # Ensure x5 channel dim matches meta_embedding dim if concatenating directly
        # For now, let's assume the down4 output (x5) has base_c*8 channels (factor=1)
        # and metadata_mlp outputs base_c*8 to match for addition.
        # Adjusting for current base_c=32, metadata_mlp output is 128, x5 is 512 or 256.
        # Let's adjust metadata_mlp output and down4 output to be same, e.g., base_c*8 (256)
        # So, self.down4 = Down(base_c*4, base_c*8) if factor = 1
        # self.metadata_mlp -> nn.Linear(metadata_dim, base_c*8), nn.Linear(base_c*8, base_c*8)
        # For simplicity, I'll keep current structure and user can refine metadata injection.
        # A common approach for this simplified injection:
        if x5.shape[1] == meta_embedding_spatial.shape[1]: # Check if channels match for addition
             x5 = x5 + meta_embedding_spatial
        else: # If not, print a warning and proceed without metadata for this run.
             print(f"Warning: Bottleneck features ({x5.shape[1]}) and metadata embedding ({meta_embedding_spatial.shape[1]}) channel mismatch. Skipping metadata addition.")


        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # (B, base_c, H/2, W/2)
        
        x = self.pre_shuffle_conv(x) # (B, 12, H/2, W/2)
        x = self.pixel_shuffle(x)    # (B, 3, H, W)
        
        return self.final_activation(x)


# --- Training ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    best_val_loss = float('inf')
    to_pil = ToPILImage()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (raw_inputs, metadata_inputs, target_outputs) in enumerate(train_loader):
            if raw_inputs is None: # From collate_fn if a sample failed
                print(f"Skipping batch {i} due to data loading error in a sample.")
                continue

            raw_inputs = raw_inputs.to(device)
            metadata_inputs = metadata_inputs.to(device)
            target_outputs = target_outputs.to(device)

            optimizer.zero_grad()
            outputs = model(raw_inputs, metadata_inputs)
            loss = criterion(outputs, target_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0: # Print every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Train Loss: {epoch_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for raw_inputs_val, metadata_inputs_val, target_outputs_val in val_loader:
                if raw_inputs_val is None: continue

                raw_inputs_val = raw_inputs_val.to(device)
                metadata_inputs_val = metadata_inputs_val.to(device)
                target_outputs_val = target_outputs_val.to(device)
                
                outputs_val = model(raw_inputs_val, metadata_inputs_val)
                loss_val = criterion(outputs_val, target_outputs_val)
                val_running_loss += loss_val.item()
            
            # Save one validation image example
            if outputs_val is not None and len(outputs_val) > 0:
                out_img = to_pil(outputs_val[0].cpu()) # Take first image in batch
                target_img = to_pil(target_outputs_val[0].cpu())
                
                # Create a directory for validation images if it doesn't exist
                val_img_dir = "validation_previews"
                os.makedirs(val_img_dir, exist_ok=True)
                
                out_img.save(os.path.join(val_img_dir, f"epoch_{epoch+1}_output.png"))
                target_img.save(os.path.join(val_img_dir, f"epoch_{epoch+1}_target.png"))


        val_loss = val_running_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    print("Training finished.")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    DATASET_BASE_DIR = r"C:\Users\borut\Desktop\dl_approach\PI_Frame\dataset" # Use raw string
    RAW_DIR = os.path.join(DATASET_BASE_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATASET_BASE_DIR, "processed")
    MODEL_SAVE_PATH = "isp_model.pth"

    # Hyperparameters (ADJUST THESE)
    BATCH_SIZE = 1 # Critical for high-res images due to VRAM. Increase if possible.
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50 # Increase for real training
    VAL_SPLIT = 0.1 # 10% for validation

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_file_pairs = get_file_pairs(RAW_DIR, PROCESSED_DIR)
    if not all_file_pairs:
        print("No file pairs found. Exiting.")
        exit()
    
    print(f"Found {len(all_file_pairs)} image pairs.")

    # Split dataset
    num_val = int(len(all_file_pairs) * VAL_SPLIT)
    num_train = len(all_file_pairs) - num_val
    train_pairs, val_pairs = torch.utils.data.random_split(all_file_pairs, [num_train, num_val])
    
    print(f"Training samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")

    train_dataset = ISPDataset(list(train_pairs), RAW_IMG_WIDTH, RAW_IMG_HEIGHT, METADATA_KEYS_ORDERED)
    val_dataset = ISPDataset(list(val_pairs), RAW_IMG_WIDTH, RAW_IMG_HEIGHT, METADATA_KEYS_ORDERED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn) # num_workers=0 for Windows usually safer
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    # Model
    # n_channels_in = 1 (Bayer input, becomes 4 after PixelUnshuffle)
    # n_channels_out = 3 (RGB output)
    model = UNetWithMetadata(n_channels_in=1, n_channels_out=3, metadata_dim=len(METADATA_KEYS_ORDERED), base_c=32).to(device)
    
    # Test metadata injection compatibility with current model structure (adjust if needed)
    # U-Net Bottleneck: self.down4 output is (base_c*16 // factor) channels. With base_c=32, factor=1 (default assuming bilinear=False in Up), it's 32*16 = 512 channels.
    # Metadata MLP output: base_c*4 = 32*4 = 128 channels.
    # These don't match for addition.
    # To fix for addition, ensure metadata_mlp output matches bottleneck channels or change injection strategy.
    # For now, the model has a warning print if they mismatch.
    # A quick fix: change metadata_mlp output to base_c*16 // factor
    # e.g., self.metadata_mlp = nn.Sequential(nn.Linear(metadata_dim, 256), nn.ReLU(), nn.Linear(256, 32*16)) # for base_c=32
    # This needs careful alignment with the U-Net's bottleneck channel count.
    # The provided U-Net uses `factor = 2 if bilinear else 1`. Assuming `bilinear=True` (default), factor=2.
    # Bottleneck `self.down4` outputs `base_c*16 // factor` = `32*16 // 2 = 256` channels.
    # Metadata MLP outputs `base_c*4 = 32*4 = 128` channels.
    # Let's adjust metadata_mlp to output 256 to match the bottleneck.
    model.metadata_mlp = nn.Sequential(
            nn.Linear(model.metadata_dim, 128), # Intermediate layer
            nn.ReLU(),
            nn.Linear(128, 256) # Output 256 channels
        ).to(device)
    # And ensure down4 outputs 256:
    # self.down4 = Down(base_c*8, base_c*16 // factor) -> Down(256, 512 // 2) = Down(256, 256). This is correct.

    criterion = nn.L1Loss() # L1 loss is common for image-to-image, can be robust
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH)

    # --- Example: How to use the trained model for inference ---
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # model.eval()
    # with torch.no_grad():
    #     # Get a sample (e.g., first from validation set)
    #     sample_raw, sample_meta, sample_target = val_dataset[0]
    #     sample_raw = sample_raw.unsqueeze(0).to(device) # Add batch dim
    #     sample_meta = sample_meta.unsqueeze(0).to(device) # Add batch dim
    #
    #     predicted_output = model(sample_raw, sample_meta)
    #     predicted_img_pil = to_pil(predicted_output[0].cpu())
    #     predicted_img_pil.save("inference_example_output.png")
    #     print("Saved one inference example to inference_example_output.png")