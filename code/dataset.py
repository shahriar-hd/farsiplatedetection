import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import yaml
from pathlib import Path
# ==========================================
# CONFIGURATION
# ==========================================

SOURCE_DIRS = [
    "car_img",
    "plate_img",
    "plate_img_dummy"
]

# Output directory for YOLO formatted dataset
OUTPUT_DIR = "yolo_dataset"

# Define the split mapping: Original folder name -> YOLO subfolder
SPLIT_MAPPING = {
    "train": "train",
    "test": "test",
    "validation": "val"
}

# ------------------------------------------
# CLASS MAPPING (Persian -> English ID)
# ------------------------------------------
# This dictionary maps Persian labels to English class names.
# YOLO requires class indices (0, 1, 2...), so we first define the names list.

CLASS_NAMES = [
    "plate",      # 0: کل ناحیه پلاک
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", # 1-10: Numbers
    "alef",       # الف
    "be",         # ب
    "pe",         # پ
    "te",         # ت
    "se",         # ث
    "jim",        # ج
    "dal",        # د
    "sin",        # س
    "shin",       # ش
    "sad",        # ص
    "ta",         # ط
    "za",         # ظ
    "ein",        # ع
    "fe",         # ف
    "ghaf",       # ق
    "kaf",        # ک
    "gaf",        # گ
    "lam",        # ل
    "mim",        # م
    "noon",       # ن
    "vav",        # و
    "he",         # ه
    "ye",         # ی
    "ze",         # ز
    "taxi",       # ت (تاکسی) often represented as 't' or specific symbol
    "disabled",   # معلولین
    "police",     # پ (پلیس)
    "D",          # Diplomatic
    "S",          # Service
    "protocol",   # تشریفات
]

# Create a lookup dictionary: Label String -> Class Index
LABEL_MAP = {
    "کل ناحیه پلاک": CLASS_NAMES.index("plate"),
    # Numbers
    "0": CLASS_NAMES.index("0"), "1": CLASS_NAMES.index("1"), 
    "2": CLASS_NAMES.index("2"), "3": CLASS_NAMES.index("3"),
    "4": CLASS_NAMES.index("4"), "5": CLASS_NAMES.index("5"), 
    "6": CLASS_NAMES.index("6"), "7": CLASS_NAMES.index("7"),
    "8": CLASS_NAMES.index("8"), "9": CLASS_NAMES.index("9"),
    # Persian Alphabets
    "الف": CLASS_NAMES.index("alef"),
    "ب": CLASS_NAMES.index("be"),
    "پ": CLASS_NAMES.index("pe"),
    "ت": CLASS_NAMES.index("te"),
    "ث": CLASS_NAMES.index("se"),
    "ج": CLASS_NAMES.index("jim"),
    "د": CLASS_NAMES.index("dal"),
    "س": CLASS_NAMES.index("sin"),
    "ش": CLASS_NAMES.index("shin"),
    "ص": CLASS_NAMES.index("sad"),
    "ط": CLASS_NAMES.index("ta"),
    "ظ": CLASS_NAMES.index("za"),
    "ع": CLASS_NAMES.index("ein"),
    "ف": CLASS_NAMES.index("fe"),
    "ق": CLASS_NAMES.index("ghaf"),
    "ک": CLASS_NAMES.index("kaf"),
    "گ": CLASS_NAMES.index("gaf"),
    "ل": CLASS_NAMES.index("lam"),
    "م": CLASS_NAMES.index("mim"),
    "ن": CLASS_NAMES.index("noon"),
    "و": CLASS_NAMES.index("vav"),
    "ه‍": CLASS_NAMES.index("he"),
    "ه": CLASS_NAMES.index("he"),
    "ی": CLASS_NAMES.index("ye"),
    "ز": CLASS_NAMES.index("ze"),
    # Special cases
    "ژ (معلولین و جانبازان)": CLASS_NAMES.index("disabled"),
    "تشریفات": CLASS_NAMES.index("protocol"), # Example for diplomatic
    "S": CLASS_NAMES.index("S"),
    "D": CLASS_NAMES.index("D"),
}
# ==========================================
# HELPER FUNCTIONS
# ==========================================

def convert_box(size, box):
    """
    Converts (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height).
    All values are normalized between 0 and 1.
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x_center = x_center * dw
    w = w * dw
    y_center = y_center * dh
    h = h * dh
    
    return (x_center, y_center, w, h)


# ==========================================
# THE CORRECTED MAIN PROCESSING FUNCTION
# ==========================================

def process_dataset():
    print("Starting dataset conversion...")
    
    # 1. Create base output directories for YOLO structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # 2. Iterate over source directories
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"Warning: Directory '{source_dir}' not found. Skipping.")
            continue

        # 3. Iterate over splits
        for original_split, yolo_split in SPLIT_MAPPING.items():
            current_path = os.path.join(source_dir, original_split)
            
            if not os.path.exists(current_path):
                continue
            
            print(f"Processing: {current_path} -> {yolo_split}")
            
            files = os.listdir(current_path)
            # Filter distinct filenames (ignoring extension) to pair xml and jpg
            filenames = set([os.path.splitext(f)[0] for f in files if f.endswith('.xml')])

            for name in filenames:
                jpg_file = os.path.join(current_path, name + ".jpg")
                xml_file = os.path.join(current_path, name + ".xml")
                
                # Check if image exists
                if not os.path.exists(jpg_file):
                    print(f"Missing image for {xml_file}")
                    continue
                
                # Read Image to get real dimensions
                img = cv2.imread(jpg_file)
                if img is None:
                    print(f"Corrupt image: {jpg_file}")
                    continue
                
                h, w, _ = img.shape
                
                # Parse XML
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                except ET.ParseError:
                    print(f"XML Parse Error: {xml_file}")
                    continue

                yolo_lines = []
                
                for obj in root.findall('object'):
                    class_name_raw = obj.find('name').text.strip()
                    
                    # Map Persian name to ID
                    if class_name_raw in LABEL_MAP:
                        class_id = LABEL_MAP[class_name_raw]
                    else:
                        # Log and skip objects with unknown labels
                        print(f"Warning: Unknown label '{class_name_raw}' in {xml_file}. Skipping object.")
                        continue
                    
                    xml_box = obj.find('bndbox')
                    b = (
                        float(xml_box.find('xmin').text),
                        float(xml_box.find('xmax').text),
                        float(xml_box.find('ymin').text),
                        float(xml_box.find('ymax').text)
                    )
                    
                    # Convert to YOLO format
                    bb = convert_box((w, h), b)
                    yolo_lines.append(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

                # Save if we found valid objects
                if yolo_lines:
                    # Create unique filename (e.g., car_img_train_935.jpg)
                    # We ensure no slashes or extra path components are in 'name' itself
                    unique_name = f"{source_dir}_{original_split}_{name}"
                    
                    # --- FIX FOR FILE NOT FOUND ERROR ---
                    
                    # Define target paths
                    target_img_path = os.path.join(OUTPUT_DIR, "images", yolo_split, unique_name + ".jpg")
                    target_lbl_path = os.path.join(OUTPUT_DIR, "labels", yolo_split, unique_name + ".txt")
                    
                    # Copy Image
                    shutil.copy(jpg_file, target_img_path)
                    
                    # Save Label file
                    with open(target_lbl_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_lines))

    print("Dataset conversion completed successfully.")
    create_yaml()

def create_yaml():
    """Generates the data.yaml file required by YOLO."""
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"Created config file at: {yaml_path}")

if __name__ == "__main__":
    process_dataset()