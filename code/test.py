import cv2
import os
from ultralytics import YOLO

# ==========================================
# 1. تنظیمات و متغیرهای پایه
# ==========================================
MODEL_PATH = "best4.pt" 
INPUT_PATH = "test_images/"  # مسیر فولدر یا یک عکس خاص
OUTPUT_FOLDER = "output_results"

CLASS_NAMES = [
    "plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "alef", "be", "pe", "te", "se", "jim", "dal", "ze", "sin", 
    "shin", "sad", "ta", "za", "ein", "fe", "ghaf", "kaf", 
    "gaf", "lam", "mim", "noon", "vav", "he", "ye", "zhe", 
    "disabled", "protocol", "S", "D"
]

# ایجاد پوشه خروجی اگر وجود نداشته باشد
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ==========================================
# 2. توابع کمکی
# ==========================================

def is_inside(box_inner, box_outer):
    """بررسی اینکه آیا مرکز کاراکتر داخل باکس پلاک هست یا خیر"""
    c_x = (box_inner[0] + box_inner[2]) / 2
    c_y = (box_inner[1] + box_inner[3]) / 2
    return (box_outer[0] < c_x < box_outer[2]) and (box_outer[1] < c_y < box_outer[3])

def format_plate_text(chars_list):
    """
    فرمت‌دهی طبق درخواست کاربر:
    دو رقم شماره | سه رقم شماره یک حرف دو رقم شماره
    """
    if len(chars_list) < 8:
        return " ".join(chars_list) # اگر تعداد کاراکترها کامل نبود
    
    # فرض بر این است که لیست بر اساس موقعیت X مرتب شده است
    # پلاک ایران: [0,1] اعداد سمت چپ | [2] حروف | [3,4,5] سه رقم وسط | [6,7] کد شهر (سمت راست)
    # توجه: بسته به آموزش مدل، جایگاه حرف ممکن است ایندکس 2 یا جای دیگر باشد.
    
    try:
        part1 = "".join(chars_list[0:2])    # دو رقم اول
        part2 = "".join(chars_list[3:6])    # سه رقم وسط
        letter = chars_list[2]              # حرف
        part3 = "".join(chars_list[6:8])    # دو رقم آخر (کد شهر)
        
        return f"{part1} | {part2} {letter} {part3}"
    except:
        return " ".join(chars_list)

# ==========================================
# 3. پردازش تصاویر
# ==========================================

def process_images():
    # بارگذاری مدل
    model = YOLO(MODEL_PATH)
    
    # تشخیص اینکه ورودی فایل است یا فولدر
    if os.path.isfile(INPUT_PATH):
        image_files = [INPUT_PATH]
    else:
        image_files = [os.path.join(INPUT_PATH, f) for f in os.listdir(INPUT_PATH) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("هیچ عکسی در مسیر مشخص شده پیدا نشد!")
        return

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None: continue

        # اجرای اینفرنس (استفاده از CPU به صورت پیش‌فرض)
        results = model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'class': CLASS_NAMES[int(class_id)],
                'score': score
            })

        plates = [d for d in detections if d['class'] == 'plate']
        characters = [d for d in detections if d['class'] != 'plate']

        for plate in plates:
            px1, py1, px2, py2 = plate['box']
            
            # استخراج کاراکترهای مربوط به این پلاک
            plate_chars = [c for c in characters if is_inside(c['box'], plate['box'])]
            
            # مرتب‌سازی از چپ به راست
            plate_chars.sort(key=lambda x: x['box'][0])
            
            # تبدیل به متن
            text_list = [c['class'] for c in plate_chars]
            plate_text = format_plate_text(text_list)
            
            # --- رسم روی تصویر ---
            # مستطیل پلاک
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 3)
            
            # نمایش متن بالای پلاک
            label = f"Plate: {plate_text}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (px1, py1 - h - 15), (px1 + w, py1), (0, 165, 255), -1) # پس‌زمینه نارنجی
            cv2.putText(frame, label, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ذخیره خروجی
        save_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        cv2.imwrite(save_path, frame)
        print(f"Processed: {os.path.basename(img_path)} -> Saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_images()