import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# =====================================================================
# KHỞI TẠO CẤU HÌNH ĐƯỜNG DẪN & SIÊU THAM SỐ (ĐỒNG BỘ VỚI RESNET)
# =====================================================================
path = "data/train"                    # Đường dẫn chứa thư mục ảnh gốc (data/train/<class_id>)
class_names_file = "class_names.txt"        # File chứa ánh xạ nhãn để đồng bộ với phần mềm GUI
batch_size_val = 64                    # Kích thước batch tối ưu 
epochs_val = 20                        # Số lượng epoch tối đa (Early Stopping sẽ tự ngắt khi tối ưu)
imageDimesions = (64, 64, 3)           # Độ phân giải 64x64 giữ chi tiết tốt cho biển báo
testRatio = 0.2                        # Tỷ lệ chia tập kiểm thử độc lập (Test)
validationRatio = 0.2                  # Tỷ lệ chia tập xác thực trong lúc train (Validation)

# =====================================================================
# BƯỚC 1: SẮP XẾP THEO SỐ TỰ NHIÊN (ĐỒNG BỘ HOÀN TOÀN TRÁNH BUG LỆCH NHÃN)
# =====================================================================
if not os.path.exists(path):
    raise FileNotFoundError(f"Thư mục dữ liệu '{path}' không tồn tại! Vui lòng kiểm tra lại.")

# Lấy danh sách thư mục con thực tế
raw_subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# Ép kiểu int khi sort để đảm bảo thứ tự số tự nhiên: 0, 1, 2, ..., 10, 11, ..., 38
subdirs = sorted(raw_subdirs, key=lambda x: int(x) if x.isdigit() else x)
noOfClasses = len(subdirs)
print(f"[INFO] Phát hiện hệ thống có: {noOfClasses} lớp thực tế.")

# Ánh xạ tên thư mục thành nhãn số liên tục từ 0 đến (N-1)
class_to_idx = {folder: idx for idx, folder in enumerate(subdirs)}
idx_to_class = {idx: folder for idx, folder in enumerate(subdirs)}

# Ghi danh sách nhãn lớp theo thứ tự chuẩn học thuật vào file để GUI đọc
with open(class_names_file, 'w', encoding='utf-8') as f:
    for idx in range(noOfClasses):
        f.write(f"{idx_to_class[idx]}\n")
print(f"[INFO] Đã lưu danh sách lớp đồng bộ số tự nhiên vào file '{class_names_file}'.")

# =====================================================================
# BƯỚC 2: ĐỌC DỮ LIỆU VÀ CHUYỂN ĐỔI HỆ MÀU (BGR -> RGB)
# =====================================================================
images = []
classNo = []

for folder in subdirs:
    folder_path = os.path.join(path, folder)
    myPicList = os.listdir(folder_path)
    label_id = class_to_idx[folder]    # Lấy nhãn chuẩn liên tục
    
    print(f"Đang đọc dữ liệu lớp '{folder}' -> Mã hóa ID chuẩn: {label_id} ({len(myPicList)} ảnh)...")
    
    for y in myPicList:
        if y.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, y)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                # Chuyển đổi màu hệ màu từ BGR sang RGB chuẩn DenseNet/Keras
                curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
                images.append(curImg)
                classNo.append(label_id)

images = np.array(images)
classNo = np.array(classNo)

print(f"\n[INFO] Tổng số lượng ảnh nạp thành công: {len(images)} ảnh.")

# =====================================================================
# BƯỚC 3: PHÂN TẦNG DỮ LIỆU ĐỒNG ĐỀU (STRATIFIED SPLIT)
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    images, classNo, 
    test_size=testRatio, 
    stratify=classNo, 
    random_state=42
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, 
    test_size=validationRatio, 
    stratify=y_train, 
    random_state=42
)

print("\n--- Kích thước tập dữ liệu sau khi Phân Tầng (Stratified) ---")
print(f"Tập Train:      {X_train.shape}, {y_train.shape}")
print(f"Tập Validation: {X_validation.shape}, {y_validation.shape}")
print(f"Tập Test:       {X_test.shape}, {y_test.shape}")

# =====================================================================
# BƯỚC 4: TIỀN XỬ LÝ ĐẶC THÙ CHO MẠNG DENSENET (PREPROCESSING)
# =====================================================================
def preprocessing(img):
    # Sử dụng hàm preprocess_input gốc của DenseNet121 từ Keras
    # Thực hiện chia tỷ lệ pixel và chuẩn hóa phân phối theo ImageNet mẫu
    return preprocess_input(img.astype(np.float32))

X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])

# =====================================================================
# BƯỚC 5: TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)
# =====================================================================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,   
    height_shift_range=0.1,
    zoom_range=0.15,  
    shear_range=0.1,  
    rotation_range=10,
    brightness_range=[0.8, 1.2]
)  
dataGen.fit(X_train)

# Chuyển đổi nhãn sang dạng mã hóa One-Hot
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# =====================================================================
# BƯỚC 6: XÂY DỰNG MÔ HÌNH DENSENET121 + DENSE CLASSIFIER CHUYÊN SÂU
# =====================================================================
def build_densenet_model(input_shape, num_classes):
    # Nạp backbone DenseNet121 pretrained trên tập ImageNet khổng lồ
    base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Đóng băng các layers đầu (giữ nguyên tri thức cơ bản: hình khối, góc cạnh, màu sắc)
    # Mở băng 30 layers cuối cùng để thực hiện tinh chỉnh (Fine-tuning) theo đặc trưng biển báo Việt Nam
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)  # Nén vector đặc trưng, chống overfitting tốt hơn Flatten
    
    # -------------------------------------------------------------
    # KHỐI DENSE HEAD (ĐỒNG BỘ THIẾT KẾ ĐỐI CHỨNG VỚI FILE RESNET)
    # -------------------------------------------------------------
    x = Dense(512, activation='relu')(x)      # Tầng Dense thứ nhất (512 nơ-ron)
    x = BatchNormalization()(x)                # Chuẩn hóa tăng tốc hội tụ và chống bão hòa gradient
    x = Dropout(0.3)(x)                        # Dropout 30% chống học vẹt (Overfitting)
    
    x = Dense(256, activation='relu')(x)      # Tầng Dense thứ hai (256 nơ-ron)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)  # Tầng đầu ra phân loại cho 39 lớp
    
    model = Model(inputs, outputs)
    
    # Sử dụng Learning Rate nhỏ (1e-4) khi Fine-tuning để bảo toàn trọng số tối ưu sẵn có của DenseNet
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# =====================================================================
# BƯỚC 7: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH DENSENET
# =====================================================================
if __name__ == "__main__":
    # Khởi tạo mô hình DenseNet121 + Dense Head
    model = build_densenet_model(imageDimesions, noOfClasses)
    print(model.summary())
    
    # Callbacks tự động điều khiển tiến trình học
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    
    # Thực thi huấn luyện
    history = model.fit(
        dataGen.flow(X_train, y_train, batch_size=batch_size_val),
        steps_per_epoch=max(1, len(X_train) // batch_size_val),
        epochs=epochs_val,
        validation_data=(X_validation, y_validation),
        callbacks=[lr_reducer, early_stopper],
        shuffle=True
    )
    
    # Vẽ đồ thị kết quả biểu diễn Loss và Accuracy để làm tài liệu so sánh trong đề tài
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Mất mát tập huấn luyện (training)')
    plt.plot(history.history['val_loss'], label='Mất mát tập xác thực (validation)')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Độ chính xác tập huấn luyện (training)')
    plt.plot(history.history['val_accuracy'], label='Độ chính xác tập xác thực (validation)')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()
    
    # Đánh giá độc lập trên tập dữ liệu Test chưa từng gặp
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ---')
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])
    
    # Lưu mô hình hoàn thiện ra file chuyên biệt
    model_name = "densenet_model.h5"
    model.save(model_name)
    print(f"[SUCCESS] Lưu mô hình DenseNet thành công vào file '{model_name}'!")

