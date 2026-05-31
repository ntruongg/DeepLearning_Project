import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# ==========================================
# KHỞI TẠO CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================

path = "data\\train"              # Đường dẫn tới thư mục dữ liệu train của bạn
class_names_file = "class_names.txt"  # File chứa danh sách nhãn lớp của bạn
batch_size_val = 32             
epochs_val = 20                  # Chạy 20 epochs
imageDimesions = (32, 32, 3)     # SỬ DỤNG 3 KÊNH MÀU RGB (Thay vì 1 kênh ảnh xám)
testRatio = 0.2                  # Tỷ lệ tập kiểm thử (Test set)
validationRatio = 0.2            # Tỷ lệ tập xác thực (Validation set)
csv_logger = CSVLogger('cnn_history.csv', append=False)


class_names = []
if os.path.exists(class_names_file):
    with open(class_names_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"[INFO] Đã nạp thành công {len(class_names)} lớp từ file {class_names_file}.")
    data = pd.DataFrame(class_names, columns=["ClassValue"])
else:
    print(f"[WARNING] Không tìm thấy file '{class_names_file}'. Sẽ tự động gán nhãn theo thư mục.")
    data = pd.DataFrame()



if not os.path.exists(path):
    raise FileNotFoundError(f"Thư mục '{path}' không tồn tại! Vui lòng kiểm tra lại cấu trúc đường dẫn.")

raw_subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

subdirs = sorted(raw_subdirs, key=lambda x: int(x) if x.isdigit() else x)

noOfClasses = len(subdirs)

class_to_idx = {folder: idx for idx, folder in enumerate(subdirs)}
idx_to_class = {idx: folder for idx, folder in enumerate(subdirs)}

# 4. Khi đọc ảnh, lấy nhãn đã được ánh xạ chuẩn:
for folder in subdirs:
    folder_path = os.path.join(path, folder)
    myPicList = os.listdir(folder_path)
    label_id = class_to_idx[folder]

    print(f"Đang đọc dữ liệu lớp '{folder}' -> Mã hóa ID: {label_id} ({len(myPicList)} ảnh)...")

images = []
classNo = []

for idx, folder in enumerate(subdirs):
    folder_path = os.path.join(path, folder)
    myPicList = os.listdir(folder_path)
    
    try:
        label_id = int(folder)
    except ValueError:
        label_id = idx
        
    print(f"Đang nhập lớp {folder} ({len(myPicList)} ảnh)...")
    
    for y in myPicList:
        if y.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, y)
            curImg = cv2.imread(img_path)
            if curImg is not None:
                # Chuyển đổi từ BGR (OpenCV mặc định) sang RGB
                curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
                images.append(curImg)
                classNo.append(label_id)

images = np.array(images)
classNo = np.array(classNo)

max_label = classNo.max() if len(classNo) > 0 else 0
noOfClasses = max(len(class_names), max_label + 1)
print(f"\n--> Tổng cộng đã nạp: {len(images)} ảnh.")
print(f"--> Xác định số lượng lớp phân loại (Classes): {noOfClasses}")


X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio, random_state=42)

print("\n--- Kích thước dữ liệu sau khi chia ---")
print(f"Train:      {X_train.shape}, {y_train.shape}")
print(f"Validation: {X_validation.shape}, {y_validation.shape}")
print(f"Test:       {X_test.shape}, {y_test.shape}")


def preprocessing(img):
    return img / 255.0

# Áp dụng chuẩn hóa cho ảnh màu (giữ nguyên cấu trúc 3D của ảnh RGB)
X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])



dataGen = ImageDataGenerator(
    width_shift_range=0.1,   
    height_shift_range=0.1,
    zoom_range=0.15,  
    shear_range=0.1,  
    rotation_range=10
)  
dataGen.fit(X_train)


y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


def myModel():
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(imageDimesions[0], imageDimesions[1], 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) # Giảm dropout ở lớp đầu để tránh mất mát thông tin
    
    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Fully Connected (Phân loại sâu)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(noOfClasses, activation='softmax')) 
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = myModel()
    print(model.summary())
    
    # CALLBACKS: Tự động giảm tốc độ học nếu validation loss ngừng cải thiện
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    
    # Huấn luyện mô hình
    history = model.fit(
        dataGen.flow(X_train, y_train, batch_size=batch_size_val),
        steps_per_epoch=max(1, len(X_train) // batch_size_val),
        epochs=epochs_val,
        validation_data=(X_validation, y_validation),
        callbacks=[lr_reducer, early_stopper, csv_logger],
        shuffle=True
    )
    
    # Vẽ biểu đồ kết quả Loss & Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='training')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()
    
    # Đánh giá trên tập Test
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ---')
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])
    
    # Lưu mô hình đã huấn luyện
    model_name = "acnnmodel.h5"
    model.save(model_name)
    print(f"[SUCCESS] Đã lưu mô hình thành công vào file '{model_name}'!")