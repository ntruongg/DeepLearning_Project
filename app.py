import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from ultralytics import YOLO
import pandas as pd

# Cấu hình các ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN MÔ HÌNH THỰC TẾ
# ==========================================
# Hãy đảm bảo các file mô hình nằm đúng vị trí theo đường dẫn dưới đây
YOLO_MODEL_PATH = r"D:\traffic\runs\detect\train-9\weights\best.pt"
CNN_MODEL_PATH = "model/cnn_model.h5"
RESNET_MODEL_PATH = "model/resnet_model.h5"       # Sửa lại đường dẫn nếu cần
DENSENET_MODEL_PATH = "model/densenet_model.h5"   # Sửa lại đường dẫn nếu cần
CLASS_NAMES_FILE = "class_names.txt"

# Kích thước ảnh đầu vào chuẩn cho các mô hình phân loại Keras
KERAS_IMAGE_SIZE = (32, 32) 
RESDENSE_IMAGE_SIZE = (64, 64)
# ==========================================
# 2. NẠP MÔ HÌNH & NHÃN LỚP KHI KHỞI ĐỘNG SERVER
# ==========================================
print("\n[HỆ THỐNG] Đang tải các mô hình vào bộ nhớ RAM...")

# Tải mô hình YOLO
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("-> Mô hình YOLO đã được nạp thành công!")
except Exception as e:
    yolo_model = None
    print(f"-> [CẢNH BÁO] Không thể tải YOLO: {e}")

# Tải mô hình CNN tự xây
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("-> Mô hình CNN đã được nạp thành công!")
except Exception as e:
    cnn_model = None
    print(f"-> [CẢNH BÁO] Không thể tải CNN: {e}")

# Tải mô hình ResNet
try:
    resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
    print("-> Mô hình ResNet đã được nạp thành công!")
except Exception as e:
    resnet_model = None
    print(f"-> [CẢNH BÁO] Không thể tải ResNet: {e}")

# Tải mô hình DenseNet
try:
    densenet_model = tf.keras.models.load_model(DENSENET_MODEL_PATH)
    print("-> Mô hình DenseNet đã được nạp thành công!")
except Exception as e:
    densenet_model = None
    print(f"-> [CẢNH BÁO] Không thể tải DenseNet: {e}")

# Nạp danh sách nhãn lớp từ file txt
class_names = []
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"[HỆ THỐNG] Đã nạp thành công {len(class_names)} nhãn lớp.")
else:
    print(f"[CẢNH BÁO] Không tìm thấy file nhãn '{CLASS_NAMES_FILE}'.")


def get_real_chart_data():
    """Hàm bổ trợ: Dùng Pandas đọc file CSV và trả về dữ liệu chuẩn cho Chart.js"""
    # Khởi tạo các mảng mặc định phòng trường hợp thiếu file
    epochs_data = []
    cnn_acc, resnet_acc, densenet_acc = [], [], []
    cnn_loss, resnet_loss, densenet_loss = [], [], []
    yolo_map = []

    # 1. Đọc dữ liệu mô hình CNN tự xây
    cnn_path = 'cnn_history.csv'
    if os.path.exists(cnn_path):
        df_cnn = pd.read_csv(cnn_path)
        cnn_acc = df_cnn['accuracy'].tolist()
        cnn_loss = df_cnn['loss'].tolist()
        # Lấy trục X (Số epoch) dựa trên số lượng hàng dữ liệu của CNN
        epochs_data = list(range(1, len(cnn_acc) + 1))

    # 2. Đọc dữ liệu mô hình ResNet
    resnet_path = 'res_history.csv'
    if os.path.exists(resnet_path):
        df_resnet = pd.read_csv(resnet_path)
        resnet_acc = df_resnet['accuracy'].tolist()
        resnet_loss = df_resnet['loss'].tolist()

    # 3. Đọc dữ liệu mô hình DenseNet
    densenet_path = 'dense_history.csv'
    if os.path.exists(densenet_path):
        df_densenet = pd.read_csv(densenet_path)
        densenet_acc = df_densenet['accuracy'].tolist()
        densenet_loss = df_densenet['loss'].tolist()

    # 4. Đọc dữ liệu mô hình YOLO (đọc từ file results.csv chuẩn của Ultralytics)
    yolo_path = r'D:\traffic\runs\detect\train-9\results.csv'
    if os.path.exists(yolo_path):
        df_yolo = pd.read_csv(yolo_path)
        # Xóa các khoảng trắng thừa ở tên cột nếu có
        df_yolo.columns = [col.strip() for col in df_yolo.columns]
        # Lấy cột mAP@50 chuẩn của YOLO
        if 'metrics/mAP50(B)' in df_yolo.columns:
            yolo_map = df_yolo['metrics/mAP50(B)'].tolist()

    # Trả về một dictionary chứa toàn bộ dữ liệu thực tế
    return {
        'epochs': epochs_data,
        'cnn_acc': cnn_acc, 'resnet_acc': resnet_acc, 'densenet_acc': densenet_acc,
        'cnn_loss': cnn_loss, 'resnet_loss': resnet_loss, 'densenet_loss': densenet_loss,
        'yolo_map': yolo_map
    }
# ==========================================
# 3. ĐIỀU HƯỚNG VÀ XỬ LÝ REQUEST
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    chart_data = get_real_chart_data()
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form['model_choice']
        
        if file and file.filename != '':
            # Lưu tệp tin ảnh gốc người dùng tải lên
            filename = file.filename
            raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(raw_filepath)
            
            # Đọc ảnh gốc bằng OpenCV để chuẩn bị xử lý dữ liệu hình ảnh
            img_bgr = cv2.imread(raw_filepath)
            if img_bgr is None:
                return render_template('index.html', result_text="Lỗi: Hệ thống không thể đọc được file ảnh này.")

            # ------------------------------------------------------
            # KỊCH BẢN 1: Sử dụng mô hình Phát hiện vật thể (YOLO)
            # ------------------------------------------------------
            if selected_model == 'YOLO':
                if yolo_model is None:
                    return render_template('index.html', result_text="Lỗi: Mô hình YOLO chưa được nạp trên hệ thống.")
                
                # Tiến hành dự đoán đóng khung vật thể
                results = yolo_model(raw_filepath, conf=0.25)
                
                # Lấy ảnh kết quả đã được YOLO vẽ sẵn khung bounding box
                annotated_img = results[0].plot()
                
                # Ghi đè bức ảnh đã vẽ khung vào thư mục static/uploads để hiển thị lên web
                cv2.imwrite(raw_filepath, annotated_img)
                
                # Thống kê kết quả tìm kiếm được
                boxes = results[0].boxes
                so_luong = len(boxes)
                if so_luong == 0:
                    result_text = "Không phát hiện thấy biển báo giao thông nào trong bức ảnh này."
                else:
                    chi_tiet = []
                    for box in boxes:
                        c_id = int(box.cls[0])
                        c_name = yolo_model.names[c_id]
                        conf = float(box.conf[0]) * 100
                        chi_tiet.append(f"{c_name} ({conf:.1f}%)")
                    result_text = f"Phát hiện thấy {so_luong} biển báo: {', '.join(chi_tiet)}"
            
            # ------------------------------------------------------
            # KỊCH BẢN 2: Sử dụng các mô hình Phân loại ảnh (CNN, ResNet, DenseNet)
            # ------------------------------------------------------
            else:
                # Lựa chọn đúng bộ não mô hình cần tính toán
                if selected_model == 'CNN':
                    current_model = cnn_model
                elif selected_model == 'ResNet':
                    current_model = resnet_model
                else:
                    current_model = densenet_model

                if current_model is None:
                    return render_template('index.html', result_text=f"Lỗi: Mô hình {selected_model} chưa được nạp thành công.")

                # TIỀN XỬ LÝ ẢNH CHUẨN KÈRAS:
                # 1. Đổi hệ màu từ BGR (OpenCV) sang RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # 2. Thay đổi kích thước về chuẩn huấn luyện (32x32)
                if current_model == cnn_model:
                    img_resized = cv2.resize(img_rgb, KERAS_IMAGE_SIZE)
                else:
                    img_resized = cv2.resize(img_rgb, RESDENSE_IMAGE_SIZE)

                # 3. Chuẩn hóa ma trận điểm ảnh tương ứng theo từng kiến trúc mạng
                if selected_model == 'CNN':
                    img_normalized = img_resized / 255.0
                elif selected_model == 'ResNet':
                    # Nếu lúc train ResNet bạn dùng hàm preprocess_input của ResNet50:
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                    img_float = img_resized.astype(np.float32)
                    img_normalized = preprocess_input(img_float)
                else:
                    # Nếu lúc train DenseNet bạn dùng hàm preprocess_input của DenseNet:
                    from tensorflow.keras.applications.densenet import preprocess_input
                    img_float = img_resized.astype(np.float32)
                    img_normalized = preprocess_input(img_float)

                # 4. Thêm chiều Batch size (1, 32, 32, 3)
                img_input = np.expand_dims(img_normalized, axis=0)

                # Thực hiện dự đoán phân lớp ảnh
                predictions = current_model.predict(img_input)
                class_idx = np.argmax(predictions[0])
                confidence = predictions[0][class_idx] * 100

                # Đối chiếu tìm tên nhãn hiển thị tương thích
                if len(class_names) > class_idx:
                    predicted_class_name = class_names[class_idx]
                else:
                    predicted_class_name = f"Lớp với ID {class_idx}"

                result_text = f"Kết quả phân loại: Biển báo thuộc lớp '{predicted_class_name}' với độ tự tin đạt {confidence:.2f}%"

            # Trả đường dẫn tương đối để giao diện HTML hiển thị
            image_url = f"uploads/{filename}"
            return render_template('index.html', 
                                   image_url=image_url, 
                                   result_text=result_text,
                                   selected_model=selected_model,
                                   chart_data=chart_data)

    return render_template('index.html',chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)