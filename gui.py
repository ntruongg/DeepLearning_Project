import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.densenet import preprocess_input

# ==========================================
# CẤU HÌNH & LOAD MÔ HÌNH
# ==========================================
MODEL_PATH = "densenet_model.h5"
CLASS_NAMES_FILE = "class_names.txt"
IMAGE_SIZE = (64, 64)  # Đúng chuẩn kích thước đầu vào của mô hình

# Kiểm tra sự tồn tại của mô hình trước khi khởi chạy
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Không tìm thấy file mô hình '{MODEL_PATH}'.")
    print("Vui lòng chạy file train.py để huấn luyện và tạo ra mô hình trước.")
    # Tạo cửa sổ thông báo lỗi nếu chạy trực tiếp bằng giao diện
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Lỗi hệ thống", f"Không tìm thấy file mô hình '{MODEL_PATH}'!\nHãy huấn luyện mô hình trước khi chạy GUI.")
    exit()

print("Đang tải mô hình...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Mô hình đã được nạp thành công!")

# Tải danh sách nhãn lớp từ class_names.txt
class_names = []
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Đã nạp {len(class_names)} nhãn lớp.")
else:
    print(f"[WARNING] Không tìm thấy '{CLASS_NAMES_FILE}'. Sẽ hiển thị chỉ số lớp (Class ID) thay thế.")

# ==========================================
# THIẾT KẾ GIAO DIỆN ĐỒ HỌA (GUI)
# ==========================================
class TrafficSignGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Phần mềm Nhận dạng Biển báo Giao thông")
        self.window.geometry("500x600")
        self.window.configure(bg="#f4f6f9")
        self.window.resizable(False, False)

        # Tiêu đề chính
        self.title_label = tk.Label(
            window, 
            text="NHẬN DẠNG BIỂN BÁO GIAO THÔNG", 
            font=("Arial", 16, "bold"), 
            fg="#1a365d", 
            bg="#f4f6f9"
        )
        self.title_label.pack(pady=20)

        # Khung hiển thị ảnh (Canvas/Label)
        self.image_label = tk.Label(
            window, 
            text="Chưa chọn hình ảnh nào", 
            font=("Arial", 11),
            bg="#e2e8f0", 
            width=30, 
            height=12,
            relief="solid",
            bd=1
        )
        self.image_label.pack(pady=10)

        # Nút nhấn chọn ảnh
        self.btn_select = tk.Button(
            window, 
            text="Chọn hình ảnh", 
            command=self.select_image,
            font=("Arial", 12, "bold"),
            fg="white", 
            bg="#3182ce", 
            activebackground="#2b6cb0",
            activeforeground="white",
            padx=20, 
            pady=8,
            cursor="hand2",
            relief="flat"
        )
        self.btn_select.pack(pady=20)

        # Khung hiển thị kết quả dự đoán
        self.result_frame = tk.LabelFrame(
            window, 
            text=" Kết quả phân loại ", 
            font=("Arial", 11, "bold"),
            bg="#f4f6f9", 
            fg="#4a5568",
            padx=15, 
            pady=15
        )
        self.result_frame.pack(padx=20, pady=10, fill="x")

        # Nhãn kết quả lớp
        self.class_label = tk.Label(
            self.result_frame, 
            text="Biển báo: --", 
            font=("Arial", 13, "bold"), 
            fg="#2d3748", 
            bg="#f4f6f9",
            anchor="w"
        )
        self.class_label.pack(fill="x", pady=2)

        # Nhãn độ tin cậy
        self.prob_label = tk.Label(
            self.result_frame, 
            text="Độ tự tin: --", 
            font=("Arial", 12), 
            fg="#718096", 
            bg="#f4f6f9",
            anchor="w"
        )
        self.prob_label.pack(fill="x", pady=2)

    def select_image(self):
        # Mở hộp thoại chọn file ảnh
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return  # Người dùng hủy chọn ảnh
        
        try:
            # 1. Đọc và xử lý ảnh để dự đoán
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Không thể đọc được tệp tin ảnh này.")
            
            # Chuyển đổi màu từ BGR sang RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Tiến hành resize về (32, 32) đúng chuẩn huấn luyện
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            
            
            #ResNet
            img_float = img_resized.astype(np.float32)
            img_preprocessed = preprocess_input(img_float)
            
            img_input = np.expand_dims(img_preprocessed, axis=0)
            
            
            #CNN            
            """"
            # Chuẩn hóa về [0, 1]
            img_normalized = img_resized / 255.0
            
            # Thêm chiều batch_size: (32, 32, 3) -> (1, 32, 32, 3)
            img_input = np.expand_dims(img_normalized, axis=0)
            """
            
            
            # 2. Thực hiện dự đoán bằng mô hình
            predictions = model.predict(img_input)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx] * 100

            # Lấy tên hiển thị tương ứng
            if len(class_names) > class_idx:
                predicted_class_name = class_names[class_idx]
            else:
                predicted_class_name = f"Lớp {class_idx}"

            # 3. Cập nhật kết quả lên giao diện
            self.class_label.config(text=f"Biển báo: {predicted_class_name}", fg="#e53e3e")
            self.prob_label.config(text=f"Độ tự tin: {confidence:.2f}%")

            # 4. Hiển thị ảnh vừa chọn lên giao diện (đã resize hiển thị)
            img_pil = Image.fromarray(img_rgb)
            # Resize để vừa vặn khung hiển thị trên GUI (ví dụ: tối đa 250x250)
            img_pil.thumbnail((250, 250))
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk  # Giữ tham chiếu để tránh bị bộ thu gom rác dọn mất

        except Exception as e:
            messagebox.showerror("Lỗi xử lý", f"Đã xảy ra lỗi khi xử lý ảnh:\n{str(e)}")

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()
