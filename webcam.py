import cv2
from ultralytics import YOLO

def detect_from_phone_screen():
    # ==========================================
    # 1. NẠP MÔ HÌNH YOLOV11
    # ==========================================
    model_path = r"D:\traffic\runs\detect\train-9\weights\best.pt"
    print("\n[HỆ THỐNG] Đang tải mô hình YOLO lên card đồ họa...")
    
    try:
        model = YOLO(model_path)
        print("[HỆ THỐNG] Tải mô hình thành công!\n")
    except Exception as e:
        print(f"Lỗi tải mô hình: {e}")
        return

    # ==========================================
    # 2. KHỞI TẠO WEBCAM
    # ==========================================
    cap = cv2.VideoCapture(0) # '0' là camera mặc định của laptop
    
    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Webcam. Vui lòng kiểm tra lại quyền truy cập camera!")
        return

    print("=> Đang mở Webcam... Hãy giơ điện thoại của bạn lên.")
    print("=> Nhấn phím 'q' trên bàn phím để THOÁT.")

    # ==========================================
    # 3. VÒNG LẶP XỬ LÝ VIDEO THEO THỜI GIAN THỰC
    # ==========================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Mất tín hiệu từ Webcam.")
            break

        # ĐẢM BẢO NHẬN DIỆN QUA MÀN HÌNH ĐIỆN THOẠI: 
        # Hạ conf=0.20 để bắt được các đặc trưng bị mờ hoặc lóa sáng do màn hình
        # iou=0.5 giúp lọc bớt các khung hình chồng chéo
        results = model(frame, conf=0.20, iou=0.5, stream=True)

        # Lấy khung hình đã được AI vẽ bounding box
        annotated_frame = frame 
        for r in results:
            annotated_frame = r.plot()

        # ==========================================
        # 4. GIAO DIỆN HỖ TRỢ NGƯỜI DÙNG (HUD)
        # ==========================================
        # Hiển thị cửa sổ
        cv2.imshow("Test Nhan Dien Bien Bao Tren Dien Thoai", annotated_frame)

        # Bấm phím 'q' để thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[HỆ THỐNG] Đang đóng cửa sổ Webcam...")
            break

    # Giải phóng camera và dọn dẹp bộ nhớ RAM
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_from_phone_screen()