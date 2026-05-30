from ultralytics import YOLO

# MỌI lệnh thực thi phải được lùi lề (indent) vào trong khối if này
if __name__ == '__main__':
    # Khởi tạo mô hình
    model = YOLO('yolo11m.pt') 
    
    # Bắt đầu huấn luyện
    model.train(
        data="D:/traffic/datat/data.yaml", 
        imgsz=640, 
        batch=4, 
        epochs=80, 
        workers=2,   # Bạn có thể tăng lên 2 hoặc 4 để load ảnh nhanh hơn
        device=0
    )