from ultralytics import YOLO
import config

def train():
    model = YOLO("yolov8n.pt")
    # Đường dẫn đến data.yaml từ dataset vừa tải
    data_yaml = "path/to/your/dataset/data.yaml"
    model.train(data=data_yaml, epochs=50, imgsz=640, device='cpu')
    model.save(str(config.PLATE_MODEL_PATH))
    print("Training completed. Model saved.")

if __name__ == "__main__":
    train()