from ultralytics import YOLO

def main():
    model = YOLO("runs/anpr_yolov8/weights/best.pt")

    metrics = model.val(
        data="dataset/data.yaml",
        imgsz=640,
        device="cpu"
    )

    print(metrics)

if __name__ == "__main__":
    main()
