from ultralytics import YOLO

def main():
    model = YOLO("runs/anpr_yolov8/weights/best.pt")

    model.predict(
        source="/Users/namanmittal/ANPR_System/dataset/images/train/WB26.jpg",
        conf=0.4,
        save=True
    )

if __name__ == "__main__":
    main()
