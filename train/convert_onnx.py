from torch.nn.qat import dynamic
from ultralytics import YOLO

def main():
    weights = "./runs/detect/face_detection_1/weights/best.pt"
    model = YOLO(weights)

    onnx_path = model.export(
        format      = "onnx",
        imgsz       = 640,
        dynamic     = False,
        simplify    = True,
        half        = True,
        device      = "cpu",
        opset       = 12
    )

    print("Export Completed!")

if __name__ == "__main__":
    main()