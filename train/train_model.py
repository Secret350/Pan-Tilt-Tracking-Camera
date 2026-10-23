from rich import box
from ultralytics import YOLO
import torch

data_path   = "../dataset/SCUT_HEAD/data.yaml"
fase_name   = "face_detection_scut"
train_device= 0

def main(name,path,device):
    model=YOLO("yolo26s.pt")
    model.train(
        name         = name,
        data         = path,
        epochs       = 100,
        device       = device,
        batch        = 16,
        imgsz        = 640,
        amp          = True,
        patience     = 30,
        mosaic       = 1.0,
        close_mosaic = 10,
        optimizer    = "auto",
        cos_lr       = True,
        box          = 9.83,
        cls          = 0.65,
        dfl          = 0.96
    )

if __name__ == "__main__":
    main(fase_name,data_path,train_device)

