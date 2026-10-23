from sympy import true
from ultralytics import YOLO

model = YOLO("../models/HEAD_NEW_MODEL/best.pt")
model.export(format="engine", device=0, quantize="fp16")