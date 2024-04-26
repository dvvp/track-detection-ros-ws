from ultralytics import YOLO

# Load a model
model = YOLO("models/best320x192v2.blob")
# model = YOLO("models/best320x320.pt")
# model = YOLO("path/to/custom_yolo_model.pt")

# Export the model  
model.export(format="onnx", imgsz=[192, 320])
