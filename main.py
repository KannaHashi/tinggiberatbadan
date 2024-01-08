import torch

# Model
model = torch.hub.load("yolov5", "yolov5n", source="local")  # or yolov5n - yolov5x6, custom

# Images
# img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
img = "Machine Learning/dataset_img/1-3.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()
results.show()# or .show(), .save(), .crop(), .pandas(), etc.