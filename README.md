Here's your modified Markdown content:

```markdown
# IHA-YOLO

Official PyTorch implementation of **IHA-YOLO**: Inter-Head Attention for Real-time Cell Detection and Counting.

## Installation

To set up the environment and install the required packages, run the following commands:

```bash
conda create -n iha_yolo python=3.10
conda activate iha_yolo
pip install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd ultralytics
pip install -e .
```

## Training

To train the IHA-YOLO model, use the following code snippet:

```python
from ultralytics import YOLO

# Load the model configuration and weights
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")

# Start training
results = model.train(data="data.yaml", epochs=200)
```

## Testing

To evaluate the trained model, you can use the following code:

```python
from ultralytics import YOLO

# Load the model configuration and weights
model = YOLO("IHA-YOLO/ultralytics/cfg/models/iha-yolo/iha-yolom.yaml").load("yolov10m.pt")

# Validate the model
metrics = model.val()

# Print evaluation metrics
print(f"Mean Average Precision @ .5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")
```

## Performance Comparison

### Detection Performance Comparison Table

![Download Performance Chart](https://github.com/user-attachments/files/16749227/chart.11.1.1.pdf)

![image](https://github.com/user-attachments/assets/03f622f4-8288-42a1-9ec2-54640809d6e3)

*Detection performance comparison of IHA-YOLO vs. various models on Mouse Embryos and BCCD datasets. The models listed in the last two sections were evaluated by us for the first time on these datasets. The highest values are highlighted in red, and the second highest in blue.*

![Qualitative Results](https://github.com/user-attachments/assets/890890f2-4f69-452e-9e0d-8e117bd13902)

## Acknowledgements

This repository is a modified version of the YOLOv10 code provided by [Ultralytics](https://github.com/ultralytics/ultralytics).
```
