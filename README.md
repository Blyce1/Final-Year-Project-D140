## Project Structure

| Folder | Description |
|---|---|
| `condition_analysis.py` | Benchmarking tool |
| `hula_ui.py` | Flight control UI |
| `yolo_obb_training.ipynb` | OBB model training |
| `yolo_pose_training.ipynb` | Pose model training |

## Requirements

- Python 3.13
- Ultralytics YOLOv8
- OpenCV 4.x
- pyhula SDK

Install dependencies:
```bash
pip install ultralytics opencv-python
```

## How to Run

**Flight Control UI**
```bash
python hula_ui.py
```
Ensure the Hula drone is connected over Wi-Fi before launching.
Model weights (such as best.pt) must be placed at the paths defined in `MODEL_PATHS` at the top of the file.

**Condition Analysis Tool**
```bash
python condition_analysis.py --source path/to/video.avi
```
Optional flags:
- `--skip N` — process every Nth frame
- `--frames N` — cap total frames processed
- `--conf 0.25` — confidence threshold
- `--imgsz 1280` — inference image size

**Model Training**

Open `yolo_obb_training.ipynb` or `yolo_pose_training.ipynb` in Jupyter and run cells sequentially. Update the `data` path in the training cell to point to your local dataset YAML file.
