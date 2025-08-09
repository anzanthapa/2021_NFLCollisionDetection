# NFL Safety and Helmet Collision Detection
## Overview
This project investigates automated helmet detection in National Football League (NFL) game footage to support injury prevention efforts. Using **YOLOv5**, we designed and trained an object detection model capable of identifying all 22 player helmets in live or recorded video frames. The primary goal was to improve upon the NFL’s baseline detection system to facilitate large-scale, accurate monitoring of helmet-to-helmet, helmet-to-shoulder, and helmet-to-ground collisions.
## Background 
The NFL has partnered with Amazon Web Services (AWS) to advance sports injury surveillance through digital athlete programs. Current manual review processes are inefficient, analyzing only a fraction of available footage. Automated helmet detection provides the foundation for scalable tracking and impact classification, enabling improved safety protocols and strategic play analysis.
## Objectives
- Detect helmets in NFL video frames using YOLOv5.
- Compare detection performance against baseline bounding box annotations.
- Optimize YOLOv5 hyperparameters to improve detection confidence and F1-score.
## Dataset
The dataset, provided by the competition organizers on Kaggle (https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment), includes:
- **Images:** 9,947 labeled endzone and sideline view frames.
- **Annotations:** Bounding boxes in `image_labels.csv`.
- **Baseline Predictions:** `train/test_baseline_helmets.csv` from the organizer’s initial model.
- **Videos:** Training and test footage for model evaluation.
## Methodology
1. **Baseline Analysis** – Evaluated NFL-provided helmet detection boxes on sample videos.
2. **YOLOv5 Implementation**  
   - Prepared dataset in YOLO format.  
   - Trained model on Kaggle using GPU acceleration.  
   - Tuned hyperparameters (batch size, epochs).  
3. **Evaluation** – Measured confidence scores and F1-scores across helmet classes, prioritizing the primary *Helmet* class.
## Key Findings
- **Epoch Increase:** Raising epochs from 1 to 10 improved confidence scores and F1 performance, with the largest gains between 1 and 5 epochs.
- **Performance:** YOLOv5 achieved comparable or better accuracy than the baseline model within the constraints of computational resources.
- **Scalability:** Further improvements are expected with more epochs, larger datasets, and alternative pretrained weights.
## Limitations
- Focused solely on helmet detection without tracking player interactions.
- Training data drawn from selected video frames; varied angles and occlusions in real gameplay may reduce accuracy.
- Computational constraints limited hyperparameter exploration.
## Future Work
- Experiment with additional YOLOv5 configurations and pretrained weights.
- Integrate multi-object tracking for continuous helmet identification across frames.
- Extend detection to collision type classification for direct injury risk assessment.
## How to Run
This repository contains a Jupyter Notebook (`nfl_helmet_detection.ipynb`) that demonstrates the complete workflow — from dataset preparation to model training, evaluation, and testing.
**To run the notebook:**
1. Open the notebook in [Kaggle](https://www.kaggle.com/) or a local Jupyter environment.
2. Ensure Python 3.8+ is installed, along with the required dependencies:
   - `torch`
   - `opencv-python`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `scikit-learn`
   - `tqdm`
   - `wandb`
   - `yolov5` (from [Ultralytics](https://github.com/ultralytics/yolov5))
3. If running locally, download the NFL dataset from the Kaggle competition and place it in the designated dataset folder.
4. Follow the cells in the notebook sequentially:
   - **Dataset Preparation**
   - **YOLOv5 Training**
   - **Evaluation**
   - **Video Testing**
5. The final section of the notebook demonstrates helmet detection on sample test videos.
## References
Key resources include:
- [Ultralytics YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
## Project Collaborators
- **Bhoj Raj Thapa**
- **Taylor Thomas**    
- **Samaneh Rabienia**  
- **Laziz Muradov**  
- **Mahsen Al-Ani**