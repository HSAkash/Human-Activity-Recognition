# Human Activity Recognition

This project focuses on detecting human activities, primarily using video-based action datasets. The system also works with sequential image datasets. To process only relevant data, remove the initial stage from the `dvc.yml` or `main.py` file as needed.

We extracted two types of features in this system:
1. **Keypoints from Human Pose** using [YOLO11-Pose](https://docs.ultralytics.com/models/yolo11/).
2. **Feature Vectors** from ImageNet.

To improve accuracy, we first remove irrelevant backgrounds or blur them. For segmentation, we use [SAM](https://segment-anything.com/), while `cv2` is used for blurring. After preprocessing, we extract features from the images using ImageNet. Finally, the two types of features are combined during training, which leads to improved results.

This approach achieved **96.07% accuracy** on the [Two Person Interaction Kinect Dataset](https://www.kaggle.com/datasets/dasmehdixtr/two-person-interaction-kinect-dataset) and **96.92% accuracy** on the [Human Activity Recognition (HAR - Video Dataset)](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset).

<div align="center">
  <img src="/assets/HAR_main_diagram.JPG" alt="Step 1: Capture Image" style="margin: 0 10px;">
</div>

## Setup process
* Create environment
```
python -m venv env
```
* Install all requirements
```
pip install -r requirements.txt
```
* run the template.py
```
python template.py
```
* Copy video dataset to data/videoDataset
<pre>
│  
├─config
├─data
|  ├─videoDataset
|      ├─class_01
|      |   ├─ 01.mp4
|      |
|      ├─class_02
|      |    ├─01.mp4
|      |    ├─02.mp4
|  
├─src
|   
</pre>
* Or if image them `Copy images to data/imageDataset
<pre>
│  
├─config
├─data
|  ├─imageDataset
|      ├─class_01
|      |   ├─forldername
|      |       ├─01.jpg
|      |       ├─02.jpg
|      |   ├─forldername
|      |       ├─01.jpg
|      |       ├─02.jpg
|      ├─class_02
|      |   ├─forldername
|      |       ├─01.jpg
|      |       ├─02.jpg
|      |   ├─forldername
|      |       ├─01.jpg
|      |       ├─02.jpg
├─src
|   
</pre>
* Run command `python`
  ```
  python main.py
  ```
* Or dvc
```
dvc repro
```

## Performance

### * 1st dataset [Two Person Interaction Kinect Dataset](https://www.kaggle.com/datasets/dasmehdixtr/two-person-interaction-kinect-dataset):
<div align="center">
  <img src="/assets/dataset_01/05.png" alt="Step 1: Capture Image" style="margin: 0 10px;">
</div>
<div align="center">
  <img src="/assets/dataset_01/06.png" alt="Step 1: Capture Image" width="50%" style="margin: 0 10px;">
</div>

##### Classification results
| Task              | Precision | Recall | F1 Score | Accuracy |
|-------------------|-----------|--------|----------|----------|
| Close Up          | 0.9836    | 0.9836 | 0.9836   | 0.9836   |
| Get Away          | 0.9918    | 0.9758 | 0.9837   | 0.9758   |
| Kick              | 0.9919    | 0.9609 | 0.9762   | 0.9609   |
| Push              | 0.9907    | 0.9907 | 0.9907   | 0.9907   |
| Shake Hands       | 0.9615    | 0.9804 | 0.9709   | 0.9804   |
| Hug               | 0.9787    | 0.9020 | 0.9388   | 0.9020   |
| Give a Notebook   | 0.9174    | 0.9737 | 0.9447   | 0.9737   |
| Punch             | 0.8977    | 0.9186 | 0.9080   | 0.9186   |
| **Average**       | **0.9642**| **0.9607** | **0.9621** | **0.9607** |





### * 2nd Dataset [Human Activity Recognition (HAR - Video Dataset)](https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset)
<div align="center">
  <img src="/assets/dataset_02/loss.png" alt="Step 1: Capture Image"  width="33%" style="margin: 0 10px;">
  <img src="/assets/dataset_02/accuracy.png" alt="Step 1: Capture Image" width="33%" style="margin: 0 10px;">
</div>
<div align="center">
  <img src="/assets/dataset_02/conf.png" alt="Step 1: Capture Image" width="50%" style="margin: 0 10px;">
</div>

##### Classification results
| Task                        | Precision | Recall | F1 Score | Accuracy |
|-----------------------------|-----------|--------|----------|----------|
| Clapping                    | 1.0000    | 1.0000 | 1.0000   | 1.0000   |
| Meet and Split              | 1.0000    | 1.0000 | 1.0000   | 1.0000   |
| Sitting                     | 1.0000    | 1.0000 | 1.0000   | 1.0000   |
| Standing Still              | 1.0000    | 1.0000 | 1.0000   | 1.0000   |
| Walking                     | 0.9167    | 0.9429 | 0.9296   | 0.9429   |
| Walking While Reading Book  | 1.0000    | 0.9444 | 0.9714   | 0.9444   |
| Walking While Using Phone   | 0.8667    | 0.8966 | 0.8814   | 0.8966   |
| **Average**                 | **0.9690**| **0.9691** | **0.9689** | **0.9692** |

