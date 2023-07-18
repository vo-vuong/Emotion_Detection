# Detect human emotions using Mediapipe, CNN

## Introduction

This project is used to detect human emotions. There are 8 different types of emotions: `surprise`, `anger`, `disgust`, `fear`, `sad`, `contempt`, `neutral`, `happy`.

These are 2 major components:

1. Face Detection: using Mediapipe library and draw bounding box.
2. Emotion Recognition: using a CNN model built from scratch. Responsible for handling emotion recognition related functionalities from an image.

There are 2 ways to use the model: webcam(default), image.

<p align="center">
  <img src="https://raw.githubusercontent.com/vo-vuong/assets/main/emotion_detection/output_happy_image.png" width=500 height=300px><br/>
    <img src="https://raw.githubusercontent.com/vo-vuong/assets/main/emotion_detection/output_sad_image.png" width=500 height=300px><br/>
  <i>demo</i>
</p>

<details open>
<summary>Install</summary>

To install the project, follow these steps.

1. Clone the project from the repository:

```bash
git clone https://github.com/vo-vuong/Emotion_Detection.git
```

2. Navigate to the project directory:

```bash
cd Emotion_Detection
```

3. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
.venv\Scripts\activate  # For Windows
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference</summary>

Run inferences on 2 different sources `webcam`(default), `image file`.

```bash
python app.py --source webcam           # webcam
                       img.jpg          # image
```

</details>

## Project Structure

```
emotion_detec/
├── constants
│   └── const.py
├── data
├── test_images
├── trained_models                      # the folder containing pretrain model
├── utils
│   ├── data_processing_helpers.py
│   ├── download_model.py
│   ├── matrix_helpers.py
│   └── output_helpers.py
├── app.py                              # main file to run test
├── dataset_analysis.ipynb              # dataset analysis file
├── dataset.py                          # setup dataset for train
├── detect.py
├── models.py                           # CNN model definition file
├── README.md
├── requirements.txt
└── train.py                            # training file
```

## Additional Resources

- [Mediapipe](https://developers.google.com/mediapipe)
- [Facial Expressions Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data)
