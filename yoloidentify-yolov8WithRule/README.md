# YOLOv8 Recognition Pipeline

This repository contains the cleaned inference pipeline for image recognition.

## What It Detects

The main detector supports these labels:

```text
gun
knife
blood
syringe
pill
powder
swastika
nazi_symbol
poker_card
dice
chip
roulette
slot_machine
middle_finger
crowd
flag
banner
signboard
police
```

An optional pretrained face detector adds:

```text
face
```

## Files

```text
scripts/recognize.py
scripts/person_recognition.py
scripts/person_stub_eval.py
weights/detector_19class.pt
weights/blood_classifier.pt
weights/face_detector.pt
data/political_persons.txt
data/celebrity_face_db/facebank/person_facebank_full.npz
data/celebrity_face_db/facebank/person_facebank_full_manifest.json
data/yolo_final_19class/data.yaml
requirements.txt
```

## Install

```powershell
pip install -r requirements.txt
```

The original working environment was:

```powershell
D:\MyAnaconda\envs\pic_recog\python.exe
```

## Run

Single image:

```powershell
python scripts/recognize.py path\to\image.jpg --enable-face
```

Save JSON result:

```powershell
python scripts/recognize.py path\to\image.jpg --enable-face --output outputs\result.json
```

Save annotated image:

```powershell
python scripts/recognize.py path\to\image.jpg --enable-face --save-image outputs\annotated.jpg
```

Update political person list:

```text
data/political_persons.txt
```

Test person recognition only:

```powershell
python scripts/person_stub_eval.py predict path\to\image.jpg
```

## Evaluation 测试命令

To run the evaluation script against the test dataset and generate accuracy reports (both business label level and original class level):

```powershell
# Install additional dependencies for evaluation
pip install pandas matplotlib seaborn tqdm tabulate

# Run the evaluation script
python scripts/evaluate.py
```

The evaluation script will output:
- `outputs/evaluation_results.csv`: Detailed prediction results for all images
- `outputs/business_label_accuracy.csv` & `.png`: Accuracy metrics for business categories
- `outputs/original_class_accuracy.csv` & `.png`: Image-level accuracy metrics for original detection classes

## Output

The output is JSON. For a single image:

```json
{
  "image": "path/to/image.jpg",
  "elapsed_seconds": 0.08,
  "device": "0",
  "categories": ["gun", "face"],
  "category_counts": {
    "gun": 1,
    "face": 2
  },
  "business_category": "暴恐",
  "face_recognition": {
    "provider": "RetinaFace + ArcFace",
    "status": "ok",
    "face_count": 2,
    "identities": [
      {
        "name": "普通人",
        "matched": false
      }
    ]
  },
  "detections": []
}
```

`categories` is the de-duplicated category list for the image. `business_category` is mapped by the rule layer. If `face_recognition.identities[].name` matches a name in `data/political_persons.txt`, the business category is `涉政`. `face_recognition` is only populated when the output contains `face`; matched identities use names from the facebank, while unmatched faces are returned as `普通人`. `detections` keeps box coordinates and confidence scores.
