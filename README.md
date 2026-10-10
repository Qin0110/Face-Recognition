# Face Recognition Attendance System

A desktop attendance system that identifies people through a webcam and records their
check-in and check-out times. Faces are recognised with an InceptionResNetV1 network
(pretrained on VGGFace2) and all data is kept locally in a SQLite database.

The application runs offline after the first installation. There is no database server
to install and no cloud service involved.

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [The dataset](#the-dataset)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Research scripts](#research-scripts)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Features

- Admin login with PBKDF2-SHA256 hashed passwords
- Guided face enrolment: 70 face images are captured automatically per user
- One-click training that rebuilds the face database and reports its accuracy
- Attendance taking with Enter / Exit status and majority voting over several frames
- Blur, face size and single-face checks so that poor frames are rejected
- User management: search, edit the department, delete a user or retake their photos
- Attendance records with user and date filters, duration per session and CSV export
- Everything stored locally in a single SQLite file

## How it works

1. **Detection** - a Haar cascade locates the face in each camera frame. Frames that are
   blurry, contain no face, contain more than one face, or where the face is too far away
   are rejected before anything else happens.
2. **Enrolment** - 70 cropped face images are saved for the new user.
3. **Training** - the images are split 70:30 into a training and a test set, and each
   training image is converted into a 512-dimension embedding by InceptionResNetV1.
   The embeddings of all users form the face database.
4. **Threshold calibration** - the test set is used to pick the cosine similarity
   threshold that best separates the same person from different people, instead of
   relying on a hard-coded value.
5. **Recognition** - a live face is compared against every stored identity. The score of
   an identity is the mean of its three best matches. If the best score is below the
   calibrated threshold, the face is reported as unknown.
6. **Attendance** - the identity is taken over 30 frames and must win at least 60% of
   them before the entry or exit is written to the database.

## Requirements

- Windows 10 or 11, 64-bit
- A webcam
- An internet connection for the first installation only
- About 6 GB of free disk space
- Python 3.12 (installed automatically if it is missing)

## Installation

1. Double-click **`setup.bat`**. It installs Python 3.12 if needed, creates the `env`
   virtual environment, installs the packages, downloads the pretrained model and trains
   on the images in `Datasets`. This takes roughly 20-45 minutes the first time.
2. Double-click **`run.bat`** to start the application.

`setup.bat` only has to run once per computer. After that, `run.bat` is all that is
needed. If `run.bat` is opened before setup has been done, it offers to run setup first.

The default accounts are `admin` / `admin` and `ADMIN` / `ADMIN`.

## Usage

| Button | What it does |
| --- | --- |
| Add New Admin | Creates another admin account |
| Add New User | Registers a person and captures 70 face images |
| Take Attendance | Records an Enter or Exit for the recognised person |
| Train Model | Rebuilds the face database, required after adding or retaking a user |
| User Information | Lists users; view, edit, delete or retake photos |
| Attendance Records | Filters attendance by user and date, exports to CSV |

Click the camera area at any time to open or close the live preview. The recognised name
is drawn above the face, which is the quickest way to check recognition before taking
attendance.

## The dataset

**No face images are included in this repository.** Face photographs are personal data,
and the images used while building the system come from a third-party dataset, so neither
is redistributed here.

### Where the data came from

Development and measurement used the
[Face Recognition Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)
published by vasukipatel on Kaggle: 31 well known people with between 30 and about 120
photographs each. Only people with at least 70 images are trained on, so 24 of the 31
were used and the rest were reported as skipped. On that data the system reached 94.2%
accuracy, 100% precision and an F1 score of 97.0%.

### Getting the data

Either register people with the application, which is the normal way:

- *Add New User* captures 70 images from the webcam and stores them correctly.

Or reuse a public dataset such as the one above. Download it, then create one folder per
person inside `Datasets`, named exactly as the user ID, holding at least 70 images:

```
Datasets/
    ALICE/
        ALICE_0.jpg
        ALICE_1.jpg
        ...
```

Then start the application and click *Train Model*.

### How the folders are used

| Folder | Content | Purpose |
| --- | --- | --- |
| `Datasets/<user>` | All images of the person | Source of everything below |
| `Prediction/<user>` | First 10 images | Final accuracy measurement |
| `Data/Train/<user>` | 42 images, cropped to the face | Builds the face database |
| `Data/Test/<user>` | 18 images, cropped to the face | Calibrates the similarity threshold |
| `Prediction/Unknown` | Photos of unregistered people | Measures rejection of strangers |

All of these are rebuilt from `Datasets` whenever training runs, and all of them are
ignored by git, so no face image is ever committed.

Collecting photographs of people is subject to privacy law in most countries. Ask for
consent before enrolling anyone, and keep the `Datasets` folder and the database file
private.

## Project structure

```
main.py                  Tkinter user interface
attendanceSystem.py      Application logic, camera session, validation
database.py              SQLite access, password hashing
face_utils.py            Face detection and image helpers
preprocess.py            Dataset selection, train/test split, face cropping
InceptionResNetV1/       Default recognition engine (facenet-pytorch)
HybridApproach/          Optional engine based on face_recognition (dlib)
External/                Prebuilt dlib wheel for Python 3.12
research/                Earlier experiments, kept for reference
setup.bat                One-time installation
run.bat                  Starts the application
```

Folders created at runtime and ignored by git: `env`, `models`, `Data`,
`Rescale_Datasets`, `Prediction`, and `face_recognition.db`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRAS_DB_PATH` | `face_recognition.db` | Location of the SQLite database |
| `FRAS_ENGINE` | `inception` | Set to `hybrid` to use the dlib engine instead |

The tuning constants are at the top of `attendanceSystem.py`, for example the number of
images captured per user, the number of frames used for attendance, and the majority
needed to accept an identity.

Run `env\Scripts\python.exe check_gpu.py` to check whether a GPU is detected. For an
NVIDIA GPU, add the CUDA index and versions to `requirements.txt` before running setup:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.2.0+cu121
torchaudio==2.2.0+cu121
torchvision==0.17.0+cu121
```

## Research scripts

`research/` holds the earlier experiments that led to the current system. They are not
used by the application and are kept so the comparison between the approaches is not
lost:

| File | What it tried |
| --- | --- |
| `CNN.py` | Recognition with face_recognition encodings straight from the webcam |
| `hybrid.py` | Cosine similarity and Euclidean distance combined, with gamma correction |
| `test(last).py` | Accuracy comparison against InsightFace |
| `face_input.py` | Early webcam capture tool |
| `preprocess_dataset.py` | Early dataset splitting and face cropping |

Run them from the project root, for example `env\Scripts\python.exe research\CNN.py`.
Two of them need extras that `setup.bat` does not install: `test(last).py` needs
`insightface` and `onnxruntime`, and `hybrid.py` expects a `hybrid_evaluation` module
that was never part of this repository. The maintained version of that approach is in
`HybridApproach/`, which does work.

## Troubleshooting

| Problem | Solution |
| --- | --- |
| Setup fails while downloading | Check the connection and run `setup.bat` again, it continues where it stopped |
| The camera does not open | Close any other application using the webcam, such as Zoom or Teams |
| "No trained data found" | Click *Train Model* |
| Nobody is recognised | Retake the photos in better light, then train again |
| Something crashed | Details are written to `error.log` in the project folder |

## Limitations

- There is no liveness detection, so a photo held in front of the camera can be accepted.
  The system is meant for cooperative use, not for high security.
- The Haar cascade detector only finds faces looking towards the camera.
- The model has to be retrained after adding a user or retaking photos.
- Admin passwords cannot be changed from the interface yet; create a new admin account
  instead.
- The scripts, camera backend and window handling target Windows.
