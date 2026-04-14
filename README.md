# Rent Prediction Pipeline (STT-AI Assignment 4)

## Hugging Face Application

[https://your-username-rent-predictor.hf.space](https://huggingface.co/spaces/Aech07/STTAI_A4)

---

## Project Overview

This project focuses on building an end-to-end pipeline for predicting house rent based on various property features such as location, size, number of rooms, and amenities.

The workflow includes data preprocessing, model training, application development, containerization, and deployment.

---

## Approach

* A **Random Forest Regressor** was used for rent prediction
* Feature engineering and encoding were applied for categorical variables
* A **Streamlit application** was developed for user interaction
* The application was containerized using **Docker**
* Deployment was carried out using **Hugging Face Spaces (Docker)**

---

## Running the Application

### Without Docker

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

### With Docker

Build the image:

```bash
docker build -t rent-app .
```

Run the container:

```bash
docker run -p 8501:8501 rent-app
```

Access the app at:

```
http://localhost:8501
```

---

## Project Structure

```
.
├── app.py
├── Dockerfile
├── requirements.txt
├── models/
│   ├── best_rf_model.pkl
│   └── encoders.pkl
├── screenshots/
└── README.md
```

---

## Model Storage

The trained model files are managed using Git LFS due to their large size.
Make sure Git LFS is installed before cloning the repository:

```bash
git lfs install
git lfs pull
```

---

## Deployment

The application is deployed on Hugging Face Spaces using a Docker-based setup.
The live link is provided at the top of this README.

---

## Screenshots

(Add screenshots here)

---

## Team Members

* Harshada Kale
* Ishika Patel
* Diya Nandan
* Ginisha Garg


---
