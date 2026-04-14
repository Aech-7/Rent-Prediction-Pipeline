# Rent Prediction Pipeline (STT-AI Assignment 4)

## 🚀 Live Application

https://your-username-rent-predictor.hf.space

---

## Project Overview

This project focuses on building an end-to-end pipeline for predicting house rent based on various property features such as location, size, number of rooms, and amenities.

The workflow includes data preprocessing, model training, application development, containerization, and deployment.

---

## Approach

* A **Random Forest Regressor** was used for rent prediction
* Feature engineering and encoding were applied to handle categorical variables
* A **Streamlit application** was developed to allow user input and display predictions
* The application was containerized using **Docker**
* Deployment was carried out using **Hugging Face Spaces (Docker)**

---

## Running the Application

### Without Docker

```bash id="pivuzp"
pip install -r requirements.txt
streamlit run app.py
```

---

### With Docker

Build the image:

```bash id="i4ar45"
docker build -t rent-app .
```

Run the container:

```bash id="cdy0ci"
docker run -p 8501:8501 rent-app
```

Access the app at:

```id="yy8v2d"
http://localhost:8501
```

---

## Project Structure

```id="udbb0v"
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

## Model File

Due to GitHub’s file size limitations, the trained model is not included in the repository.

Download link:
https://your-drive-link

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
* [Teammate Name]
* [Teammate Name]

---
