# 🧠 SOFTDSNL Activity 6: Custom Image Classification with Kaggle Dataset

## 📌 Overview
For your midterm project, you will build a **custom image classification model** using **any dataset from Kaggle**.  
This project extends what we did with MNIST and CIFAR-10, but now you are free to explore real-world datasets.

You will also **deploy your trained model in Django**, so that it accepts image uploads (via Postman) and responds with predictions.

---

## 🎯 Learning Objectives
By completing this project, you will:
- Learn how to source and prepare datasets from Kaggle.
- Train and evaluate a CNN model for image classification.
- Connect your trained model to a Django backend.
- Test predictions via Postman.

---

## File Structure

```
softdsnl-act6
│── requirements.txt
│── README.md
│── train_model.py
│── test_model.py
│── data/  # extracted dataset from Kaggle (In this repo, I use this link (https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). Thanks Sachin!)      
│── my_test_images/    # your own test images for predictions (This is for postman)
│── ml_api_project/    # Django project folder
```
## 📝 Instructions

### 1. Choose a Kaggle Dataset
- Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and pick an image classification dataset.  
- Examples: 
  - Cats vs Dogs
  - Handwritten Letters
  - Fruits/Vegetables classification
- Download the dataset and place it in your project directory.

---

### 2. Preprocess the Data
- Load and normalize images.
- Resize all images to a fixed size (e.g., 64x64 or 128x128).
- Split into training and testing sets.

---

### 3. Build and Train a CNN
- Use TensorFlow/Keras to define a CNN model.
- Compile, train, and evaluate the model.

```
Check train_model.py for the code.
---

### 4. Deploy with Django
- Create a Django project.
- Add an endpoint (e.g., `/predict/`) that:
  1. Accepts an image file upload.
  2. Preprocesses the image (resize, normalize).
  3. Loads your trained model.
  4. Returns the predicted class in JSON.

---

### 5. Test with Postman
- Send **10 test requests** (one per category in your dataset) (Use the images from my_test_images folder.)
- Take screenshots of successful predictions.

---
