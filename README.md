# 🧠 Sentiment Analysis of Product Reviews 🌟

Category   ➡️   Data Science
Subcategory   ➡️   NLP Engineer
Difficulty   ➡️   Medium

## 🌐 Background

In the age of the Internet, the e-commerce market has seen exponential growth. A pivotal part of online shopping is the reviews that users drop on products they've purchased. These reviews, a treasure trove of data, when analyzed right, can deliver deep insights into customer satisfaction and market preferences.

A leading e-commerce company wishes to harness this data goldmine. They've amassed a plethora of product reviews and are gearing up to analyze them to enhance the service quality they deliver. This is where YOUR challenge begins.

### 🗂️ Dataset

The dataset you'll be harnessing is a modified version of an Amazon product reviews set, consisting of:

- `Summary`: A succinct review summary penned by the user.
- `Text`: The full-blown review content.

The `Score` column, representing the product rating given by the user (on a 1 to 5 scale), has been axed for this challenge. This will be the target variable your model must predict.

## 🎯 Tasks

- Task 1: Your mission, should you choose to accept it, is to craft a classification model that can predict the "rating" of a review, relying solely on its textual content. The company has made some tweaks and omissions to safeguard user privacy. Thus, they'll supply you with just two features to work on: 'Summary' and 'Text'.

## 📊 Evaluation

As part of our commitment to providing a clear and consistent assessment of performance, this practice will be evaluated automatically using the **F1 Micro Score** metric.

## System

We propose starting point "train.py" to finetune models,

source .venv/bin/activate

python src/train.py

## ❓ FAQs

**Q1: What is the aim of analyzing product reviews in this challenge?**
A1: The aim is to develop a classification model that can predict the "rating" of a review based solely on its textual content, providing the e-commerce company with deeper insights into customer satisfaction and market preferences.

**Q2: What information is provided in the dataset for developing the sentiment analysis model?**
A2: The dataset provides two features: 'Summary', which is a concise review summary, and 'Text', which is the full content of the review.

**Q3: What does the `Score` column represent that the model needs to predict?**
A3: The `Score` column represents the product rating given by the user on a scale of 1 to 5, and it is the target variable that your model needs to predict.

**Q4: How will the model's predictions be evaluated in this challenge?**
A4: Predictions will be automatically evaluated using the F1 Score metric, which balances precision and recall, offering a more detailed view of the model's predictive power in text classification.
