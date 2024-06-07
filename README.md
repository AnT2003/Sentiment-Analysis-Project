# Tiki Sentiment Analysis Project

## Table of Contents
- [Introduction](#introduction)
- [How it works](#how-it-works)
- [Files and Directories](#files-and-directories)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Introduction
Hi, Welcome to my Project
This project aims to collect comment data from the Tiki.vn website and analyze their sentiments. Subsequently, we build a machine learning model to predict the sentiment of comments based on their content. You can use the developed web application to input text and receive predictions about its sentiment.

## How it works
1. **Data Collection**: Use web crawling tools to gather data from the Tiki.vn website with request module on Python. The collected data includes information about products, comments, and customer ratings.
2. **Data Preprocessing**: Preprocess comments to remove missing values, duplicate values, unnecessary characters, normalize text, tokenize Vietnamese words then vectorize and balance data.
3. **Analysis and Visualizations**: Ggenerated some plots like word cloud to visualize the most frequent words used in customer comments. The size of each word represents its frequency in the comments.
4. **Model Building**: Using the preprocessed data, I train a machine learning model, such as Logistic Regression, SGD Classifier, Naive Bayes.
5. **Web Application**: Using flask to develop a basic web application where users can input text and receive sentiment predictions from the trained model.

## Usage
**Run the Web Application**: Dowload file `app.py` and folder `templates` then run `app.py` and access `http://localhost:5000` in your browser to use the web application.
Enter the text that you want to predict the sentiment and click on button "Predict Sentiment" and then this app will return the prediction result

## Conclusion
This project successfully collects data from Tiki.vn, builds 3 machine learning model and use model Logistic Regression with accuracy score approximate 89% to predict comment sentiment, and deploys a web application for users to utilize the model. Improvements in model accuracy can be achieved through further tuning and additional data.

## Future Work
- Enhance model accuracy by experimenting with different algorithms and hyperparameters.
- Implement sentiment analysis for other languages to cater to a wider audience.
- Integrate user authentication and user-specific sentiment analysis for personalized experiences.

## Acknowledgements
We would like to thank the Tiki.vn website for providing valuable data for this project. We also appreciate the open-source community for their contributions to libraries and tools used in this project.
