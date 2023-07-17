# Fake News Detection using Machine Learning

This project uses machine learning algorithms to detect fake news from real news. The dataset consists of news articles with labels indicating whether they are fake or real. The dataset is processed and cleaned, then split into training and testing sets. Two machine learning algorithms are implemented - Support Vector Machines (SVM) and Logistic Regression (LR) - to predict whether a given news article is fake or real.

## Requirements
- pandas
- numpy
- seaborn
- matplotlib
- NLTK
- sklearn
- simple_colors

You can install these libraries using pip:

```
pip install pandas numpy seaborn matplotlib NLTK sklearn simple_colors
```

## Data Preprocessing
The dataset is preprocessed by removing the null values, adding a new field, dropping features that are not needed, and performing text processing to remove symbols and stopwords.

## Model Fitting
Two machine learning algorithms are implemented - SVM and LR - to predict whether a news article is fake or real. The dataset is split into training and testing sets, and the algorithms are trained on the training set. The accuracy of each model is calculated, and a confusion matrix is plotted to visualize the performance of each model.
- Accuracy of SVM model was: 0.9831
- Accuracy of LR model was:  0.9761 

## Model Testing
A function is implemented to test the SVM model on a given news article. The function takes in a news article as input and returns whether it is fake or real.

## Credits
- This code was created by [Nada Osama](https://github.com/NadaOsamaa)
- [Fake News Detection Dataset](https://www.kaggle.com/datasets/jruvika/fake-news-detection?select=data.csv) was obtained from Kaggle
