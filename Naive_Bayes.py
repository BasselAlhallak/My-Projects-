import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import datetime
import re


def convert_date_to_week_day(data_frame, column_label):
    dates = get_column_values(data_frame, column_label)
    week_days = []
    for date in dates:
        day, month, year = (int(x) for x in date.split('.'))
        ans = datetime.date(year, month, day)
        week_days.append(ans.strftime("%A"))
    week_days = np.asarray(week_days)
    data_frame[column_label] = week_days


def read_csv_file(csv_file_path):
    """
    read csv file and drop unnamed column
    :param csv_file_path:
    :return: pandas data frame of csv data
    """
    df = pd.read_csv(csv_file_path, delimiter=';')
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    return df


def get_df_header(data_frame):
    """

    :param data_frame:
    :return: list of csv data categories
    """
    column_header = list(data_frame)
    return column_header


def get_column_values(data_frame, column_label):
    """
    :param data_frame:
    :param column_label:
    :return: 1-D array of csv column values
    """
    column_value = data_frame[column_label].values
    column_value = column_value.astype(str)
    return column_value


def drop_column(data_frame, column_label):
    """
    :param data_frame:
    :param column_label:
    :return: data frame after dropping a column
    """
    data_frame.drop(column_label, axis=1, inplace=True)
    return data_frame


def get_data_frame_as_list(data_frame):
    """
    :param data_frame:
    :return: data frame as list of rows in string format
    """
    data_frame_list = data_frame.to_string(header=False, index=False, index_names=False).split('\n')
    return data_frame_list


def clean_text(data_frame_list):
    """

    :param data_frame_list:
    :return: data_frame_list after processing
    """
    cleaned_data_frame_list = []
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    for item in data_frame_list:
        item = item.lower()
        item = replace_by_space_re.sub(' ', item)
        cleaned_data_frame_list.append(item)
    return cleaned_data_frame_list


def text_vectorizer(list_of_strings):
    """
    Convert a collection of text documents to a matrix of token counts
    :param list_of_strings:
    :return: array of vectorized data
    """
    vectorizer_text = CountVectorizer()
    x = vectorizer_text.fit_transform(list_of_strings)
    features_vectorized = x.toarray()
    return features_vectorized


def weight_features(features_vectorized):
    """
    weight features by their frequency
    :param features_vectorized:
    :return: none
    """
    unique_tokens, counts_tokens = np.unique(features_vectorized, return_counts=True)
    features_vectorized = features_vectorized.astype(float)
    for unique_token, counts_token in zip(unique_tokens, counts_tokens):
        np.place(features_vectorized, features_vectorized == unique_token, unique_token / counts_token)


def perform_naive_bayes_classification(features, target):
    """
    create train and test sets from the features
    :param features: type: list, value; vectorized feature text
    :param target: type; list, value: vectorized class labels
    :return: 1) 1-D matrix of predicted classes
             2) 1-D matrix of test classes
    """
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    return y_pred, y_test


def check_classification_accuracy(y_test, y_pred):
    """
    by comparing the class test set with the class predictions
    :param y_test:
    :param y_pred:
    :return: model accuracy type float
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def get_plot_confusion_matrix(y_test, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    compute and plot confusion matrix for performance evaluation
    :param y_test:
    :param y_pred:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()