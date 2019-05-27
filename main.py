
from Naive_Bayes import *

# path to csv data
data_set_path = 'C:/Users/Bassel/Desktop/OSS-SAKI-VUE/Exercise 1/Exercise 1 - Transaction Classification - Data Set.csv'

if __name__ == "__main__":

    data_frame = read_csv_file(data_set_path)

    convert_date_to_week_day(data_frame, 'Buchungstag')

    classification_classes = get_column_values(data_frame, 'label')

    classes = np.unique(classification_classes, return_counts=True)[0]

    data_frame = drop_column(data_frame, 'Buchungstext')
    data_frame = drop_column(data_frame, 'Betrag')
    data_frame = drop_column(data_frame, 'Auftragskonto')
    data_frame = drop_column(data_frame, 'Buchungstag')
    data_frame = drop_column(data_frame, 'Valutadatum')
    data_frame = drop_column(data_frame, 'Waehrung')
    data_frame = drop_column(data_frame, 'label')

    data_frame_list = get_data_frame_as_list(data_frame)

    cleaned_data_frame_list = clean_text(data_frame_list)

    features = text_vectorizer(cleaned_data_frame_list)
    weight_features(features)

    y_pred, y_test = perform_naive_bayes_classification(features, classification_classes)

    accuracy = check_classification_accuracy(y_pred, y_test)
    print('\nClassification accuracy: \n{}'.format(accuracy))

    get_plot_confusion_matrix(y_test, y_pred, classes, title='Confusion matrix')
