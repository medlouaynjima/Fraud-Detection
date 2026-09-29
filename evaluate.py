import numpy as np
import pandas as pd
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def get_class_from_fraud_probability(y_proba, threshold=0.5):
    """
    A function that chooses the classes of a vector of transactions based on their fraud
    probabilities.

    Args:
        y_proba (dataframe): a dataframe of probabilities given by the classification model for a list of transactions.
        threshold (float): the threshhold of classification under which the model chooses not fraud and over which it chooses fraud. Defaults to 0.5.

    Returns:
        vector : a vector of 0s and 1s for each classification, 0 being not fraud and 1 being fraud.
    """
    probabilities = y_proba[y_proba.columns[0]].tolist()
    classified_classes = [
        0 if probability < threshold else 1 for probability in probabilities
    ]
    y_pred = pd.DataFrame(data=classified_classes, columns=["predictions"])
    return y_pred


def get_confusion_matrix(y_test, y_pred):
    """
    A function that returns the confusion matrix based on the predictions of a
    classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the classified entities.
        y_pred (dataframe): the dataframe containing the predictions given by the model for the test set.

    Returns:
        array: the confusion matrix.
    """

    matrix = metrics.confusion_matrix(y_test, y_pred)
    return matrix


def plot_matrix(y_test, y_pred):
    """
    A function that plots the confusion matrix of a given model.

    Args:
        model (object): the classification model.
        X_test (dataframe): a dataframe containing the attributes of the test set.
        y_test (_type_): a dataframe containing the labels of the test set.
    """
    cm = metrics.confusion_matrix(y_test, y_pred)
    f = sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.show()


def compute_MME(y_test, y_pred):
    """
    A function that returns the mean misclassification error for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the mean misclassification rate.
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    mme = 1 - accuracy
    return mme


def compute_accuracy(y_test, y_pred):
    """
    A function that returns the accuracy score for predictions given by a classification
    model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the accuracy score.
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def compute_recall(y_test, y_pred):
    """
    A function that returns the recall score for predictions given by a classification
    model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the recall score.
    """
    recall = metrics.recall_score(y_test, y_pred)
    return recall


def compute_TNR(y_test, y_pred):
    """
    A function that returns the true negative rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the true negative rate.
    """
    matrix = get_confusion_matrix(y_test, y_pred)
    TN = matrix[0][0]
    FP = matrix[1][0]
    return TN / (TN + FP)


def compute_FPR(y_test, y_pred):
    """
    A function that returns the false positive rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the false positive rate.
    """
    TNR = compute_TNR(y_test, y_pred)
    FPR = 1 - TNR
    return FPR


def compute_FNR(y_test, y_pred):
    """
    A function that returns the false negative rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the false negative rate.
    """
    TPR = compute_recall(y_test, y_pred)
    FNR = 1 - TPR
    return FNR


def compute_BER(y_test, y_pred):
    """
    A function that returns the balanced error rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the balanced error rate.
    """
    FPR = compute_FPR(y_test, y_pred)
    FNR = compute_FNR(y_test, y_pred)
    BER = (FPR + FNR) / 2
    return BER


def compute_BAR(y_test, y_pred):
    """
    A function that returns the balanced accuracy rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the balanced accuracy rate.
    """
    BAR = metrics.balanced_accuracy_score(y_test, y_pred)
    return BAR


def compute_GMEAN(y_test, y_pred):
    """
    A function that returns the geometric mean for predictions given by a classification
    model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the geometric mean.
    """
    TPR = compute_recall(y_test, y_pred)
    TNR = compute_TNR(y_test, y_pred)
    G_mean = math.sqrt(TPR * TNR)
    return G_mean


def compute_precision(y_test, y_pred):
    """
    A function that returns the precision score for predictions given by a classification
    model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the precision score.
    """
    precision = metrics.precision_score(y_test, y_pred)
    return precision


def compute_NPV(y_test, y_pred):
    """
    A function that returns the negative predictive value for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the negative predictive value.
    """
    matrix = get_confusion_matrix(y_test, y_pred)
    TN = matrix[0][0]
    FN = matrix[0][1]
    return TN / (TN + FN)


def compute_FDR(y_test, y_pred):
    """
    A function that returns the false discovery rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the false discovery rate.
    """
    precision = compute_precision(y_test, y_pred)
    FDR = 1 - precision
    return FDR


def compute_FOR(y_test, y_pred):
    """
    A function that returns the false ommission rate for predictions given by a
    classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the false ommission rate.
    """
    NPV = compute_NPV(y_test, y_pred)
    FOR = 1 - NPV
    return FOR


def compute_F1_score(y_test, y_pred):
    """
    A function that returns the F1 score for predictions given by a classification model.

    Args:
        y_test (dataframe): a dataframe containing the true labels of the predicted transactions.
        y_pred (dataframe): a dataframe containing the predictions given by the model.

    Returns:
        float: the F1 score.
    """
    F1 = metrics.f1_score(y_test, y_pred)
    return F1


def threshhold_based_metrics(
    y_test: list,
    y_proba: list,
    unique_threshholds: list,
    metrics=[
        "Threshold",
        "MME",
        "Recall",
        "TNR",
        "FPR",
        "FNR",
        "BER",
        "BAR",
        "G-mean",
        "Precision",
        "NPV",
        "FDR",
        "FOR",
        "F1 Score",
    ],
):
    """
    A function that gives you the chosen threshhold-based metrics for a list of different
    threshholds.

    Args:
        y_proba (dataframe): the fraud probabilities given by the classification model.
        y_test (dataframe): the true labels for the classified entities.
        unique_threshholds (vector): the list of chosen threshholds.
        metrics(vector): the list of vectors to be shown.
    Returns:
        dataframe: a dataframe containing the threshhold-based metrics for every threshhold for the same classification model.
    """

    columns = [
        "Threshold",
        "MME",
        "Recall",
        "TNR",
        "FPR",
        "FNR",
        "BER",
        "BAR",
        "G-mean",
        "Precision",
        "NPV",
        "FDR",
        "FOR",
        "F1 Score",
    ]
    metrics_df = pd.DataFrame(columns=columns)
    for threshhold in unique_threshholds:
        y_pred = get_class_from_fraud_probability(y_proba, threshhold)
        MME = compute_MME(y_test, y_pred)
        recall = compute_recall(y_test, y_pred)
        TNR = compute_TNR(y_test, y_pred)
        FPR = compute_FPR(y_test, y_pred)
        FNR = compute_FNR(y_test, y_pred)
        BER = compute_BER(y_test, y_pred)
        BAR = compute_BAR(y_test, y_pred)
        Gmean = compute_GMEAN(y_test, y_pred)
        precision = compute_precision(y_test, y_pred)
        NPV = compute_NPV(y_test, y_pred)
        FDR = compute_FDR(y_test, y_pred)
        FOR = compute_FOR(y_test, y_pred)
        F1 = compute_F1_score(y_test, y_pred)
        metrics_df.loc[len(metrics_df.index)] = [
            threshhold,
            MME,
            recall,
            TNR,
            FPR,
            FNR,
            BER,
            BAR,
            Gmean,
            precision,
            NPV,
            FDR,
            FOR,
            F1,
        ]
    return metrics_df[metrics]


def compute_AUC_ROC(y_test, y_proba):
    """
    A function that returns the area under the curve of the receiving operating
    characteristic for a classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the test set.
        y_proba (dataframe): the dataframe containing the probability scores of the predicted transactions.

    Returns:
        float: the area under the ROC curve.
    """
    AUC = metrics.roc_auc_score(y_test, y_proba)
    return AUC


def plot_ROC_curve(y_test, y_proba):
    """
    A function that plots the receiving operating characteristic curve for a
    classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the test set.
        y_proba (dataframe): the dataframe containing the probability scores of the predicted transactions.
    """
    fpr_list, tpr_list, threshholds = metrics.roc_curve(y_test, y_proba)
    roc_curve, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=15)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("Recall", fontsize=15)
    ax.plot(fpr_list, tpr_list, color="blue")
    plt.show()


def compute_AUC_PR(y_test, y_proba):
    """
    A function that returns the area under the curve of precision plotted against recall
    for a classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the test set.
        y_proba (dataframe): the dataframe containing the probability scores of the predicted transactions.

    Returns:
        float: the area under the PR curve.
    """
    precision_list, recall_list, threshholds = metrics.precision_recall_curve(
        y_test, y_proba
    )
    auc = metrics.auc(recall_list, precision_list)
    return auc


def plot_PR_curve(y_test, y_proba):
    """
    A function that plots the precision recall curve for a classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the test set.
        y_proba (dataframe): the dataframe containing the probability scores of the predicted transactions.
    """
    precision_list, recall_list, threshholds = metrics.precision_recall_curve(
        y_test, y_proba
    )
    PR_curve, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Precision Recall Curve", fontsize=15)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)
    ax.plot(recall_list, precision_list, "b", color="blue")
    plt.show()


def compute_AP(y_test, y_proba):
    """
    A function that returns the average precision value for a classification model.

    Args:
        y_test (dataframe): the dataframe containing the true labels of the test set.
        y_proba (dataframe): the dataframe containing the probability scores of the predicted transactions.

    Returns:
        float: the average precision.
    """
    AP = metrics.average_precision_score(y_test, y_proba)
    return AP


def precision_top_k_day(df_day, top_k=100):
    """
    A function that returns for the precision for the top k (if not assigned, top 100)
    transactions for the first day of the test set.

    Args:
        df_day (dataframe): the dataframe containing the attributes of the transactions alongside the probability value of the predictions.
        top_k (int, optional): the number of the top transactions used in calculating the precision. Defaults to 100.

    Returns:
        list_detected_fraudulent_transactions(vector): the list of the transactions detected fraudulent.
        precision_top_k(float): the value of the top-k precision.
    """

    # Order transactions by decreasing probabilities of frauds
    df_day = df_day.sort_values(by="predictions", ascending=False).reset_index(
        drop=False
    )

    # Get the top k most suspicious transactions
    df_day_top_k = df_day.head(top_k)
    list_detected_fraudulent_transactions = list(
        df_day_top_k[df_day_top_k.TX_FRAUD == 1].TRANSACTION_ID
    )

    # Compute precision top k
    precision_top_k = len(list_detected_fraudulent_transactions) / top_k

    return list_detected_fraudulent_transactions, precision_top_k


def precision_top_k(predictions_df, top_k=100):
    """
    The function that returns the top k precision (top 100 if not assigned) for the entire
    test set.

    Args:
        predictions_df (dataframe): the dataframe containing the attributes of the transactions alongside the probability value of the predictions.
        top_k (int, optional): the number of the top transactions used in calculating the precision. Defaults to 100.

    Returns:
        nb_fraudulent_transactions_per_day(vector): list containing the number of fraudulent transactions of each day.
        precision_top_k_per_day_list(vector): list containing the top-k precision per day.
        mean_precision_top_k(float): the mean of the list containing daily top k precisions.
    """

    # Sort days by increasing order
    list_days = list(predictions_df["TX_TIME_DAYS"].unique())
    list_days.sort()

    precision_top_k_per_day_list = []
    nb_fraudulent_transactions_per_day = []

    # For each day, compute precision top k
    for day in list_days:
        df_day = predictions_df[predictions_df["TX_TIME_DAYS"] == day]
        df_day = df_day[["TRANSACTION_ID", "CUSTOMER_ID", "TX_FRAUD", "predictions"]]

        nb_fraudulent_transactions_per_day.append(len(df_day[df_day.TX_FRAUD == 1]))

        _, _precision_top_k = precision_top_k_day(df_day, top_k=top_k)

        precision_top_k_per_day_list.append(_precision_top_k)

    # Compute the mean
    mean_precision_top_k = np.round(np.array(precision_top_k_per_day_list).mean(), 3)

    # Returns number of fraudulent transactions per day,
    # precision top k per day, and resulting mean
    return (
        nb_fraudulent_transactions_per_day,
        precision_top_k_per_day_list,
        mean_precision_top_k,
    )


def card_precision_top_k_day(df_day, top_k):
    """
    A function that returns the top k card precision for a single day.

    Args:
        df_day (dataframe): a dataframe containing the attributes of transactions for a day including the probability value of the prediction.
        top_k (int): the number of the transactions used in calculating the metric.

    Returns:
        list_detected_compromised_cards(vector): list of fraudulent transactions for the given day.
        card_precision_top_k(float): the value of the top-k card precision for that day.
    """

    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = (
        df_day.groupby("CUSTOMER_ID")
        .max()
        .sort_values(by="predictions", ascending=False)
        .reset_index(drop=False)
    )

    # Get the top k most suspicious cards
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(
        df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID
    )

    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k

    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    """
    A function that returns the top k card precision for the entire test set.

    Args:
        predictions_df (dataframe):  a dataframe containing the attributes of transactions including the probability value of the prediction.
        top_k (int): the number of the transactions used in calculating the metric.
        remove_detected_compromised_card (boolean): a boolean indicating whether or not the compromised cards are removed from the dataset. defaults to true.

    Returns:
        nb_compromised_cards_per_day(vector): list containing the number of compromised cards per day.
        card_precision_top_k_per_day_list(vector): list containing the top-k card precision per day.
        mean_card_precision_top_k(float): the mean of the list containing daily top k card precisions.
    """

    # Sort days by increasing order
    list_days = list(predictions_df["TX_TIME_DAYS"].unique())
    list_days.sort()

    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []

    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []

    # For each day, compute precision top k
    for day in list_days:
        df_day = predictions_df[predictions_df["TX_TIME_DAYS"] == day]
        df_day = df_day[["predictions", "CUSTOMER_ID", "TX_FRAUD"]]

        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[
            df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False
        ]

        nb_compromised_cards_per_day.append(
            len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique())
        )

        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(
            df_day, top_k
        )

        card_precision_top_k_per_day_list.append(card_precision_top_k)

        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)

    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return (
        nb_compromised_cards_per_day,
        card_precision_top_k_per_day_list,
        mean_card_precision_top_k,
    )


if __name__ == "__main__":
    y_test = []
    y_proba = []
    unique_threshholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    threshhold_based_metrics(y_test, y_proba, unique_threshholds, metrics=["Threshold"])
