"""
This module contains various custom transformers and feature engineering classes for preprocessing transaction data.
The transformers are designed to be compatible with sklearn's `BaseEstimator` and `TransformerMixin` classes, allowing them to be easily integrated into sklearn pipelines.

Classes:
    ColumnNameTransformer: A mapper that renames columns in a DataFrame based on a provided mapping.
    ColumnDeleter: A transformer that removes specified columns from a DataFrame.
    DateFeaturesTransformer: A transformer that adds date-related features to the dataset.
    BinAttackFeature: A transformer that adds features related to bin attack detection based on transaction data.
    CustomerSpendingBehaviourFeatures: A transformer that adds features related to customer spending behavior based on transaction data.
    MerchantSpendingBehaviourFeatures: A transformer that adds features related to merchant spending behavior based on transaction data.
    TerminalSpendingBehaviourFeatures: A transformer that adds features related to terminal spending behavior based on transaction data.
    CustomScaler: A custom scaler that scales specified columns in a DataFrame using `StandardScaler`.

Each transformer class includes methods for fitting and transforming the data, compatible with the sklearn transformer interface.
The module is designed to streamline the preprocessing and feature engineering steps.
"""

from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler


class ColumnNameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_mapping):
        self.column_mapping = column_mapping

    def fit(self, X, y=None):
        # This transformer does not need to learn anything from the data
        return self

    def transform(self, X):
        X.rename(columns=self.column_mapping, inplace=True)

        return X


class ColumnDeleter(BaseEstimator, TransformerMixin):
    """
    A transformer that removes specified columns from a DataFrame. The purpose is to
    streamline the dataset for further processing, removing columns that may no longer be
    needed.

    Args:
        BaseEstimator:A base class for all estimators in sklearn.
        TransformerMixin:A mixin class provided by sklearn for transformer classes.
                                    By inheriting from TransformerMixin, a transformer class gains access to additional methods that are commonly used in sklearn transformers

    Attributes:
        columns (list): A list of column names to be removed from the DataFrame.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Removes specificied columns from the input DataFrame.

        Args:
            X (pd.DataFrame): Input dataframe

        Returns:
            X (pd.DataFrame): Output dataframe, contains the transformed df with the specified columns removed while ignoring errors if any specified columns are not present.
        """
        X = X.drop(self.columns, axis=1, errors="ignore")
        return X


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that adds various date-related features to the dataset, and extracts
    features like month, day of the week, whether it's a weekend or not...

    It also calculates the age of the customer based on the difference between the transaction date and the date of birth
    Args:
        BaseEstimator:A base class for all estimators in sklearn.
        TransformerMixin:A mixin class provided by sklearn for transformer classes.
                                    By inheriting from TransformerMixin, a transformer class gains access to additional methods that are commonly used in sklearn transformers
    Methods:
    transform(X): Adds date-related features to the input DataFrame X.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Standard sklearn method that is called during the training phase of a ML model,
        but here it's implemented as a no-op method, it just returns the transformer
        itself ; it is included to adhere to the sklearn transformer interface, so it is
        compatible with the sklearn pipelines and tools.
        """
        return self

    def transform(self, X):
        """
        Enhances the input dataframe X with insightful date-related features.

        Args:
            X (pd.DataFrame): Input dataframe to transform, it contains transaction data.


        Returns:
            X (pd.DataFrame): Transformed dataframe enriched with features like month, day of the week, weekend indicator, month start and month end flags, and an estimate of the customer's age based on transaction data.
        """
        X["DateTimeSystem"] = pd.to_datetime(X["DateTimeSystem"])
        X["month"] = X["DateTimeSystem"].dt.month
        X["day_of_week"] = X["DateTimeSystem"].dt.dayofweek + 1
        X["is_wknd"] = X["DateTimeSystem"].dt.weekday // 4
        X["is_month_start"] = X["DateTimeSystem"].dt.is_month_start.astype(int)
        X["is_month_end"] = X["DateTimeSystem"].dt.is_month_end.astype(int)
        if "DOB":
            # Calculate age based on the difference between transaction date and date of birth
            X["age"] = (
                (pd.to_datetime(X["DateTimeSystem"]) - pd.to_datetime(X["DOB"])).dt.days
                / 365
            ).astype(int)

        # i created a new feature 'time_of_day' to represent day and night
        X["time_of_day"] = np.where(X["DateTimeSystem"].dt.hour < 19, "day", "night")

        return X


class BinAttackFeature(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rules={
            "binAttackFeatures": {
                "window_sizes": ["1h"],
            }
        },
    ):
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def bin_extaction(transaction, numero_carte):
            """
            This function takes in a pandas data frame and a column name containing credit
            card numbers. It then extracts the bin from the first 6 characters of each
            credit card number and adds it as a new column to the data frame. The function
            returns the modified data frame.

            Parameters:
            transaction (pandas dataframe): The input dataframe
            card_colum (string): The name of the column containing the credit card numbers

            Returns:
            pandas dataframe: The modified dataframe with the extracted bin numbers added as a new column
            """
            transaction["bin"] = transaction[numero_carte].apply(lambda x: str(x)[:6])
            return transaction

        def bin_spending_behaviour(BIN_transactions):
            """
            This function computes the behavioral spending patterns of the bin using the
            behavioral insights from the transaction data. It computes the number of
            transactions, the number of rejected transactions, and the percentage of
            rejected transactions for each window size.

            Args:
                BIN_transactions (pd.DataFrame): The input dataframe containing the transaction data.
                date_time_column (str): The name of the column containing the transaction date and time.
                amount_column (str): The name of the column containing the transaction amount.
                window_sizes (List[int]): A list of window sizes in hours to compute the features for.

            Returns:
                pd.DataFrame: The input dataframe with the new features added.
            """

            BIN_transactions = BIN_transactions.sort_values("DateTimeSystem")
            BIN_transactions.index = BIN_transactions["DateTimeSystem"]
            binAttackFeaturesRules = self.rules["binAttackFeatures"]
            for window in binAttackFeaturesRules["window_sizes"]:
                nb_tx_window = BIN_transactions["Montant"].rolling(f"{window}").count()
                rejected_tx_window = (
                    BIN_transactions["IS_rejected"].rolling(f"{window}").sum()
                )

                BIN_transactions[f"bin_nb_tx_{window}"] = nb_tx_window.tolist()
                BIN_transactions[f"bin_rejected_nb_tx_{window}"] = (
                    rejected_tx_window.tolist()
                )
                BIN_transactions[f"%bin_rejected_nb_tx_{window}"] = (
                    BIN_transactions[f"bin_rejected_nb_tx_{window}"]
                    / BIN_transactions[f"bin_nb_tx_{window}"]
                )
                BIN_transactions[f"is_bin_attack_{window}"] = (
                    BIN_transactions[f"%bin_rejected_nb_tx_{window}"] >= 0.5
                ) & (BIN_transactions[f"bin_nb_tx_{window}"] >= 100)
                BIN_transactions.index = BIN_transactions.TRANSACTION_ID

            return BIN_transactions

        X_reset = bin_extaction(X.reset_index(drop=True), "NumCarte")
        X_reset["TRANSACTION_ID"] = X_reset.index
        X_reset["IS_rejected"] = X_reset.CodeReponse != 0
        X_reset["DateTimeSystem"] = pd.to_datetime(X_reset["DateTimeSystem"])

        X_grouped = X_reset.groupby(by="bin", group_keys=True).apply(
            lambda x: bin_spending_behaviour(x)
        )
        return X_grouped.reset_index(drop=True)


class CustomerSpendingBehaviourFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rules={
            "customerSpendingBehaviour": {
                "window_sizes": ["1h"],
            }
        },
    ):
        """
        Initializes the CustomerSpendingBehaviourFeatures transformer.

        Args:
            rules (dict, optional): Dictionary containing rules for different features. Defaults to rule_set.
        """
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Enhances the input dataframe with customer spending behavior features.

        Args:
            X (pd.DataFrame): Input DataFrame containing transaction data

        Returns:
            X (pd.DataFrame): Transformed DataFrame enriched with customer spending behavior features.
            Features include rolling statistics like the number of transactions, the median and average amounts over specified time windows for each customer's transaction.
        """

        def _customer_spending_behaviour(terminal_transactions):
            terminal_transactions = terminal_transactions.sort_values("DateTimeSystem")

            terminal_transactions.index = terminal_transactions["DateTimeSystem"]
            customerSpendingBehaviourRules = self.rules["customerSpendingBehaviour"]
            for window in customerSpendingBehaviourRules["window_sizes"]:
                med_amount_window = (
                    terminal_transactions["Montant"].rolling(f"{window}").median()
                )
                nb_tx_window = (
                    terminal_transactions["Montant"].rolling(f"{window}").count()
                )
                avg_amount_window = (
                    terminal_transactions["Montant"].rolling(f"{window}").mean()
                )

                terminal_transactions[f"customer_nb_tx_{window}"] = nb_tx_window.values
                terminal_transactions[f"customer_median_amount_{window}"] = (
                    med_amount_window.values
                )
                terminal_transactions[f"customer_avg_amount_{window}"] = (
                    avg_amount_window.values
                )

            terminal_transactions["diff_first_day"] = (
                terminal_transactions["DateTimeSystem"]
                - terminal_transactions["DateTimeSystem"].iloc[0]
            ).dt.days

            return terminal_transactions

        transformed_data = X.groupby("NumCarte").apply(_customer_spending_behaviour)
        transformed_data = transformed_data.reset_index(drop=True)
        return transformed_data


class MerchantSpendingBehaviourFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rules={
            "byCodeRejectionFeatures": {
                "windows": ["1h"],
                "utils": {"codes": [51, 61]},
            },
            "highVelocityAuthorizationFeatures": {"windows": ["1h"]},
            "highRejectionFeatures": {"windows": ["1h"]},
            "highTechnicalFallbackFeatures": {"windows": ["1h"]},
            "90DayRuleFeatures": {"windows": ["1d"]},
            "lastTransactionDistance": {"windows": None},
        },
    ):
        """
        Initializes the instance of the transformer.

        Args:
            rules (dict, optional): a dictionary containing necessary information for each mastercard rule. Defaults to rule_set.
        """
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _byCodeRejectionFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of rejected transactions
            for each rejection code in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """merchant_tx_code_
            for code in self.rules["byCodeRejectionFeatures"]["utils"]["codes"]:
                for window in window_sizes:
                    code_rejected_tx_window = (
                        terminal_transactions[f"code_{code}"].rolling(window).sum()
                    )
                    terminal_transactions[f"{code}_{window}"] = (
                        code_rejected_tx_window.tolist()
                    )
            return terminal_transactions

        def _highVelocityAuthorizationFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of accepted transactions
            and total transactions in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_tx_window = (
                    terminal_transactions["CodeAutorisation"].rolling(window).count()
                )
                nb_accepted_tx_window = (
                    terminal_transactions["IsAccepted"].rolling(window).sum()
                )
                terminal_transactions[f"merchant_nb_tx_{window}"] = (
                    nb_tx_window.tolist()
                )
                terminal_transactions[f"merchant_nb_accepted_tx_{window}"] = (
                    nb_accepted_tx_window.tolist()
                )
            return terminal_transactions

        def _highRejectionFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of rejected transactions
            in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_rejected_tx_window = (
                    terminal_transactions["IsRejected"].rolling(window).sum()
                )
                terminal_transactions[f"merchant_nb_rejected_tx_{window}"] = (
                    nb_rejected_tx_window.tolist()
                )
            return terminal_transactions

        def _highTechnicalFallbackFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of technical fallback
            transactions, their ratio and whether or not they exceed 5%, in given time
            windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_mag_tx_window = terminal_transactions["MAG"].rolling(window).sum()
                nb_tx_window = (
                    terminal_transactions["CodeAutorisation"].rolling(window).count()
                )
                ratio_mag_window = (nb_mag_tx_window / nb_tx_window) * 100
                terminal_transactions[f"merchant_nb_mag_{window}"] = (
                    nb_mag_tx_window.tolist()
                )
                terminal_transactions[f"merchant_ratio_mag_{window}"] = (
                    ratio_mag_window.tolist()
                )
                terminal_transactions[f"merchant_ratio_mag_{window}_exceeds"] = (
                    terminal_transactions[f"merchant_ratio_mag_{window}"] > 5
                )
            return terminal_transactions

        def _90DayRuleFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of transactions and the
            average of their amounts for each rejection codes in given time windows. (1
            day and 90 days)

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_tx_window = (
                    terminal_transactions["CodeAutorisation"].rolling(window).count()
                )
                avg_tx_window = terminal_transactions["Montant"].rolling(window).mean()
                terminal_transactions[f"merchant_nb_tx_{window}"] = (
                    nb_tx_window.tolist()
                )
                terminal_transactions[f"merchant_avg_tx_{window}"] = (
                    avg_tx_window.tolist()
                )
            return terminal_transactions

        def _merchantSpatialTimeFeatures(terminal_transactions):
            """
            A function that calculate the distance between two merchants
            terminal_transactions (dataframe): the dataframe to be transformed.

            Returns:
            dataframe: the transformed dataframe.
            """

            def _haversine(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                km = 6371 * c
                return km

            terminal_transactions = terminal_transactions.sort_values("DateTimeSystem")
            terminal_transactions["last_merch_long"] = (
                terminal_transactions["MerchantLong"]
                .shift()
                .fillna(terminal_transactions["MerchantLong"])
            )
            terminal_transactions["last_merch_lat"] = (
                terminal_transactions["MerchantLat"]
                .shift()
                .fillna(terminal_transactions["MerchantLat"])
            )
            terminal_transactions["last_trans_date_trans_time"] = (
                terminal_transactions["DateTimeSystem"]
                .shift()
                .fillna(terminal_transactions["DateTimeSystem"])
            )
            terminal_transactions["diff_time_last_trans_sec"] = (
                terminal_transactions["DateTimeSystem"]
                - terminal_transactions["last_trans_date_trans_time"]
            ).dt.seconds
            terminal_transactions["dist_with_last_merchant"] = [
                _haversine(
                    terminal_transactions["last_merch_lat"].iloc[i],
                    terminal_transactions["last_merch_long"].iloc[i],
                    terminal_transactions["MerchantLat"].iloc[i],
                    terminal_transactions["MerchantLong"].iloc[i],
                )
                for i in range(len(terminal_transactions))
            ]
            return terminal_transactions

        def _merchantSpendingFeatures(terminal_transactions):
            """
            A function that adds the features given in all of the past functions.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.

            Returns:
                dataframe: the transformed dataframe.
            """
            # change the index to datetime
            terminal_transactions.set_index("DateTimeSystem", inplace=True)
            terminal_transactions.sort_index(inplace=True)
            if "byCodeRejectionFeatures" in self.rules.keys():
                for code in self.rules["byCodeRejectionFeatures"]["utils"]["codes"]:
                    terminal_transactions[f"code_{code}"] = (
                        terminal_transactions["CodeReponse"] == code
                    )
                terminal_transactions = _byCodeRejectionFeatures(
                    terminal_transactions,
                    self.rules["byCodeRejectionFeatures"]["windows"],
                )
                print("_byCodeRejectionFeatures")
                print(terminal_transactions.index)
            if "highVelocityAuthorizationFeatures" in self.rules.keys():
                terminal_transactions = _highVelocityAuthorizationFeatures(
                    terminal_transactions,
                    self.rules["highVelocityAuthorizationFeatures"]["windows"],
                )
                print("_highVelocityAuthorizationFeatures")
            if "highRejectionFeatures" in self.rules.keys():
                terminal_transactions = _highRejectionFeatures(
                    terminal_transactions,
                    self.rules["highRejectionFeatures"]["windows"],
                )
                print("_highRejectionFeatures")
            if "90DayRuleFeatures" in self.rules.keys():
                terminal_transactions = _90DayRuleFeatures(
                    terminal_transactions, self.rules["90DayRuleFeatures"]["windows"]
                )
                print("_90DayRuleFeatures")
            if "highTechnicalFallbackFeatures" in self.rules.keys():
                terminal_transactions = _highTechnicalFallbackFeatures(
                    terminal_transactions,
                    self.rules["highTechnicalFallbackFeatures"]["windows"],
                )
                print("_highTechnicalFallbackFeatures")
            if "lastTransactionDistance" in self.rules.keys():
                terminal_transactions.reset_index(inplace=True)

                terminal_transactions = _merchantSpatialTimeFeatures(
                    terminal_transactions
                )
                print("_merchantSpatialTimeFeatures")
            # change back the index to ID
            terminal_transactions.reset_index(inplace=True)
            return terminal_transactions

        X["ID"] = X.index
        X["IsAccepted"] = X["CodeReponse"] == 0
        X["IsRejected"] = X["CodeReponse"] != 0
        X["card_read"] = X["POSEntryMode"].isin(["EMV", "MAG", "ECM"])
        X["non_card_read"] = X["POSEntryMode"].isin(["MOT"])
        X["MAG"] = X["POSEntryMode"] == "MAG"
        # Apply the transformation function to each group (merchant) of the input data
        transformed_data = X.groupby("NomAdresseCommercant").apply(
            lambda x: _merchantSpendingFeatures(x)
        )
        # Reset the index and drop the 'commercant' column
        transformed_data = transformed_data.reset_index(drop=True)
        return transformed_data


class TerminalSpendingBehaviourFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rules: dict = {
            "highAuthorizationFeatures": {"windows": ["5min"]},
            "highRejectionFeatures": {"windows": ["1h"]},
            "averageTransactionAmountFeatures": {"windows": ["1d"]},
        },
    ):
        """
        Initializes the TerminalSpendingBehaviorFeatures transformer.

        Args:
            rules (dict, optional): a dictionary containing necessary information for each mastercard rule. Defaults to rule_set.
        """
        self.rules = rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _highAuthorizationFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of accepted transactions
            and total transactions in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_tx_window = (
                    terminal_transactions["CodeAutorisation"].rolling(window).count()
                )
                nb_accepted_tx_window = (
                    terminal_transactions["IsAccepted"].rolling(window).sum()
                )
                terminal_transactions[f"terminal_nb_tx_{window}"] = (
                    nb_tx_window.tolist()
                )
                terminal_transactions[f"terminal_nb_accepted_tx_{window}"] = (
                    nb_accepted_tx_window.tolist()
                )
            return terminal_transactions

        def _highRejectionFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the number of rejected transactions
            in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                nb_rejected_tx_window = (
                    terminal_transactions["IsRejected"].rolling(window).sum()
                )
                terminal_transactions[f"terminal_nb_rejected_tx_{window}"] = (
                    nb_rejected_tx_window.tolist()
                )
            return terminal_transactions

        def _averageTransactionAmountFeatures(terminal_transactions, window_sizes):
            """
            A function that adds features containing the median and average of transaction
            amounts in given time windows.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.
                window_sizes (tuple): the tuple containing window sizes.

            Returns:
                dataframe: the transformed dataframe.
            """
            for window in window_sizes:
                median_amount_window = (
                    terminal_transactions["Montant"].rolling(window).median()
                )
                avg_amount_window = (
                    terminal_transactions["Montant"].rolling(window).mean()
                )
                terminal_transactions[f"terminal_median_tx_amount_{window}"] = (
                    median_amount_window.tolist()
                )
                terminal_transactions[f"terminal_average_tx_amount_{window}"] = (
                    avg_amount_window.tolist()
                )
            return terminal_transactions

        def _TerminalSpendingBehaviour(terminal_transactions):
            """
            A function that adds the features given in all of the past functions.

            Args:
                terminal_transactions (dataframe): the dataframe to be transformed.

            Returns:
                dataframe: the transformed dataframe.
            """
            # Order transactions chronologically
            terminal_transactions.set_index("DateTimeSystem", inplace=True)
            terminal_transactions.sort_index(inplace=True)
            if "highAuthorizationFeatures" in self.rules.keys():
                terminal_transactions = _highAuthorizationFeatures(
                    terminal_transactions,
                    self.rules["highAuthorizationFeatures"]["windows"],
                )
            if "highRejectionFeatures" in self.rules.keys():
                terminal_transactions = _highRejectionFeatures(
                    terminal_transactions,
                    self.rules["highRejectionFeatures"]["windows"],
                )
            if "averageTransactionAmountFeatures" in self.rules.keys():
                terminal_transactions = _averageTransactionAmountFeatures(
                    terminal_transactions,
                    self.rules["averageTransactionAmountFeatures"]["windows"],
                )
            terminal_transactions.reset_index(inplace=True)
            # Return the dataframe with the new features
            return terminal_transactions

        # Apply the transformation function to each group (merchant) of the input data
        transformed_data = X.groupby("IDTerminal").apply(
            lambda x: _TerminalSpendingBehaviour(x)
        )

        # Reset the index and drop the 'IDTerminal' column
        transformed_data = transformed_data.reset_index(drop=True)
        # Return the transformed data
        return transformed_data


class CustomScaler:
    def __init__(self, scale_columns):
        self.scale_columns = scale_columns
        self.scalers = ColumnTransformer(
            transformers=[("num_scaler", StandardScaler(), self.scale_columns)],
            remainder="passthrough",
        )

    def fit_transform(self, data):
        result_array = self.scalers.fit_transform(data)
        result_df = pd.DataFrame(
            result_array,
            columns=self.scale_columnsall_col.extend(
                data.columns[~data.columns.isin(self.scale_columns)]
            ),
        )
        return result_df

    def transform(self, data):
        result_array = self.scalers.transform(data)
        result_df = pd.DataFrame(
            result_array,
            columns=self.scale_columnsall_col.extend(
                data.columns[~data.columns.isin(self.scale_columns)]
            ),
        )
        return result_df
