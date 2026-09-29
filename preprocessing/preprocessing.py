from typing import Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import inspect
import preprocessing.column_calculator as cc

DEFAULT_WINDOWS = ["60min", "24h"]
ENTITY_IDS = {
    "terminal": "de41_acceptor_terminal_id",
    "merchant": "de42_aceptor_merchant_code",
    "card": "de02_pan",
}


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
        X["requestdate"] = pd.to_datetime(X["requestdate"])
        X["month"] = X["requestdate"].dt.month
        X["day_of_week"] = X["requestdate"].dt.dayofweek + 1
        X["is_wknd"] = X["requestdate"].dt.weekday // 4
        X["is_month_start"] = X["requestdate"].dt.is_month_start.astype(int)
        X["is_month_end"] = X["requestdate"].dt.is_month_end.astype(int)
        if "dob" in X.columns:
            # Calculate age based on the difference between transaction date and date of birth
            X["age"] = (
                (pd.to_datetime(X["requestdate"]) - pd.to_datetime(X["dob"])).dt.days
                / 365
            ).astype(int)

        # i created a new feature 'time_of_day' to represent day and night
        X["time_of_day"] = np.where(X["requestdate"].dt.hour < 19, "day", "night")
        X = cc.merchantSpatialTimeFeatures(X)

        return X


class ColumnCalculator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        metrics: dict,
        entity: Literal["card", "merchant", "terminal", "bin"],
    ):

        self.entity = entity
        self.metrics = metrics

    def fit(self, X, y=None):
        # nothing to do here
        return self

    def calculate_columns(self, df_max_window, entity, metrics, metrics_funcs):
        transformed_data = df_max_window
        transformed_data.set_index("requestdate", inplace=True)
        transformed_data.sort_index(inplace=True)
        for metric in self.metrics:
            func = metrics_funcs.get(metric)
            if func:
                transformed_data = func(
                    df_max_window=df_max_window,
                    window_sizes=metrics[metric]["windows"],
                    entity=entity,
                )
        transformed_data.reset_index(inplace=True)

        return transformed_data

    def transform(self, X) -> pd.DataFrame:
        transformed_data = X

        metrics_funcs = {
            name: func for name, func in inspect.getmembers(cc, inspect.isfunction)
        }

        transformed_data = X  # Initialize with X or the initial data
        transformed_data = transformed_data.groupby(ENTITY_IDS.get(self.entity)).apply(
            lambda x: self.calculate_columns(
                df_max_window=x,
                entity=self.entity,
                metrics=self.metrics,
                metrics_funcs=metrics_funcs,
            )
        )
        transformed_data = transformed_data.reset_index(drop=True)

        return transformed_data


def transform_dataframe(
    df_max_window: pd.DataFrame,
    entity: Literal["card", "merchant", "terminal"],
):

    all_metrics = [
        metric[0]
        for metric in inspect.getmembers(cc, inspect.isfunction)
        if metric[0] not in ["merchantSpatialTimeFeatures"]
    ]

    metrics_with_default_windows = dict(
        zip(all_metrics, len(all_metrics) * [{"windows": DEFAULT_WINDOWS}])
    )
    entitys_metrics_calculator = ColumnCalculator(
        metrics=metrics_with_default_windows,
        entity=entity,
    )

    return entitys_metrics_calculator.transform(df_max_window)
