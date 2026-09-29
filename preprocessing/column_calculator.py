from typing import Literal

import pandas as pd
import numpy as np
from math import asin, cos, radians, sin, sqrt


ENTITY_IDS = {
    "terminal": "de41_acceptor_terminal_id",
    "merchant": "de42_aceptor_merchant_code",
    "card": "de02_pan",
}


def authorization_requests_count_and_value(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of authorization requests for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of authorization requests for each window size.
    """

    for window_size in window_sizes:
        filtered_df = df_max_window

        number_transactions = (
            filtered_df["de04_amount_transaction"].rolling(window_size).count()
        )
        value_transactions = (
            filtered_df["de04_amount_transaction"].rolling(window_size).sum()
        )
        filtered_df[f"{entity}_authorization_tx_number_{window_size}"] = (
            number_transactions.tolist()
        )
        filtered_df[f"{entity}_authorization_tx_value_{window_size}"] = (
            value_transactions.to_list()
        )
    return filtered_df


def approval_rate_and_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the approval rate of authorization requests for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the approval rate of authorization requests for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_accepted"] = filtered_df["de39_response_code"] == 0
    for window_size in window_sizes:

        total_tx = filtered_df["de39_response_code"].rolling(window_size).count()
        approved_tx = filtered_df["is_accepted"].rolling(window_size).sum()
        approval_rate = (approved_tx / total_tx).fillna(0)

        filtered_df[f"{entity}_approval_rate_{window_size}"] = approval_rate.tolist()
        filtered_df[f"{entity}_approval_count_{window_size}"] = approved_tx.tolist()
    filtered_df.drop(columns=["is_accepted"], inplace=True)
    return filtered_df


def ratio_cnp_cp(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
    non_card_read_codes: list[int] = [2, 3, 4, 5, 42, 80],
    card_read_codes: list[int] = [1, 21, 22, 51, 52, 71, 72, 81],
) -> pd.DataFrame:
    """
    Calculate the ratio of non-card read transactions to card read transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the ratio of non-card read transactions to card read transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_non_card_read"] = filtered_df["de22_pos_entry_mode"].isin(
        non_card_read_codes
    )
    filtered_df["is_card_read"] = filtered_df["de22_pos_entry_mode"].isin(
        card_read_codes
    )
    for window_size in window_sizes:

        non_card_read_tx = filtered_df["is_non_card_read"].rolling(window_size).sum()

        card_read_tx = filtered_df["is_card_read"].rolling(window_size).sum()

        non_card_read_ratio = (non_card_read_tx / card_read_tx).fillna(0)
        filtered_df[f"{entity}_non_card_read_count_{window_size}"] = (
            non_card_read_tx.tolist()
        )
        filtered_df[f"{entity}_card_read_count_{window_size}"] = card_read_tx.tolist()
        filtered_df[f"{entity}_ratio_cnp_cp_{window_size}"] = (
            non_card_read_ratio.tolist()
        )
    filtered_df.drop(columns=["is_card_read", "is_non_card_read"], inplace=True)
    return filtered_df


def ratio_and_count_pan_key_entry(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the ratio of transactions where the PAN (Primary Account Number) was entered manually
    to the total number of transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the ratio of transactions where the PAN was entered manually
                  to the total number of transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_pan_key_entry"] = filtered_df["de22_pos_entry_mode"] == 1
    for window_size in window_sizes:

        total_tx = filtered_df["de39_response_code"].rolling(window_size).count()
        pan_entry_key_tx = filtered_df["is_pan_key_entry"].rolling(window_size).sum()

        pan_entry_key_ratio = (pan_entry_key_tx / total_tx).fillna(0)
        filtered_df[f"{entity}_pan_key_entry_count_{window_size}"] = (
            pan_entry_key_tx.tolist()
        )
        filtered_df[f"{entity}_ratio_pan_key_entry_{window_size}"] = (
            pan_entry_key_ratio.tolist()
        )
    filtered_df.drop(columns=["is_pan_key_entry"], inplace=True)
    return filtered_df


def technical_falback_ratio_and_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the ratio of technical fallback transactions to the total number of authorized transactions
    for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the ratio of technical fallback transactions
                  to the total number of authorized transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_technical_fallback"] = (filtered_df["de39_response_code"] == 0) & (
        filtered_df["de22_pos_entry_mode"] == 80
    )
    filtered_df["is_accepted"] = filtered_df["de39_response_code"] == 0
    for window_size in window_sizes:
        total_auth_tx = filtered_df["is_accepted"].rolling(window_size).sum()
        technical_fallback_tx = (
            filtered_df["is_technical_fallback"].rolling(window_size).sum()
        )

        technical_fallback_ratio = (technical_fallback_tx / total_auth_tx).fillna(0)
        filtered_df[f"{entity}_technical_falback_count_{window_size}"] = (
            technical_fallback_tx.tolist()
        )
        filtered_df[f"{entity}_ratiotechnical_falback_{window_size}"] = (
            technical_fallback_ratio.tolist()
        )
    filtered_df.drop(columns=["is_technical_fallback", "is_accepted"], inplace=True)
    return filtered_df


def reversals_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of reversal transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of reversal transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_reversal"] = (
        df_max_window["mti_message_type_identifier"] == 420
    ) & (df_max_window["de39_response_code"] == 0)
    for window_size in window_sizes:

        reversals_count_tx = filtered_df["is_reversal"].rolling(window_size).sum()

        filtered_df[f"{entity}_reversals_count_{window_size}"] = (
            reversals_count_tx.to_list()
        )
    filtered_df.drop(columns=["is_reversal"], inplace=True)
    return filtered_df


def int_transactions_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of international transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of international transactions for each window size.
    """
    filtered_df = df_max_window
    for window_size in window_sizes:

        international_tx_count = (
            filtered_df["is_international"].rolling(window_size).sum()
        )

        filtered_df[f"{entity}_int_transactions_count_{window_size}"] = (
            international_tx_count.to_list()
        )
    return filtered_df


def contactless_transactions_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of contactless authorized transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of contactless authorized transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_contactless"] = (df_max_window["de39_response_code"] == 0) & (
        df_max_window["de22_pos_entry_mode"] == 7
    )
    for window_size in window_sizes:
        filtered_df = df_max_window

        contactless_tx = filtered_df["is_contactless"].rolling(window_size).sum()

        filtered_df[f"{entity}_contactless_transactions_count_{window_size}"] = (
            contactless_tx.to_list()
        )
    filtered_df.drop(columns=["is_contactless"], inplace=True)
    return filtered_df


def mag_stripe_authorization_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of magnetic stripe authorized transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of magnetic stripe authorized transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_mag"] = (df_max_window["de39_response_code"] == 0) & (
        df_max_window["de22_pos_entry_mode"] == 2
    )
    for window_size in window_sizes:
        filtered_df = df_max_window

        mag_stripe_tx = filtered_df["is_mag"].rolling(window_size).sum()
        filtered_df[f"{entity}_mag_stripe_authorization_count_{window_size}"] = (
            mag_stripe_tx.to_list()
        )
    filtered_df.drop(columns=["is_mag"], inplace=True)
    return filtered_df


def authorization_rounded_amount_count(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
    round_value: int = 1000,
) -> pd.DataFrame:
    """
    Calculate the count of transactions with rounded authorization amounts for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.
    round_value (int): The value to which the transaction amount should be rounded,
    make sure to check on the data currency Default is 1.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of transactions with rounded authorization amounts for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_rounded"] = (
        filtered_df["de04_amount_transaction"] % round_value == 0
    )
    for window_size in window_sizes:
        filtered_df = df_max_window

        round_number_tx = filtered_df["is_rounded"].rolling(window_size).sum()
        filtered_df[f"{entity}_authorization_rounded_amount_count_{window_size}"] = (
            round_number_tx.to_list()
        )
    filtered_df.drop(columns=["is_rounded"], inplace=True)
    return filtered_df


def microAmountAuthorizationCount(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
    microtransaction_threshold: int = 1000,  # in millim
) -> pd.DataFrame:
    """
    Calculate the count of micro-amount authorization transactions for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.
    microtransaction_threshold (int): The threshold amount below which transactions are considered micro-amounts. Default is 1000 millim.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of micro-amount authorization transactions for each window size.
    """
    filtered_df = df_max_window
    filtered_df["is_micro"] = (
        filtered_df["de04_amount_transaction"] < microtransaction_threshold
    )
    for window_size in window_sizes:

        micro_tx_count = filtered_df["is_micro"].rolling(window_size).sum()

        filtered_df[f"{entity}_micro_amount_authorization_count_{window_size}"] = (
            micro_tx_count.to_list()
        )
    filtered_df.drop(columns=["is_micro"], inplace=True)
    return filtered_df


def authorization_refund_count_and_value(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["customer", "merchant", "terminal"],
) -> pd.DataFrame:
    filtered_df = df_max_window
    filtered_df["is_refund"] = (filtered_df["de39_response_code"] == 0) & (
        filtered_df["de03_processing_code"] == 20
    )
    filtered_df["value_refund"] = (
        filtered_df["de04_amount_transaction"] * filtered_df["is_refund"]
    )
    for window_size in window_sizes:

        number_of_refunds = filtered_df["is_refund"].rolling(window_size).sum()
        value_of_refunds = filtered_df["value_refund"].rolling(window_size).sum()
        filtered_df[f"{entity}_authorization_refund_count_{window_size}"] = (
            number_of_refunds.to_list()
        )
        filtered_df[f"{entity}_authorization_refund_value_{window_size}"] = (
            value_of_refunds.to_list()
        )
    filtered_df.drop(columns=["is_refund", "value_refund"], inplace=True)
    return filtered_df


def count_unique_responsecodes(
    df_max_window: pd.DataFrame,
    window_sizes: tuple[str],
    entity: Literal["card", "merchant", "terminal"],
) -> pd.DataFrame:
    """
    Calculate the count of unique response codes for a given entity within specified time windows.

    Parameters:
    processing_time (pd._libs.tslibs.timestamps.Timestamp): The current processing time.
    entity_metrics (pd.DataFrame): DataFrame to store the calculated metrics.
    df_max_window (pd.DataFrame): DataFrame containing transaction data.
    window_sizes (tuple[str]): Tuple of window sizes (e.g., '5m', '15m') to calculate metrics for.
    entity (Literal["card", "merchant", "terminal"]): The entity to calculate metrics for.

    Returns:
    pd.DataFrame: Updated entity_metrics DataFrame with the count of unique response codes for each window size.
    """
    code_array = df_max_window["de39_response_code"].unique()
    unique_codes = [
        int(code) for code in code_array[(~np.isnan(code_array)) & (code_array > 0)]
    ]
    filtered_df = df_max_window

    for window_size in window_sizes:

        for code in unique_codes:
            filtered_df[f"code_{code}"] = filtered_df["de39_response_code"] == code
            unique_code_count = filtered_df[f"code_{code}"].rolling(window_size).sum()
            filtered_df[f"{entity}_count_response_code_{code}_{window_size}"] = (
                unique_code_count.to_list()
            )
            filtered_df.drop(columns=[f"code_{code}"], inplace=True)

    return filtered_df


def merchantSpatialTimeFeatures(df_max_window):

    def _haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km

    df_max_window = df_max_window.sort_values("requestdate")

    df_max_window["last_trans_date_trans_time"] = (
        df_max_window["requestdate"].shift().fillna(df_max_window["requestdate"])
    )

    df_max_window["diff_time_last_trans_sec"] = (
        df_max_window["requestdate"] - df_max_window["last_trans_date_trans_time"]
    ).dt.seconds
    if "merch_lat" in df_max_window.columns and "merch_long" in df_max_window.columns:
        df_max_window["last_merch_long"] = (
            df_max_window["merch_long"].shift().fillna(df_max_window["merch_long"])
        )

        df_max_window["last_merch_lat"] = (
            df_max_window["merch_lat"].shift().fillna(df_max_window["merch_lat"])
        )

        df_max_window["dist_with_last_merchant"] = [
            _haversine(
                df_max_window["last_merch_lat"].iloc[i],
                df_max_window["last_merch_long"].iloc[i],
                df_max_window["merch_lat"].iloc[i],
                df_max_window["merch_long"].iloc[i],
            )
            for i in range(len(df_max_window))
        ]
    return df_max_window
