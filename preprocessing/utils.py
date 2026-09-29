def window_to_minutes(window: str) -> int:
    """transforms a time window string to the number of minutes

    Args:
        window (str): the sting representing the time window.

    Raises:
        ValueError: A raise if the time window format is not supported.

    Returns:
        int: number of minutes
    """
    if window.endswith("min"):
        return int(window[:-3])
    elif window.endswith("h"):
        return int(window[:-1]) * 60
    elif window.endswith("D"):
        return int(window[:-1]) * 60 * 24
    else:
        raise ValueError("Unsupported time window format")