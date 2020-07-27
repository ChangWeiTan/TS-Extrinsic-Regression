def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data
