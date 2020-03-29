def uniform_scaling(data, max_len):
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len/max_len)] for j in range(max_len)]

    return scaled_data
