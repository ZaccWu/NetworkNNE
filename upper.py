import numpy as np


def upper(input_matrix):
    """
    Extract the upper-triangular elements (including diagonal) of a square matrix and return as 1D array.
    """
    input_matrix = np.asarray(input_matrix)
    k = input_matrix.shape[0]

    # 获取上三角布尔索引
    mask = np.triu(np.ones((k, k), dtype=bool))

    # 提取上三角元素
    output = input_matrix[mask]
    return output