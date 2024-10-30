import ctypes

# Define the C struct in ctypes
class Edge(ctypes.Structure):
    _fields_ = [("parent", ctypes.c_int), ("child", ctypes.c_int)]

if __name__ == '__main__':
    lib = ctypes.CDLL('./CDAGOperation.so')
    lib.evaluate_dag_performance_up.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    lib.evaluate_dag_performance_up.restype = ctypes.c_int

    lib.evaluate_dag_performance_down.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    lib.evaluate_dag_performance_down.restype = ctypes.c_int

    #python_tuples = [(0, 10), (10, 14), (14, 2), (14, 7), (7, 3), (7, 6), (3, 8), (8, 9), (9, 11), (9, 13), (13, 1), (6, 4), (4, 5), (4, 12)]
    python_tuples = [(0, 1), (0, 2), (0, 3), (1, 4)]

    c_tuples = (Edge * len(python_tuples))(
        *[Edge(parent, child) for parent, child in python_tuples]
    )


#    python_matrix = [[0, 0, 0.3751, 0, 0, 0, 0, 0, 0, 0, 0.2694, 0, 0, 0, 0,],
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3763, 0, 0.2737, 0,],
# [0.3751, 0, 0, 0, 0, 0.3986, 0.3727, 0, 0, 0, 0, 0, 0, 0, 0.3501,],
# [0, 0, 0, 0, 0, 0, 0, 0.8589, 0.4265, 0.8732, 0, 0.3877, 0, 0, 0,],
# [0, 0, 0, 0, 0, 0.3498, 0.2891, 0, 0, 0, 0, 0.7307, 0.3777, 0, 0,],
# [0, 0, 0.3986, 0, 0.3498, 0, 0, 0.2653, 0, 0, 0, 0, 0, 0, 0,],
# [0, 0, 0.3727, 0, 0.2891, 0, 0, 0.4209, 0, 0, 0, 0, 0, 0, 0,],
# [0, 0, 0, 0.8589, 0, 0.2653, 0.4209, 0, 0, 0, 0, 0, 0, 0, 0.5883,],
# [0, 0, 0, 0.4265, 0, 0, 0, 0, 0, 0.754, 0, 0, 0, 0, 0,],
# [0, 0, 0, 0.8732, 0, 0, 0, 0, 0.754, 0, 0, 0.8514, 0, 0.6349, 0,],
# [0.2694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.303, 0, 0, 0, 0.317,],
# [0, 0.3763, 0, 0.3877, 0.7307, 0, 0, 0, 0, 0.8514, 0, 0, 0, 0, 0,],
# [0, 0, 0, 0, 0.3777, 0, 0, 0, 0, 0, 0, 0, 0.04035, 0, 0,],
# [0, 0.2737, 0, 0, 0, 0, 0, 0, 0, 0.6349, 0, 0, 0, 0.1733, 0.1919,],
# [0, 0, 0.3501, 0, 0, 0, 0, 0.5883, 0, 0, 0.317, 0, 0, 0.1919, 0,]]
    python_matrix = [[0.0, 0.8, 0.4, 0.5, 0.0],
                     [0.8, 0.0, 0.3, 0.1, 0.7],
                     [0.4, 0.3, 0.0, 0.1, 0.1],
                     [0.5, 0.1, 0.1, 0.0, 0.1],
                     [0.0, 0.7, 0.1, 0.1, 0.0]]

    MatrixType = ctypes.POINTER(ctypes.c_float) * len(python_matrix)
    matrix_data = MatrixType(
        *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in python_matrix]
    )

    result = lib.evaluate_dag_performance_up(c_tuples, len(python_tuples), matrix_data, len(python_matrix[0]), 10, 15, 1500)
    print("UP result " + str(result))

    result = lib.evaluate_dag_performance_down(c_tuples, len(python_tuples), matrix_data, len(python_matrix[0]), 10, 15, 1500)
    print("DOWN result " + str(result))