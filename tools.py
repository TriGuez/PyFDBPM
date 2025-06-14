import numpy as np

def silica_index(lambda_m):

    lambda_um = lambda_m * 1e6
    sl1, sa1 = 68.4043e-3, 0.6961663
    sl2, sa2 = 116.2414e-3, 0.4079426
    sl3, sa3 = 9896.161e-3, 0.8974794

    X1 = sl1 / lambda_um
    X2 = sl2 / lambda_um
    X3 = sl3 / lambda_um

    X1c = X1**2
    X2c = X2**2
    X3c = X3**2

    T1 = 1 - X1c
    T2 = 1 - X2c
    T3 = 1 - X3c

    epsilon = 1 + sa1 / T1 + sa2 / T2 + sa3 / T3
    return np.sqrt(epsilon)
