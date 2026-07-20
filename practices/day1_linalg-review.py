import numpy as np


def main():
    A = np.array([[5, 6],
                  [7, 8]])
    B = np.full((2, 2), 7)

    C = np.linalg.inv(A)
    D = np.linalg.det(B)
    
    print("A的转置为\n", A.T)
    print("A的逆为\n", C)
    print("B的行列式为\n", D)


if __name__ == '__main__':
    main()

