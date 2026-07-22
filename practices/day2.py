import numpy as np
import matplotlib.pyplot as plt


def mat_show(A: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))

    plt.gca().set_aspect('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.axvline(0, color = 'grey', ls = '--', lw = 0.1)
    plt.axhline(0, color = 'grey', ls = '--', lw = 0.1)

    plt.quiver([0,0], [0,0], A[0], A[1], angles = 'xy', scale_units = 'xy', scale = 1, color = ['r', 'b'], width = 0.01)
    
    plt.grid(alpha = 0.2)
    plt.title('特征向量图')

    plt.show()


def main():
    A = np.array([[1, 0],
                  [0, 2]])
    eigvals, eigvec = np.linalg.eig(A)

    print("A矩阵的特征值为", eigvals)
    print("A矩阵的特征向量为\n",eigvec)

    mat_show(eigvec)


if __name__ == '__main__':
    main()





