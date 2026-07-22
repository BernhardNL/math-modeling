import numpy as np
import scipy.linalg as sl


def sci_mat(A: np.ndarray) -> np.ndarray:
    return sl.expm(A)


def just_cul(A: np.ndarray, t: float, N: int) -> np.ndarray:
    n = A.shape[0]  # A矩阵为方阵，只需要（）的第一个值作为生成单位阵的依据
    term = np.eye(n)
    result = np.eye(n)

    for k in range(1, N):
        term = term @ A * t / k
        result += term

    return result


def main() -> None:
    A = np.array([[0, 1],
                 [-2, -3]])

    result1 = sci_mat(A)
    result2 = just_cul(A, 1, 500)

    print("由scipy库的线性代数模块算出的结果为\n", result1)
    print("由循环500次算出的结果为\n", result2)


if __name__ == '__main__':
    main()
