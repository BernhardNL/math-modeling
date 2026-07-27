import numpy as np


def np_for_control(mat_a: np.ndarray, mat_b: np.ndarray) -> bool:
    target = np.linalg.matrix_rank(mat_a)
    C_ctrb = mat_b

    for k in range(1, target):  # target既是用来比较的rank也是kalman判据的列数
        C_ctrb = np.hstack([C_ctrb, mat_a @ mat_b])
        mat_a = mat_a @ mat_a

    C_rank = np.linalg.matrix_rank(C_ctrb)

    if target == C_rank:
        return True
    else:
        return False


def main() -> None:
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[0],
                  [1]]) 

    result = np_for_control(A, B)

    print(f"A:\n{A}")
    print(f"B:\n{B}")

    if result:
        print("能控")
    else:
        print("不能控")


if __name__ == '__main__':
    main()
