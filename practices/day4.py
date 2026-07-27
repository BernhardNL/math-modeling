import numpy as np


def np_for_control(mat_a: np.ndarray, mat_b: np.ndarray) -> bool:
    target = mat_a.shape[0]
    c_ctrb = mat_b
    mat_a_k = mat_a  #空间换时间

    for k in range(1, target):  # target既是用来比较的rank也是kalman判据的列数（实际是状态空间的方向数）
        c_ctrb = np.hstack([c_ctrb, mat_a_k @ mat_b])
        mat_a_k = mat_a_k @ mat_a

    c_rank = np.linalg.matrix_rank(c_ctrb)

    if target == c_rank:
        return True
    else:
        return False


def  ctr_grammian(mat_a: np.ndarray, mat_b: np.ndarray, n: int = 10) -> np.ndarray:
    assert mat_b.shape == (mat_a.shape[0],1), "格式非SISO"  # 简化版grammian，仅用于尝试所以SISO

    gram = mat_b @ mat_b.T
    term = mat_a @ mat_b @ mat_b.T @ mat_a.T

    for k in range(1, n):
        gram = gram + term 
        term = mat_a @ term @ mat_a.T

    return gram


def main() -> None:
    A = np.array([[1, 2],
                  [3, 4]], dtype = float)
    B = np.array([[0],
                 [1]], dtype = float)  # 为了防止简化的gram溢出，使用float64，也可np.float64（等价）

    result = np_for_control(A, B)
    gram = ctr_grammian(A, B)
    print(f"A:\n{A}")
    print(f"B:\n{B}")

    if result:
        print("能控")
    else:
        print("不能控")

    print(f"能控grammain矩阵为\n{gram}")

if __name__ == '__main__':
    main()
