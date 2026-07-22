import numpy as np


class MyMat(np.ndarray):

    def __new__(cls,data):
        return np.asarray(data).view(cls)
    @staticmethod
    def mat_cheng(A: MyMat, B: MyMat) -> MyMat:
        A_shap = A.shap()
        B_shap = B.shap()

        
                

