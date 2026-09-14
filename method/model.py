
import numpy as np
import math
from method import ConstructHW

import warnings
warnings.filterwarnings("ignore")


class SPLHNMF_Model(object):

    def __init__(self,name = 'SPLHNMF'):
        super().__init__()
        self.name = name

    def SPLHNMF(self, X, c_embeding, d_embeding, r, alpha, beta, lamda_l, tol, max_iter = 1000):

        m = X.shape[0]
        d = X.shape[1]

        '''初始化要保证非负'''
        np.random.seed(0)
        U = np.mat(np.random.rand(m, r))
        V = np.mat(np.random.rand(d, r))

        '''Hypergraph Learning'''
        Dv_u,S_u = ConstructHW.constructHW(c_embeding)
        Dv_v,S_v = ConstructHW.constructHW(d_embeding)

        'Initialize self-paced weight'
        W = np.mat(np.ones(m*d).reshape(m, d))


        k = 1
        k_end = 0.008
        gama = 1.2
        eps = np.finfo(float).eps

        while k > k_end:
            print('k为{}值'.format(k))
            for i in range(max_iter):
                output_X_old = U * V.T

                '''Updating the matrix U'''
                max_value_U = np.linalg.norm(U, 2, axis=1)
                d_U = 1 / max_value_U
                A = np.mat(np.diag(d_U.tolist()))

                new_X = np.mat(np.multiply(W, X))
                temp_U_m = new_X * V + alpha * S_u * U
                temp_U_d = np.multiply(W, U * V.T) * V + lamda_l * U + lamda_l * A * U + alpha * Dv_u * U
                temp_U = temp_U_m / (temp_U_d + eps)
                U = np.mat(np.multiply(U, temp_U))

                '''Updating the matrix V'''
                max_value_V = np.linalg.norm(V, 2, axis=1)
                d_V = 1 / max_value_V
                B = np.mat(np.diag(d_V.tolist()))

                temp_V_m = new_X.T * U + beta * S_v * V
                temp_V_d = np.multiply(W.T, V * U.T) * U + lamda_l * V + lamda_l * B * V + beta * Dv_v * V
                temp_V = temp_V_m / (temp_V_d + eps)
                V = np.mat(np.multiply(V, temp_V))

                output_X = U * V.T
                err = np.linalg.norm(output_X - output_X_old) / np.linalg.norm(output_X_old)

                print('SPLHNMF err:', err)
                if err < tol:
                    print("第{}次迭代的误差{}".format(i, err))
                    break

            '''reconstructed matrix'''
            New_output_X = U * V.T

            for I in range(New_output_X.shape[0]):
                for J in range(New_output_X.shape[1]):
                    loss = math.pow((X[I, J] - New_output_X[I, J]), 2)
                    if (loss <= 1 / math.pow((k + 1 / gama), 2)):
                        W[I, J] = 1
                    elif (loss >= 1 / math.pow(k, 2)):
                        W[I, J] = 0
                    else:
                        W[I, J] = gama * (1 / np.sqrt(loss) - k)

            k = k / 1.2

        predict_X = np.array(U * V.T)

        return predict_X

    def __call__(self):

        return getattr(self, self.name, None)












