import pandas as pd
from method import model
import numpy as np
from WKNN import WKNN_method
from graph import Graph

class Experiments(object):
    def __init__(self,dru_dis,dru_sim,dis_sim,model_name='SPLHNMF', **kwargs):
        super(Experiments, self).__init__()
        self.dru_dis_mat = dru_dis
        self.dru_sim = dru_sim
        self.dis_sim = dis_sim
        self.dru_dis = dru_dis
        self.model = model.SPLHNMF_Model(model_name)
        self.parameters = kwargs

    def CV_triplet(self):
        drug_num = self.dru_dis_mat.shape[0]

        metrics_mat = np.zeros((1, 6))
        valid_drug_num = 0
        metrics_list = []
        for i in range(drug_num):
            # 当前药物所有已知的药物-疾病关联
            disease_index = np.flatnonzero(self.dru_dis_mat[i, :] == 1)

            print('disease_index:',disease_index)
            # 跳过没有已知关联的药物
            if disease_index.size == 0:
                print(f'第{i + 1}个药物没有正样本，跳过')
                continue

            print(
                f'第{i + 1}/{drug_num}次验证，'
                f'隐藏正样本数：{disease_index.size}'
            )

            valid_drug_num += 1

            # 当前药物的所有正样本索引
            test_index = (
                np.full(
                    disease_index.size,
                    i,
                    dtype=int
                ),
                disease_index
            )


            train_matrix = np.array(self.dru_dis_mat, copy=True)

            # 将当前药物的所有已知关联删除
            train_matrix[test_index] = 0

            graph_SC = np.mat(Graph(np.array(self.dru_sim), 4))
            spar_SC = np.multiply(self.dru_sim, graph_SC)

            graph_SD = np.mat(Graph(np.array(self.dis_sim), 4))
            spar_SD = np.multiply(self.dis_sim, graph_SD)

            concat_drug = np.mat(np.hstack([train_matrix, spar_SC]))
            concat_disease = np.mat(np.hstack([train_matrix.T, spar_SD]))


            new_train_matrix = np.mat(train_matrix)

            predict_mat = self.model()(new_train_matrix, concat_drug, concat_disease,
                                       r=self.parameters['r'], alpha=self.parameters['alpha'],
                                       beta=self.parameters['beta'], lamda_l=self.parameters['lamda_l'],
                                       tol=1e-4, max_iter=1000)

            drug_sum = []
            for num in range(10):
                metrics = self.cv_tensor_model_evaluate(self.dru_dis_mat, predict_mat, test_index, num)
                drug_sum.append(metrics)

            metrics_arr = np.vstack(drug_sum)
            print(metrics_arr)

            best_idx = np.argmax(metrics_arr[:, 0])
            best_metrics = metrics_arr[best_idx:best_idx + 1, :]

            print('第{}个药物平均指标: {}'.format(i + 1, best_metrics))
            metrics_mat += best_metrics
            metrics_list.append(best_metrics)

        result = metrics_mat / (valid_drug_num)

        metrics_array = np.array(metrics_list)

        std_result = np.std(
            metrics_array,
            axis=0,
            ddof=1
        )

        print('\n平均指标:')
        print(result)

        print('\n标准差:')
        print(std_result)

        print(
            '\nAUC: {:.4f} ± {:.4f}'.format(
                result[0, 0], std_result[0]
            )
        )
        print(
            'AUPR: {:.4f} ± {:.4f}'.format(
                result[0, 1], std_result[1]
            )
        )
        print(
            'F1-score: {:.4f} ± {:.4f}'.format(
                result[0, 2], std_result[2]
            )
        )
        print(
            'Precision: {:.4f} ± {:.4f}'.format(
                result[0, 3], std_result[3]
            )
        )
        print(
            'Recall: {:.4f} ± {:.4f}'.format(
                result[0, 4], std_result[4]
            )
        )
        print(
            'Accuracy: {:.4f} ± {:.4f}'.format(
                result[0, 5], std_result[5]
            )
        )

        return result

    def cv_tensor_model_evaluate(self,association_mat,predict_mat,test_positive_index,seed):

        association_mat = np.asarray(association_mat)
        predict_mat = np.asarray(predict_mat)

        row_indices = np.asarray(
            test_positive_index[0],
            dtype=int
        ).reshape(-1)

        positive_cols = np.asarray(
            test_positive_index[1],
            dtype=int
        ).reshape(-1)


        if positive_cols.size == 0:
            raise ValueError('当前测试药物没有正样本')

            # 当前留出的药物编号
        drug_index = int(row_indices[0])

        # 防止 test_positive_index 中混入其他药物
        if not np.all(row_indices == drug_index):
            raise ValueError('test_positive_index 中包含多个药物的索引')

        positive_real_score = association_mat[
            test_positive_index
        ]
        positive_predict_score = predict_mat[
            test_positive_index
        ]

        # 负样本只从当前药物所在行选择
        negative_cols = np.flatnonzero(
            association_mat[drug_index, :] == 0
        )

        rng = np.random.default_rng(seed)
        selected_negative_cols = rng.choice(
            negative_cols,
            size=positive_cols.size,
            replace=negative_cols.size < positive_cols.size
        )

        test_negative_index = (
            np.full(
                selected_negative_cols.size,
                drug_index,
                dtype=int
            ),
            selected_negative_cols
        )

        negative_real_score = association_mat[
            test_negative_index
        ]
        negative_predict_score = predict_mat[
            test_negative_index
        ]

        real_score = np.concatenate([
            negative_real_score,
            positive_real_score
        ]).reshape(1, -1)

        predict_score = np.concatenate([
            negative_predict_score,
            positive_predict_score
        ]).reshape(1, -1)

        return self.get_metrics(np.mat(real_score[0]), np.mat(predict_score[0]))

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]
        print(auc[0, 0], aupr[0, 0], f1_score, precision, recall, accuracy)
        return auc[0, 0], aupr[0, 0], f1_score, precision, recall, accuracy



if __name__ == '__main__':

    # Reading data
    C_sim = pd.read_csv('./Drug_data/Fdataset/drug_sim_snf1.csv',index_col=0)
    D_sim = pd.read_csv('./Drug_data/Fdataset/disease_sim_snf1.csv',index_col=0)
    CD_data = pd.read_csv('./Drug_data/Fdataset/drug_dis_mat.csv',header=None)

    C_sim_mat = np.mat(np.array(C_sim))
    D_sim_mat = np.mat(np.array(D_sim))
    CD_data_mat = np.mat(np.array(CD_data))


    '''improved association'''
    ## 解决假阴性问题
    new_train_matrix = WKNN_method(CD_data_mat, C_sim_mat, D_sim_mat, 5, 0.9)
    new_train_matrix_data = np.mat(new_train_matrix)

    '''基于新的X'''

    experiment = Experiments(new_train_matrix_data,C_sim_mat,D_sim_mat,model_name='SPLHNMF',
                             r = 43, alpha = 0.04, beta = 0.04,
                             lamda_l = 0.00001, tol = 1e-5, max_iter = 1000)
    print(experiment.CV_triplet())
    # r_list = [5,9,13,17,21,25,29]
    #r_list = [3, 7, 11, 15, 19, 23, 27,31,35,39,43,47]
#     r_list =[7]
#     # alpha_list = [0.00002,0.0002,0.002,0.2,1,2]
#     # beta_list = [0.00004,0.0004,0.004,0.4,1,2]
#     # 其他固定参数
#     alpha =0.002
#     beta = 0.04
#     lamda_l = 0.0001
#     tol = 1e-5
#     max_iter = 1000
#     results_list = []
#
#     for r in r_list:
#         experiment = Experiments(new_train_matrix_data, C_sim_mat, D_sim_mat, model_name='SPLHNMF',
#                                  r=r, alpha=alpha, beta=beta,
#                                  lamda_l=lamda_l, tol=tol, max_iter=max_iter)
#         print(experiment.CV_triplet())
#         result = experiment.CV_triplet()
#         results_list.append({
#             'r': r,
#             'alpha': alpha,
#             'beta': beta,
#             'result': result
#         })
#         print(f"r={r}, alpha={0.002}, beta={0.04} -> CV result: {result}")
#
# with open('result.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['r', 'alpha', 'beta', 'AUC', 'AUPR', 'F1', 'Precision', 'Recall', 'Accuracy'])
#     for item in results_list:
#         res = item['result']
#         # 将 (1,6) 矩阵转换为一维列表
#         if hasattr(res, 'A'):      # np.matrix
#             res_flat = res.A[0]
#         elif hasattr(res, 'flatten'):  # np.array
#             res_flat = res.flatten()
#         else:
#             res_flat = res
#         writer.writerow([item['r'], item['alpha'], item['beta']] + list(res_flat))
#
# print("结果已保存到 result.csv")
