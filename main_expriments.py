import pandas as pd
from method import model
import numpy as np
from WKNN import WKNN_method

class Experiments(object):
    def __init__(self,dru_dis,dru_sim,dis_sim,model_name='SPLHNMF', **kwargs):
        super(Experiments, self).__init__()
        self.dru_dis_mat = dru_dis
        self.dru_sim = dru_sim
        self.dis_sim = dis_sim
        self.dru_dis = dru_dis
        self.model = model.SPLHNMF_Model(model_name)
        self.parameters = kwargs

    # def CV_triplet(self):
    #     k_folds = 5
    #     index_matrix = np.array(np.where(self.dru_dis_mat == 1))
    #     positive_num = index_matrix.shape[1]
    #     sample_num_per_fold = int(positive_num / k_folds)
    #
    #     np.random.seed(0)
    #     np.random.shuffle(index_matrix.T)
    #
    #     metrics_mat = np.zeros((1, 6))
    #     for k in range(k_folds):
    #         print('第{}次交叉验证'.format(k+1))
    #         train_matrix = np.array(self.dru_dis_mat, copy=True)
    #         if k != k_folds - 1:
    #             train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
    #         else:
    #             train_index = tuple(index_matrix[:, k * sample_num_per_fold:])
    #
    #         train_matrix[train_index] = 0
    #
    #         concat_drug = np.mat(np.hstack([train_matrix, self.dru_sim]))
    #         concat_disease = np.mat(np.hstack([train_matrix.T, self.dis_sim]))
    #
    #         new_train_matrix = np.mat(train_matrix)
    #
    #         predict_mat = self.model()(new_train_matrix, concat_drug, concat_disease,
    #                                       r=self.parameters['r'], alpha=self.parameters['alpha'],
    #                                       beta=self.parameters['beta'],lamda_l=self.parameters['lamda_l'],
    #                                       tol=1e-4, max_iter=1000)
    #
    #         for num in range(5):
    #             metrics_mat = metrics_mat + self.cv_tensor_model_evaluate(self.dru_dis_mat,
    #                                                                             predict_mat,
    #                                                                             train_index, num)
    #     result = metrics_mat / (k_folds*5)
    #     return result
    def CV_triplet(self):
        k_folds = 10
        index_matrix = np.array(np.where(self.dru_dis_mat == 1))
        positive_num = index_matrix.shape[1]
        sample_num_per_fold = int(positive_num / k_folds)

        np.random.seed(0)
        np.random.shuffle(index_matrix.T)

        metrics_mat = np.zeros((1, 6))  # 用于累计所有折的所有评估结果
        for k in range(k_folds):
            print('第{}次交叉验证'.format(k + 1))
            train_matrix = np.array(self.dru_dis_mat, copy=True)
            if k != k_folds - 1:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                train_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_matrix[train_index] = 0

            concat_drug = np.mat(np.hstack([train_matrix, self.dru_sim]))
            concat_disease = np.mat(np.hstack([train_matrix.T, self.dis_sim]))

            new_train_matrix = np.mat(train_matrix)

            predict_mat = self.model()(new_train_matrix, concat_drug, concat_disease,
                                       r=self.parameters['r'], alpha=self.parameters['alpha'],
                                       beta=self.parameters['beta'], lamda_l=self.parameters['lamda_l'],
                                       tol=1e-4, max_iter=1000)

            fold_sum = np.zeros((1, 6))  # 用于累计当前折的5次评估结果
            for num in range(10):
                metrics = self.cv_tensor_model_evaluate(self.dru_dis_mat, predict_mat, train_index, num)
                fold_sum += metrics
                metrics_mat += metrics

            fold_avg = fold_sum / 10
            print(f'第{k + 1}折平均指标: {fold_avg}')

        result = metrics_mat / (k_folds * 5)
        return result

    def cv_tensor_model_evaluate(self, association_mat, predict_mat, train_index, seed):
        test_po_num = np.array(train_index).shape[1]
        test_index = np.array(np.where(association_mat == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        test_ne_index = tuple(test_index[:, :test_po_num])
        real_score = np.column_stack(
            (np.mat(association_mat[test_ne_index].flatten()), np.mat(association_mat[train_index].flatten())))
        predict_score = np.column_stack(
            (np.mat(predict_mat[test_ne_index].flatten()), np.mat(predict_mat[train_index].flatten())))


        return self.get_metrics(real_score, predict_score)

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

import csv

if __name__ == '__main__':

    # Reading data
    C_sim = pd.read_csv('./Drug_data/Cdataset/drug_sim_snf.csv',index_col=0)
    D_sim = pd.read_csv('./Drug_data/Cdataset/disease_sim_snf.csv',index_col=0)
    CD_data = pd.read_csv('./Drug_data/Cdataset/drug_dis_mat.csv',header=None)

    C_sim_mat = np.mat(np.array(C_sim))
    D_sim_mat = np.mat(np.array(D_sim))
    CD_data_mat = np.mat(np.array(CD_data))

    '''improved association'''
    ## 解决假阴性问题
    new_train_matrix = WKNN_method(CD_data_mat, C_sim_mat, D_sim_mat, 5, 0.9)
    new_train_matrix_data = np.mat(new_train_matrix)

    '''基于新的X'''

    experiment = Experiments(new_train_matrix_data,C_sim_mat,D_sim_mat,model_name='SPLHNMF',
                             r = 39, alpha = 0.04, beta = 0.04,
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
