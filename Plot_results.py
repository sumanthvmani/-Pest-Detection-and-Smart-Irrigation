import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
import cv2 as cv
from sklearn import metrics
Types_of_Images = 13


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CO-HC-ARMNet', 'GaOA-HC-ARMNet', 'GOA-HC-ARMNet', 'SBO-HC-ARMNet', 'ISBO-HC-ARMNet']
    Types = ['Pest Detection', 'Plant disease prediction', 'Smart irrigation']
    for i in range(Fitness.shape[0]):
        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ ', Types[i], 'Statistical Report ',
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, marker='*', markerfacecolor='red',
                 markersize=5, label='CO-HC-ARMNet')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, marker='*', markerfacecolor='green',
                 markersize=5, label='GaOA-HC-ARMNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, marker='*', markerfacecolor='blue',
                 markersize=5, label='GOA-HC-ARMNet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, marker='*', markerfacecolor='magenta',
                 markersize=5, label='SBO-HC-ARMNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, marker='*', markerfacecolor='black',
                 markersize=5, label='ISBO-HC-ARMNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/%s.png" % (Types[i]))
        plt.show()



def Images_Sample():
    for n in range(Types_of_Images):
        cls = ['Apple', 'Cherry', 'Batavian Orange', 'Corn', 'Gauva', 'Grape', 'Mango', 'Peach', 'Pepper', 'Potato',
               'Sapota', 'Straberry', 'Tomato']
        Original = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(6, 7):
            print(cls[n])
            Orig_1 = Original[i]
            Orig_2 = Original[i + 1]
            Orig_3 = Original[i + 2]
            Orig_4 = Original[i + 3]
            Orig_5 = Original[i + 4]
            plt.suptitle('Sample Images from ' + cls[n] + ' ', fontsize=20)
            plt.subplot(1, 5, 1).axis('off')
            plt.imshow(Orig_1)
            plt.subplot(1, 5, 2).axis('off')
            plt.imshow(Orig_2)
            plt.subplot(1, 5, 3).axis('off')
            plt.imshow(Orig_3)
            plt.subplot(1, 5, 4).axis('off')
            plt.imshow(Orig_4)
            plt.subplot(1, 5, 5).axis('off')
            plt.imshow(Orig_5)
            path = "./Results/Img_Resss/cls_%s_%s_image.png" % (i - 5, cls[n])
            plt.savefig(path)
            plt.show()
            cv.imwrite('./Results/Img_Resss/Sample-' + str(i - 5) + '-' + str(cls[n]) + '.png', Orig_1)
            cv.imwrite('./Results/Img_Resss/Sample-' + str(i - 4) + '-' + str(cls[n]) + '.png', Orig_2)
            cv.imwrite('./Results/Img_Resss/Sample-' + str(i - 3) + '-' + str(cls[n]) + '.png', Orig_3)
            cv.imwrite('./Results/Img_Resss/Sample-' + str(i - 2) + '-' + str(cls[n]) + '.png', Orig_4)
            cv.imwrite('./Results/Img_Resss/Sample-' + str(i - 1) + '-' + str(cls[n]) + '.png', Orig_5)




def plot_results_Smart_Irrigation():
    eval = np.load('Eval_all_Smart.npy', allow_pickle=True)
    Terms = ['MSE', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'ONE-NORM', 'TWO-NORM', 'INFINITY-NORM']   #MEP
    Graph_Terms = np.array([0, 1, 2, 3, 4, 5, 6]).astype(int)
    Algorithm = ['TERMS', 'CO-HC-ARMNet', 'GaOA-HC-ARMNet', 'GOA-HC-ARMNet', 'SBO-HC-ARMNet', 'ISBO-HC-ARMNet']
    Classifier = ['TERMS', 'DNN', '1DCNN', 'LSTM', 'HC-RMNet ', 'ISBO-HC-ARMNet']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Smart Irrigation Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Smart Irrigation Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j]]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j]]

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.13, 0.13, 0.7, 0.7])
            plt.plot(learnper, Graph[:, 0], color='c', linewidth=3, marker='*', markerfacecolor='lime', markersize=16,
                     label="CO-HC-ARMNet")
            plt.plot(learnper, Graph[:, 1], color='m', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                     label="GaOA-HC-ARMNet")
            plt.plot(learnper, Graph[:, 2], color='lime', linewidth=3, marker='*', markerfacecolor='c', markersize=16,
                     label="GOA-HC-ARMNet")
            plt.plot(learnper, Graph[:, 3], color='b', linewidth=3, marker='*', markerfacecolor='#75bbfd', markersize=12,
                     label="SBO-HC-ARMNet")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=16,
                     label="ISBO-HC-ARMNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(learnper, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Terms[j]])
            path1 = "./Results/Smart_Irrigation_%s_line.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.13, 0.13, 0.7, 0.7])
            # ax = plt.axes(projection="3d")
            # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#c0737a', width=0.10, label="DNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#9a0eea', width=0.10, label="1DCNN")
            ax.bar(X + 0.20, Graph[:, 7], color='#fe01b1', width=0.10, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='#ffb07c', width=0.10, label="HC-RMNet")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="ISBO-HC-ARMNet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.19),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.10, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
            plt.xlabel('Activation Function')
            plt.ylabel(Terms[Graph_Terms[j]])
            # plt.legend(loc=1)
            # path1 = "./Results/_%s_bar.png" % (Terms[Graph_Terms[j]])
            path1 = "./Results/Smart_Irrigation_%s_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()




def Plot_Results_Plant_disease():
    for i in range(1):
        Eval = np.load('Eval_ALL_plant.npy', allow_pickle=True)[i]
        # Terms = ['Accuracy','Precision','FOR','PT','BM','MK','Prevalence', 'TS']
        Terms = np.asarray(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
                            'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])
        Graph_Term = np.array([0, 3, 5, 6, 7, 8, 12, 14]).astype(int)
        Graph_Term_Array = np.array(Graph_Term)
        Algorithm = ['TERMS', 'CO-HC-ARMNet', 'GaOA-HC-ARMNet', 'GOA-HC-ARMNet', 'SBO-HC-ARMNet', 'ISBO-HC-ARMNet']
        Classifier = ['TERMS', 'DNN', '1DCNN', 'LSTM', 'HC-RMNet ', 'ISBO-HC-ARMNet']
        value = Eval[4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[Graph_Term])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Graph_Term])
        print('-------------------------------------------------- Plant disease Algorithm Comparison - Kfold',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms[Graph_Term])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Graph_Term])
        print('--------------------------------------------------- Plant disease Classifier Comparison - Kfold',
              '--------------------------------------------------')
        print(Table)
        Eval = np.load('Eval_ALL_plant.npy', allow_pickle=True)[i]
        BATCH = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    Graph[k, l] = Eval[k, l, Graph_Term[j]+4]
            X = np.arange(5)
            plt.plot(BATCH, Graph[:, 0], '-.', color='#7b5804', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="CO-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 1], '-.', color='#ef4026', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="GaOA-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 2], '-.', color='lime', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="GOA-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 3], '-.', color='#fe02a2', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="SBO-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 4], '-.', color='k', linewidth=3, marker='*', markerfacecolor='white', markersize=16,
                     label="ISBO-HC-ARMNet")
            plt.xlabel('Kfold')
            plt.xticks(X + 1, ('1', '2', '3', '4', '5'))
            # plt.xticks(BATCH, ('4', '8', '16', '32', '48'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Plant_disease_%s_line.png" % (Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#a55af4', edgecolor='k', width=0.12, hatch="..", label="CNN")
            ax.bar(X + 0.12, Graph[:, 6], color='#fa2a55', edgecolor='k', width=0.12, hatch="..", label="1DCNN")
            ax.bar(X + 0.23, Graph[:, 7], color='#7b5804', edgecolor='k', width=0.12, hatch='..', label="LSTM")
            ax.bar(X + 0.36, Graph[:, 8], color='lime', edgecolor='k', width=0.12, hatch="..", label="HC-RMNet")
            ax.bar(X + 0.48, Graph[:, 9], color='k', edgecolor='w', width=0.12, hatch="//", label="ISBO-HC-ARMNet")
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('Kfold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Plant_disease_%s_bar.png" % ((Terms[Graph_Term[j]]))
            plt.savefig(path1)
            plt.show()



def Plot_Results_pest_Detection():
    for i in range(1):
        Eval = np.load('Eval_ALL_Pest.npy', allow_pickle=True)[i]
        # Terms = ['Accuracy','Precision','FOR','PT','BM','MK','Prevalence', 'TS']
        Terms = np.asarray(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 score',
                            'MCC', 'FOR', 'PT', 'BA', 'FM', 'BM', 'MK', 'PLHR', 'Lrminus', 'DOR', 'Prevalence', 'TS'])

        Graph_Term = np.array([0, 1, 2, 3, 4, 5]).astype(int)
        Graph_Term_Array = np.array(Graph_Term)
        Algorithm = ['TERMS', 'CO-HC-ARMNet', 'GaOA-HC-ARMNet', 'GOA-HC-ARMNet', 'SBO-HC-ARMNet', 'ISBO-HC-ARMNet']
        Classifier = ['TERMS', 'DNN', '1DCNN', 'LSTM', '-HC-RMNet ', 'ISBO-HC-ARMNet']
        value = Eval[4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[Graph_Term])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Graph_Term])
        print('-------------------------------------------------- pest Detection Algorithm Comparison - Epochs',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms[Graph_Term])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Graph_Term])
        print('--------------------------------------------------- pest Detection Classifier Comparison - Epochs',
              '--------------------------------------------------')
        print(Table)

        Eval = np.load('Eval_ALL_Pest.npy', allow_pickle=True)[i]

        BATCH = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    Graph[k, l] = Eval[k, l, Graph_Term[j]+4]
            X = np.arange(5)
            plt.plot(BATCH, Graph[:, 0], '-.', color='b', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="CO-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 1], '-.', color='#0ffef9', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="GaOA-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 2], '-.', color='lime', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="GOA-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 3], '-.', color='#920a4e', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                     label="SBO-HC-ARMNet")
            plt.plot(BATCH, Graph[:, 4], '-.', color='k', linewidth=3, marker='*', markerfacecolor='white', markersize=16,
                     label="ISBO-HC-ARMNet")
            plt.xlabel('Epochs')
            plt.xticks(X + 1, ('100', '200', '300', '400', '500'))
            # plt.xticks(BATCH, ('4', '8', '16', '32', '48'))
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/pest_Detection_%s_line.png" % (Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='b', edgecolor='k', width=0.12, hatch="..", label="DNN")
            ax.bar(X + 0.12, Graph[:, 6], color='#b1d1fc', edgecolor='k', width=0.12, hatch="..", label="1DCNN")
            ax.bar(X + 0.23, Graph[:, 7], color='#be03fd', edgecolor='k', width=0.12, hatch='..', label="LSTM")
            ax.bar(X + 0.36, Graph[:, 8], color='lime', edgecolor='k', width=0.12, hatch="..", label="HC-RMNet")
            ax.bar(X + 0.48, Graph[:, 9], color='k', edgecolor='w', width=0.12, hatch="//", label="ISBO-HC-ARMNet")
            plt.xticks(X + 0.25, ('100', '200', '300', '400', '500'))
            plt.xlabel('Epochs')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/pest_Detection_%s_bar.png" % ((Terms[Graph_Term[j]]))
            plt.savefig(path1)
            plt.show()


# CO-HC-ARMNet
# GaOA-HC-ARMNet
# GOA-HC-ARMNet
# SBO-HC-ARMNet
# ISBO-HC-ARMNet
#
#
# DNN
# 1DCNN
# LSTM
# HC-RMNet
# ISBO-HC-ARMNet

def plot_roc():
    lw = 2
    cls = ['Densenet', 'InceptionNet', 'RAN', 'H-ARAN ', 'MCOA-H-ARAN']
    colors = cycle(["c", "#632de9", "#856798", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    Dataset = ['Dataset1', 'Dataset2']
    for j, color in zip(range(5), colors):
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[3, j], Predicted[3, j])
        auc = metrics.roc_auc_score(Actual[3, j], Predicted[3, j])
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[j]
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    plot_results_Smart_Irrigation()
    Plot_Results_pest_Detection()
    Plot_Results_Plant_disease()
    plot_Con_results()










