import numpy as np
import torch
from sklearn import tree, svm, naive_bayes
import argparse
import os
import pickle
from joblib import dump, load
from sklearn.metrics import accuracy_score, confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--data_mode", type=str, default='') # _runs_out_align,
    parser.add_argument("--ckpt", type=str, default="") # ./models/spatial_model
    parser.add_argument("--model_type", type=str, default="svm") # ./models/spatial_model
    return parser.parse_args()

# TODO split the features (combining temporal and spatial features) and labels into a file


def load_data(data_path="./"):
    file_list = os.listdir(data_path)
    X, Y = [], []
    for file in file_list: # each file contains feature X and label Y
        data = torch.load(os.path.join(data_path, file))
        X.append(data['features'])
        Y.append(data['label'])
    return np.array(X), np.array(Y)


def eval_model(clf, test_x, test_y, cls_type='svm'):
    pred_y = clf.predict(test_x)
    acc = accuracy_score(y_true=test_y, y_pred=pred_y)
    confu_mat = confusion_matrix(y_true=test_y, y_pred=pred_y)
    print(f"CLS Model {cls_type}, got acc: {acc:.3f} and its confusion matrix is {confu_mat}")


def main():
    if args.mode == 'train':
        train_x, train_y = load_data()
        clf = svm.SVC()
        clf.fit(train_x, train_y)
        dump(clf, "ensemble_cls.joblib")
        val_x, val_y = load_data()
        eval_model(clf, val_x, val_y)

    elif args.mode == 'test':
        test_x, test_y = load_data()
        clf = load("ensemble_cls.joblib")
        eval_model(clf, test_x, test_y)


if __name__ == '__main__':
    args = get_args()
    main()