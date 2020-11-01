import numpy as np
import torch
from sklearn import tree, svm, naive_bayes
import argparse
import os, random
import pickle
from joblib import dump, load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

label2int = {'ambient':0, 'symphonic':1, 'metal':2, 'rocknroll':3, 'country':4}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--data_mode", type=str, default='') # _runs_out_align,
    parser.add_argument("--ckpt", type=str, default="") # ./models/spatial_model
    parser.add_argument("--model_type", type=str, default="rf") # decision_tree, svm, rf, lr,
    return parser.parse_args()


def load_data(data_path="./"):
    data = torch.load(os.path.join(data_path))
    X, Y = data['features'], data['label']
    return X, Y

# for reading spatial data
def load_data_in_file(data_path="./"):
    file_list = os.listdir(data_path)
    X, Y = [], []
    template = np.load(os.path.join('./data/spatial', 'roi_template.npy'))
    for ind, file in enumerate(file_list):  # each file contains feature X and label Y
        data = torch.load(os.path.join(data_path, file))

        # data only has simple features
        # features = data['features'].flatten()

        # randomly select
        # rand_idx = [i for i in range(data['img'].shape[1])]
        # random.shuffle(rand_idx)
        # features = data['img'][:, rand_idx[:3], :, :].transpose(1, 2, 3, 0).flatten() # get 3 images

        # without template
        features = np.sum(data['img'], axis=1).transpose(1, 2, 0).flatten()

        # use template, only consider non zero
        # features = np.sum(data['img'], axis=1).transpose(1, 2, 0) * template
        # idx = np.where(features != 0)
        # features = features[idx[0], idx[1], idx[2]].flatten()

        X.append(np.expand_dims(features, axis=0))  # reduce dimension later?
        Y.append(label2int[data['label']])
        if ind % 100 == 0:
            print(ind)
    return np.concatenate(X, axis=0), np.array(Y)


def eval_model(clf, test_x, test_y, cls_type='svm'):
    pred_y = clf.predict(test_x)
    acc = accuracy_score(y_true=test_y, y_pred=pred_y)
    confu_mat = confusion_matrix(y_true=test_y, y_pred=pred_y)
    print(f"CLS Model {cls_type}, got acc: {acc:.3f} and its confusion matrix is \n{confu_mat}")


def main():
    template = '_template'
    data_name = 'spatial_features_reduction'
    if args.mode == 'train':
        train_x, train_y = load_data(f'./data/spatial/{data_name}{template}_train.pt')
        if args.model_type == 'svm':
            clf = svm.SVC()
        elif args.model_type == 'decision_tree':
            clf = tree.DecisionTreeClassifier()
        elif args.model_type == 'rf':
            clf = RandomForestClassifier()
        print("Start Training")
        clf.fit(train_x, train_y)
        dump(clf, f"./models/ml/ensemble_cls_{args.model_type}{template}.joblib")
        print("Start Evaluation")
        val_x, val_y = load_data(f'./data/spatial/{data_name}{template}_val.pt')
        eval_model(clf, val_x, val_y, cls_type=args.model_type)

    elif args.mode == 'test':
        test_x, test_y = load_data(f'./data/spatial/{data_name}{template}_test.pt')
        clf = load(f"./models/ml/ensemble_cls_{args.model_type}{template}.joblib")
        eval_model(clf, test_x, test_y, cls_type=args.model_type)


def reduce_dimension():
    '''
    given data to reduce its dimension using PCA/t-sne
    :return:
    '''
    template = ''
    pca_path = f"./models/ml/pca{template}.joblib"
    if not os.path.exists(os.path.join(pca_path)):
        train_x, train_y = load_data_in_file('./data/spatial/train_runs_out_warp')
        print("Start Training")
        pca = PCA(n_components=256, svd_solver='auto', whiten=True).fit(train_x)
        print("Start Transforming")
        dump(pca, pca_path)
        train_x_pca = pca.transform(train_x)
        torch.save({'features': train_x_pca, 'label': train_y}, f'./data/spatial/spatial_features_reduction{template}_train.pt')
    else:
        pca = load(pca_path)
    print("Start")
    val_x, val_y = load_data_in_file('./data/spatial/val_runs_out_warp')
    test_x, test_y = load_data_in_file('./data/spatial/test_runs_out_warp')

    val_x_pca = pca.transform(val_x)
    torch.save({'features': val_x_pca, 'label': val_y}, f'./data/spatial/spatial_features_reduction{template}_val.pt')
    del val_x, val_y
    test_x_pca = pca.transform(test_x)
    torch.save({'features': test_x_pca, 'label': test_y}, f'./data/spatial/spatial_features_reduction{template}_test.pt')


if __name__ == '__main__':
    args = get_args()
    main()
    # reduce_dimension()

    # Template
    # CLS Model svm, got acc: 0.611 and its confusion matrix is
    # [[54 19 12  5  5]
    #  [ 6 61  3  7 18]
    #  [ 5  8 57  8 17]
    #  [ 9  6  5 61 14]
    #  [ 9  9  8 12 57]]

    # CLS Model decision_tree, got acc: 0.419 and its confusion matrix is
    # [[33 17 28  9  8]
    #  [11 40 10 17 17]
    #  [11 16 36  4 28]
    #  [10 16  5 39 25]
    #  [10  7 15 12 51]]

    # CLS Model rf, got acc: 0.623 and its confusion matrix is
    # [[56 16 12  6  5]
    #  [ 8 59  3  7 18]
    #  [ 7  6 57  8 17]
    #  [ 3  6  5 67 14]
    #  [ 9  9  8 12 57]]

    # Without Template
    # CLS Model svm, got acc: 0.611 and its confusion matrix is
    # [[54 19 12  5  5]
    #  [ 6 61  3  7 18]
    #  [ 5  8 57  8 17]
    #  [ 9  6  5 61 14]
    #  [ 9  9  8 12 57]]

    # CLS Model decision_tree, got acc: 0.505 and its confusion matrix is
    # [[51 19 12  8  5]
    #  [11 44 10 12 18]
    #  [ 5  8 45 12 25]
    #  [ 5  7  7 57 19]
    #  [ 9 16 15 12 43]]

    # CLS Model rf, got acc: 0.623 and its confusion matrix is
    # [[51 19 12  6  7]
    #  [ 6 58  3 10 18]
    #  [ 5  5 60 11 14]
    #  [ 3  4  5 69 14]
    #  [ 6  9 10 12 58]]