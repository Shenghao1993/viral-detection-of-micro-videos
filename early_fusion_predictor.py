# encoding=utf8

import os
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


def load_social_features(video_id, video_user, user_details):
    vid = [] #video id list
    for line in open(video_id):
        vid.append(line.strip())
   
    vid_uid_dict = {} #vid-uid mapping
    for line in open(video_user):
        data = line.strip().split('::::')
        vid_uid_dict[data[0]] = data[1]
    
    social_features = {} #uid-social_feature mapping
    for line in open(user_details):
        data = line.strip().split("::::")
        # You should modify here to add more user social information
        #here we only use two user social infomation: loops and followers. You should consider more user social information. For more details about other social information, pls refer to ./data/README.txt -> 4.user_details.txt 
        social_features[data[0]] = [float(i) for i in data[1:6]]

    res = [] #social_feature vector for each video
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            #note: there are some users don't have social features, just assgin zero-vector to them
            res.append([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(res, dtype=np.float32)

def svc_param_selection(X, y, nfolds):
    Cs = [1, 10, 100, 1000]
    gammas = [0.001, 0.003, 0.005, 0.01]
    epsilon = [0.1, 0.01]
    param_grid = {'C': Cs, 'gamma' : gammas, "epsilon":epsilon}
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def main():
    data_dir = './data/'
    feature_dir = './features'
    
    # load data
    print("Loading data...")
    hist_feature = np.load(data_dir + 'histogram_feature.npz')['arr_0']
    imgNet_feature = np.load(data_dir + 'imageNet_feature.npz')['arr_0']
    vSenti_feature = np.load(data_dir + 'visual_senti_feature.npz')['arr_0']
    sen2vec_feature = np.load(data_dir + 'text_sentence2vec_feature.npz')['arr_0']
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')

    senti_feature = []
    for line in open (os.path.join(feature_dir, 'senti_scores.txt')):
        senti_feature.append(line.strip().split('\t'))
    senti_feature = np.array(senti_feature)
    print(senti_feature)
    
    # feature dimension reduction: it's up to you to decide the size of reduced dimensions; the main purpose is to reduce the computation complexity
    pca = PCA(n_components=20)
    hist_feature = pca.fit_transform(hist_feature)
    # 20, 40, 10
    pca = PCA(n_components=10)
    imgNet_feature = pca.fit_transform(imgNet_feature)
    pca = PCA(n_components=20)
    vSenti_feature = pca.fit_transform(vSenti_feature)
    pca = PCA(n_components=10)
    sen2vec_feature = pca.fit_transform(sen2vec_feature)
    
    # contatenate all the features(after dimension reduction)
    concat_feature = np.concatenate([hist_feature, imgNet_feature, vSenti_feature, sen2vec_feature, social_feature, senti_feature], axis=1) 
    print("The input data dimension is: (%d, %d)" %(concat_feature.shape))
    
    # load ground-truth
    ground_truth = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        loop_count = float(line.strip().split('::::')[0])
        like_count = float(line.strip().split('::::')[1])
        repost_count = float(line.strip().split('::::')[2])
        comment_count = float(line.strip().split('::::')[3])
        ground_truth.append((loop_count + like_count + repost_count + comment_count) / 4)
    ground_truth = np.array(ground_truth, dtype=np.float32)
    
    # print("Start tuning model parameters...")

    # print(svc_param_selection(concat_feature, ground_truth, 10))

    print("Start training and predict...")
    kf = KFold(n_splits=10)
    nMSEs = []
    pop_predicts = np.empty([0, 1])

    for train, test in kf.split(concat_feature):
        # model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
        model = SVR(kernel='rbf', C=75000, gamma = 0.0001, epsilon = 0.01)
        # model = GradientBoostingRegressor(max_depth=10, n_estimators=200, learning_rate=0.1, random_state=42)
        # train
        model.fit(concat_feature[train], ground_truth[train])
        # predict
        predicts = model.predict(concat_feature[test])
        # nMSE(normalized Mean Squared Error) metric calculation
        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)
        pop_predicts = np.concatenate((pop_predicts, [[predict] for predict in predicts]))
        print("This round of nMSE is: %f" %(nMSE))
    
    print('Average nMSE is %f.' %(np.mean(nMSEs)))
    return pop_predicts

if __name__ == "__main__":
    main()
