# encoding=utf8

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


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
        social_features[data[0]] = [float(i) for i in data[1:6]]

    res = [] #social_feature vector for each video
    for v in vid:
        try:
            res.append(social_features[vid_uid_dict[v]])
        except:
            #note: there are some users don't have social features, just assgin zero-vector to them
            res.append([0.0, 0.0, 0.0, 0.0, 0.0]) 

    return np.array(res, dtype=np.float32) 


def main():
    data_dir = './data/'
    
    # load data
    print("Loading data...")
    social_feature = load_social_features(data_dir + 'video_id.txt', data_dir + 'video_user.txt', data_dir + 'user_details.txt')
    
    # contatenate all the features(after dimension reduction)
    # concat_feature = np.concatenate([social_feature], axis=1) 
    print("The input data dimension is: (%d, %d)" %(social_feature.shape))
    
    # load ground-truth
    ground_truth = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        #you can use more than one popularity index as ground-truth and average the results; for each video we have four indexes: number of loops(view), likes, reposts, and comments; the first one(loops) is compulsory.
        loop_count = float(line.strip().split('::::')[0])
        like_count = float(line.strip().split('::::')[1])
        repost_count = float(line.strip().split('::::')[2])
        comment_count = float(line.strip().split('::::')[3])
        ground_truth.append((loop_count + like_count + repost_count + comment_count) / 4)
    ground_truth = np.array(ground_truth, dtype=np.float32)
    
    
    print("Start training and predict...")
    kf = KFold(n_splits=10)
    nMSEs = []
    pop_predicts = np.empty([0, 1])

    for train, test in kf.split(social_feature):
        # model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
        model = SVR(kernel='rbf', C=30000, gamma = 0.0001, epsilon = 0.01)
        # train
        model.fit(social_feature[train], ground_truth[train])
        # predict
        predicts = model.predict(social_feature[test])
        # nMSE(normalized Mean Squared Error) metric calculation
        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)
        pop_predicts = np.concatenate((pop_predicts, [[predict] for predict in predicts]))

        print("This round of nMSE is: %f" %(nMSE))
    
    print('Average nMSE is %f.' %(np.mean(nMSEs)))
    return pop_predicts

if __name__ == "__main__":
    main()
