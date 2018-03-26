import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

def main():
    data_dir = './data'
    feature_dir = './features'

    # Load data
    print("Loading data...")
    senti_feature = []

    for line in open (os.path.join(feature_dir, 'senti_scores.txt')):
        senti_feature.append(line.strip().split('\t'))
    senti_feature = np.array(senti_feature)
    print(senti_feature)

    # load ground-truth
    ground_truth = []
    for line in open(os.path.join(data_dir, 'ground_truth.txt')):
        #you can use more than one popularity index as ground-truth and average the results; for each video we have four indexes: number of loops(view), likes, reposts, and comments; the first one(loops) is compulsory.
        #ground_truth.append(float(line.strip().split('::::')[0]))
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
    
    for train, test in kf.split(senti_feature):
        # model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
        model = SVR(kernel='rbf', C=75000, gamma = 0.001, epsilon = 0.01)
        # model = GradientBoostingRegressor(max_depth=2, n_estimators=10, learning_rate=0.01, random_state=42)
        # train
        model.fit(senti_feature[train], ground_truth[train])
        # predict
        predicts = model.predict(senti_feature[test])
        # nMSE(normalized Mean Squared Error) metric calculation
        nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
        nMSEs.append(nMSE)
        pop_predicts = np.concatenate((pop_predicts, [[predict] for predict in predicts]))
        print("This round of nMSE is: %f" %(nMSE))
    
    print('Average nMSE is %f.' %(np.mean(nMSEs)))
    return pop_predicts

if __name__ == '__main__':
    main()