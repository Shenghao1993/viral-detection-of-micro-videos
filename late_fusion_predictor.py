import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import visual_predictor
import textual_predictor
import social_predictor

def main():
	feature_dir = './features/'
	data_dir = './data/'

	visual_predicts = np.array(visual_predictor.main())
	textual_predicts = np.array(textual_predictor.main())
	social_predicts = np.array(social_predictor.main())
	combined_predicts = np.concatenate([visual_predicts, textual_predicts, social_predicts], axis = 1)
	print(combined_predicts.shape)

	fout = open(os.path.join(feature_dir, 'predicts.txt'), 'w')
	for predict in combined_predicts:
		pop = '\t'.join(str(pop) for pop in predict)
		fout.write('%s\n' %pop)
	fout.close()

	# combined_predicts = []

	# for line in open (os.path.join(feature_dir, 'predicts.txt')):
	# 	combined_predicts.append([float(predict) for predict in line.strip().split('\t')])
	# combined_predicts = np.array(combined_predicts)
	# print(combined_predicts)

	# load ground-truth
	ground_truth = []
	for line in open(os.path.join(data_dir, 'ground_truth.txt')):
		loop_count = float(line.strip().split('::::')[0])
		like_count = float(line.strip().split('::::')[1])
		repost_count = float(line.strip().split('::::')[2])
		comment_count = float(line.strip().split('::::')[3])
		ground_truth.append((loop_count + like_count + repost_count + comment_count) / 4)
	ground_truth = np.array(ground_truth, dtype=np.float32)

	kf = KFold(n_splits=10)
	nMSEs = []

	for train, test in kf.split(combined_predicts):
		# model initialize: you can tune the parameters within SVR(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html); Or you can select other regression models
		model = LinearRegression()
		# train
		model.fit(combined_predicts[train], ground_truth[train])
		# predict
		predicts = model.predict(combined_predicts[test])
		# nMSE(normalized Mean Squared Error) metric calculation
		nMSE = mean_squared_error(ground_truth[test], predicts) / np.mean(np.square(ground_truth[test]))
		nMSEs.append(nMSE)

		print("This round of nMSE is: %f" %(nMSE))
    
	print('Average nMSE is %f.' %(np.mean(nMSEs)))

if __name__ == "__main__":
    main()