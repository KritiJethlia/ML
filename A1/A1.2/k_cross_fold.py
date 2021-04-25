import numpy as np
import random

def k_cross_fold(filtered_sentences, targets, k):
	'''
	Creates dataset for k fold cross validation
	'''
	filtered_sentences = np.array(filtered_sentences)
	targets = np.array(targets)
	indices = list(range(filtered_sentences.shape[0]))
	random.shuffle(indices)

	filtered_sentences = filtered_sentences[indices]
	targets = targets[indices]

	fold_size = int(filtered_sentences.shape[0]/k)

	fold_sentences = []
	fold_targets = []

	for i in range(k):
		fold_sentences.append(filtered_sentences[i*fold_size:min(fold_size*(i+1), filtered_sentences.shape[0])])
		fold_targets.append(targets[i*fold_size:min(fold_size*(i+1), filtered_sentences.shape[0])])

	return np.array(fold_sentences), np.array(fold_targets)


if __name__ == '__main__':
	from pre_process import *
	from k_cross_fold import *

	filtered_sentences, targets = pre_process('dataset_NB.txt')
	fold_sentences, fold_targets = k_cross_fold(filtered_sentences, targets, 7)
	print(fold_sentences)
	print(fold_sentences.shape)
	print(fold_targets.shape)

