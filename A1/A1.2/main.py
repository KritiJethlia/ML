import numpy as np
from pre_process import *
from k_cross_fold import *
from k_cross_fold import *	
from create_freq_table import *

def naive_bayes(path = 'dataset_NB.txt'):
	filtered_sentences, targets = pre_process(path)
	fold_sentences, fold_targets = k_cross_fold(filtered_sentences, targets, 7)
	mean_accuracy = []
	for i in range(7):
		test_sentences = fold_sentences[i, :].flatten()
		test_targets = fold_targets[i, :].flatten()

		train_sentences = np.concatenate((fold_sentences[:i, :].flatten(), fold_sentences[i+1:,:].flatten()))
		train_targets = np.concatenate((fold_targets[:i, :].flatten(), fold_targets[i+1:, :].flatten()))

		spam_prob ,not_spam_prob, vocab, spam_sentences, not_spam_sentences = create_freq_table(train_sentences, train_targets)
		# print(len(vocab))

		accuracy = test(spam_prob, not_spam_prob, test_sentences, test_targets, vocab, spam_sentences, not_spam_sentences)
		mean_accuracy.append(accuracy)

	return sum(mean_accuracy)/len(mean_accuracy), mean_accuracy


def test(spam_prob, not_spam_prob, test_sentences, test_targets, vocab, spam_sentences, not_spam_sentences):

	prediction = []
	prior = np.log(spam_sentences/not_spam_sentences)

	for i, sentence in enumerate(test_sentences):
		likelihood = 0
		for word in sentence:
			if word in vocab:
				likelihood+=np.log(spam_prob[word]/not_spam_prob[word])

		posterior = prior + likelihood
		if posterior>=0:
			prediction.append(1)
		else:
			prediction.append(0)

	prediction = np.array(prediction)

	return 1 - sum(np.bitwise_xor(prediction, test_targets))/prediction.shape[0]


if __name__ =='__main__':
	mean_accuracy, accuracy = naive_bayes()
	print(np.array(accuracy)*100)
	print(f'Mean Accuracy = {round(mean_accuracy*100, 2)} %')
	print(f'Lowest Accuracy = {round(min(accuracy)*100, 2)} %')
	print(f'Highest Accuracy = {round(max(accuracy)*100, 2)} %')

	




