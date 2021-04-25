def create_freq_table(filtered_sentences, targets):
# Store spam and not spam frequencies of word
	spam_freq = {}
	not_spam_freq = {}
	spam_vocab = set()
	not_spam_vocab = set()
	spam_sentences = 0
	not_spam_sentences = 0
	word_freq_spam = 0
	word_freq_not_spam = 0

	for i, sentence in enumerate(filtered_sentences):
		spam = targets[i]
		if spam:
			spam_sentences+=1
		else:
			not_spam_sentences+=1


		for word in sentence:
			if spam:
				spam_vocab.add(word)
				spam_freq[word] = spam_freq.get(word, 0) + 1
				word_freq_spam+=1

			else:
				not_spam_vocab.add(word)
				not_spam_freq[word] = not_spam_freq.get(word, 0) + 1
				word_freq_not_spam+=1

	spam_prob, not_spam_prob = {}, {}

	keys = set(list(spam_freq.keys()) + list(not_spam_freq.keys())) 

	vocab = set([])
	vocab = vocab.union(spam_vocab)
	vocab = vocab.union(not_spam_vocab)

	for word in keys:
		spam_prob[word] = (spam_freq.get(word, 0) + 1)/(word_freq_spam + len(vocab))
		not_spam_prob[word] = (not_spam_freq.get(word, 0) + 1)/(word_freq_not_spam + len(vocab))

	# spam_vocab.union(not_spam_vocab)
	# vocab = spam_vocab

	return spam_prob ,not_spam_prob, vocab, spam_sentences, not_spam_sentences 	


if __name__ == '__main__':
	from pre_process import *
	from k_cross_fold import *

	filtered_sentences, targets = pre_process('dataset_NB.txt')
	fold_sentences, fold_targets = k_cross_fold(filtered_sentences, targets, 7)
	train_sentences = []
	train_targets = []

	for i in range(6):
		for j in range(len(fold_sentences[i])):
			train_sentences.append(fold_sentences[i][j])
			train_targets.append(fold_targets[i][j])

	spam_prob ,not_spam_prob, vocab, spam_sentences, not_spam_sentences = create_freq_table(train_sentences, train_targets)
	print(not_spam_prob)