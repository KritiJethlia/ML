import re

def pre_process(path):
	with open(path, 'r', encoding='latin1') as f:
		contents = f.read()
		contents = contents.split('\n')
		num_lines = len(contents)

	with open(path, 'r') as file:
		sentences = []
		targets = []
		for i in range(num_lines):
			line = file.readline()
			target = line[-2]
			sentence = line[:-2]
			if i == 999:
				target = 0
			sentences.append(sentence)
			targets.append(target)

		targets = list(map(int, targets))

	
	filtered_sentences = []
	stopwords = set([])

	with open('stopword_list.txt', 'r') as file:
		temp = file.read()
		for word in temp:
			stopwords.add(word)

	pattern1 = r'\W'
	pattern2 = r'\d'
	for i in range(len(sentences)):
		sentence = sentences[i]
		words = sentence.split(' ')
		filtered_sentence = []
		for w in words:
			w = w.lower()
			w = re.sub(pattern1, '', w)
			w = re.sub(pattern2, '', w)
			if w and w not in stopwords and not w.isdigit():
				filtered_sentence.append(w)
		filtered_sentences.append(filtered_sentence)

	return filtered_sentences, targets


if __name__ == '__main__':
	filtered_sentences, targets = pre_process('dataset_NB.txt')
	