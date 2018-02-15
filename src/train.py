# train.py ouputs a bag-of-words model for given document
# classes for later use with classify.py
#
# NOTES: all probabilities are done in log space to prevent
# underflow and for speed of calculation.

import sys
import os
import json
import math
from collections import Counter

# train takes a given set of documents and a document
# class and outputs a json file representing a model
# for that class. A vocabulary limit is provided to
# prevent the build up of many words with small counts,
# which can negatively impact classification accuracy as
# explained below.
def train_class(class_documents, document_class, vocabulary_limit):
	
	# accepts a list of words and a word count map to update
	def append_document(document, counts):
		for word in document:
			if word in counts:
				counts[word] += 1
			else:
				counts[word] = 1

	counts = {}
	for doc in class_documents:
		append_document(doc, counts)

	trimmed = {}
	common = Counter(counts)

	# print("top 5 counts for class: " + document_class)
	# for key, value in common.most_common(5):
	# 	print("{}: {}".format(key, value))
	
	# the vocabulary is trimmed to a certain number of words for
	# each class. This is to prevent issues stemming from how
	# p(w | c) is calculated. the function is: 

	#                (count of w in c) + 1
	# p(w | c) = -----------------------------
	#            (total word count of c) + |V|
	#
	# where |V| is the shared vocabulary size of all classes.
	# see formula (6.14) in "6.pdf" in the background folder for
	# an explanation of the (+ 1) and (+ |V|)

	# If a given class has a very large vocabulary, the denominator
	# will necessarily be large, additionally, if a word has a very
	# low count, the numerator will be correspondingly small. Both
	# cases lead lower probabilities. All this boils down essentially
	# to evening the playing field a bit for the document classes.

	# NOTE: this step leads to slight differences in between
	# model trainings. Some words will have the equal counts, and
	# order is not necessarily preserved between runs. Anecdotally
	# I've only observed accuracy differences of (< 1%).
	for key, value in common.most_common(vocabulary_limit):
		trimmed[key] = value

	total_word_count = 0
	for word in trimmed:
		total_word_count += trimmed[word]

	class_model = {
		"class" : document_class,
		"class_document_count" : len(class_documents),
		"total_word_count" : total_word_count,
		"word_counts" : trimmed
	}

	return class_model

def train_models(model_name="model", vocabulary_limit=1000):
	# takes a path name to load documents from and returns
	# a list of documents in the form of a list of words
	def load_document_lists(training_path):
		absolute_path = os.getcwd() + "/" + training_path + "-train"
		ret = []
		for entry in os.scandir(absolute_path):
			if entry.is_file():
				with open(entry.path, "r") as document_file:
					# all input files are only one line and
					# have no newline character
					ret.append(document_file.readline().split())
		return ret					

	# currently hard coded, but could easily be swapped
	# for other models
	doc_classes = ["../data/spam", "../data/nonspam"]
	# doc_classes = ["spam", "nonspam"]

	total_word_count = 0
	total_vocabulary_size = 0
	total_document_count = 0
	model = {}
	model["models"] = []

	for doc_class in doc_classes:
		class_docs = load_document_lists(doc_class)
		class_model = train_class(class_docs, doc_class, vocabulary_limit)
		total_word_count += class_model["total_word_count"]
		total_vocabulary_size += len(class_model["word_counts"])
		total_document_count += class_model["class_document_count"]
		model["models"].append(class_model)
		# model["models"].append(train_class(class_docs, doc_class))

	model["total_word_count"] = total_word_count
	model["total_vocabulary_size"] = total_vocabulary_size
	model["total_document_count"] = total_document_count

	meta = {
		"file_name": model_name,
		"vacabulary_limit": vocabulary_limit
	}
	model["meta"] = meta

	# # write model to file
	# with open("model.json", "w") as f:
	with open(model_name + ".json", "w") as f:
		json.dump(model, f)

	# write model to compressed file
	# with ZipFile("model.zip", "w") as zf:
	# 	zf.writestr("model.json", json.dumps(model), ZIP_DEFLATED)

# train_models()

def main(args):
	# TODO: integrate argparse instead of this junk
	if (len(args) > 0):
		model_name = args[0]
		if len(args == 1):
			train_models(model_name)
		else:
			vacabulary_limit = int(args[1])
			train_models(model_name, vocabulary_limit)
	else:
		train_models()

main(sys.argv[1:])
