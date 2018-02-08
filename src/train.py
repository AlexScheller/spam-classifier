# train.py ouputs a bag-of-words model for given document
# classes for later use with classify.py
#
# NOTES: all probabilities are done in log space to prevent
# underflow and for speed of calculation.

import os
import json
import math

# train takes a given set of documents and a document
# class and outputs a json file representing a model
# for that class.
def train_class(class_documents, document_class):
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

	total_word_count = 0
	for word in counts:
		total_word_count += counts[word]

	class_model = {
		"class" : document_class,
		"class_document_count" : len(class_documents),
		"total_word_count" : total_word_count,
		"word_counts" : counts
	}

	# # write the model to a file
	# with open(document_class + "-model.json", "w") as f:
	# 	json.dump(json_output, f)
	return class_model

def train():
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

	doc_classes = ["spam", "nonspam"]
	total_word_count = 0
	total_vocabulary_size = 0
	total_document_count = 0
	model = {}
	model["models"] = []

	for doc_class in doc_classes:
		class_docs = load_document_lists(doc_class)
		class_model = train_class(class_docs, doc_class)
		total_word_count += class_model["total_word_count"]
		total_vocabulary_size += len(class_model["word_counts"])
		total_document_count += class_model["class_document_count"]
		model["models"].append(train_class(class_docs, doc_class))

	model["total_word_count"] = total_word_count
	model["total_vocabulary_size"] = total_vocabulary_size
	model["total_document_count"] = total_document_count

	# write model to file
	with open("model.json", "w") as f:
		json.dump(model, f)

train()
