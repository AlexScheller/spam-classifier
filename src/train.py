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
def train_class(class_documents, document_class, all_document_count):
	# accepts a list of words and a model to update
	def append_document(document, model):
		for word in document:
			if word in model:
				model[word] += 1
			else:
				model[word] = 1

	model = {}
	for doc in class_documents:
		append_document(doc, model)

	json_output = {
		"class" : document_class,
		"class_document_count" : len(class_documents),
		"all_document_count" : all_document_count,
		"model" : model
	}

	with open(document_class + "-model.json", "w") as f:
		json.dump(json_output, f)

def train():
	# takes a path name to load documents from and returns
	# a list of documents in the form of a list of words
	def load_document_lists(training_path):
		absolute_path = os.getcwd() + "/" + training_path
		ret = []
		for entry in os.scandir(absolute_path):
			if entry.is_file():
				with open(entry.path, "r") as document_file:
					# all input files are only one line and
					# have no newline character
					ret.append(document_file.readline().split())
		return ret					

	class_documents = load_document_lists("dummy")

	train_class(class_documents, "test", len(class_documents))

train()
