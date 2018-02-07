# train.py ouputs a bag-of-words model for the spam and
# non-spam document classes for later use with classify.py

import os
import json

# train takes a given set of documents and a document
# class and outputs a json file representing a model
# for that class.
def train(training_path, document_class):
	documents = os.listdir(training_path);
	model = {}
	# accepts a document file and a model to update
	def append_document(document, model):
		with open(training_path + "/" + document, 'r') as doc_file:
			# all input files are only one line and have no newline
			# character
			words = doc_file.readline().split()
			for word in words:
				if word in model:
					model[word] += 1
				else:
					model[word] = 1

	for doc in documents:
		append_document(doc, model)

	json_output = {
		"class" : document_class,
		"model" : model
	}

	with open(document_class + "-model.json", "w") as f:
		json.dump(json_output, f)

train("dummy", "test")