# train.py ouputs a bag-of-words model for given document
# classes for later use with classify.py
#
# NOTES: all probabilities are done in log space to prevent
# underflow and for speed of calculation.

import sys
import os
import json
import math
import argparse
from collections import Counter

# train_class takes a given set of documents and a document
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
	# cases lead lower probabilities. Limiting the vocabulary count
	# essentially boils down to evening the playing field a bit for
	# the document classes.

	# NOTE: this step leads to slight differences in between
	# model trainings since some words will have the equal counts, and
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

# train_models is the main training driver. It iterates over
# the classes, training a model for each.
def train_models(training_path, model_name, vocabulary_limit):

	# takes a path name to load documents from and returns
	# a list of documents in the form of a list of words
	def load_document_lists(training_path):
		ret = []
		for entry in os.scandir(training_path):
			if entry.is_file():
				with open(entry.path, "r") as document_file:
					# all input files are only one line and
					# have no newline character
					ret.append(document_file.readline().split())
		return ret					

	total_word_count = 0
	total_vocabulary_size = 0
	total_document_count = 0
	model = {}
	model["models"] = []

	abs_training_path = os.path.abspath(training_path)
	for doc_class in os.scandir(abs_training_path):
		if doc_class.is_dir():
			class_docs = load_document_lists(doc_class.path)
			class_model = train_class(class_docs, doc_class.name, vocabulary_limit)
			total_word_count += class_model["total_word_count"]
			total_vocabulary_size += len(class_model["word_counts"])
			total_document_count += class_model["class_document_count"]
			model["models"].append(class_model)

	model["total_word_count"] = total_word_count
	model["total_vocabulary_size"] = total_vocabulary_size
	model["total_document_count"] = total_document_count

	meta = {
		"file_name": model_name,
		"vacabulary_limit": vocabulary_limit
	}
	model["meta"] = meta

	# # write model to file
	with open(model_name + ".json", "w") as f:
		json.dump(model, f)

# Setup for a few command line parameters
def parse_flags():
	parser = argparse.ArgumentParser()
	name_help_string = "Specifies the name of the outputted json model."
	parser.add_argument("-m", "--model-name", type=str,
						help=name_help_string, default="model")
	path_help_string = ("Specifies the directory containing documents"
						" to train with. Should be an absolute or relative"
						" file path.")
	parser.add_argument("-td", "--training-directory", type=str,
						help=path_help_string, default="../data/training/")
	vocab_help_string = ("Specifies the number limit of unique words in"
						 " a given class's vocabulary.")
	parser.add_argument("-vl", "--vocab-limit", type=int,
						help=vocab_help_string, default=1000)
	return parser.parse_args()

def main():
	flags = parse_flags()
	train_models(flags.training_directory,
				flags.model_name,
				flags.vocab_limit)

main()
