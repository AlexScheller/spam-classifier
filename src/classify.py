# classify.py makes use of a bag-of-words model produced by train.py
# to classify documents.
import os
import json
import math
import argparse

# loads the count models from file, and returns an dictionary
# representation with necessary probabilities calculated in
# log space
def load_models(json_model):

	ret_model = {}
	ret_model["class_models"] = []

	total_vocabulary_size = json_model["total_vocabulary_size"]
	total_document_count = json_model["total_document_count"]

	for doc_class in json_model["models"]:
		
		new_class = {}
		new_class["class_name"] = doc_class["class_name"]
		floating_prior_prob = doc_class["class_document_count"] / total_document_count
		new_class["class_prior_prob"] = math.log(floating_prior_prob)
		
		new_class["word_cond_probs"] = {}
		for word, count in doc_class["word_counts"].items():
			# Laplace smoothing included
			floating_prob = (count + 1) / (doc_class["total_word_count"] + total_vocabulary_size)
			new_class["word_cond_probs"][word] = math.log(floating_prob)

		ret_model["class_models"].append(new_class)

	return ret_model

# This method classifies with a simple word count model,
# and is actually fairly accurate.
# included for interest, but unused.
def classify_by_word_presence(document, model):
	prob_class = ""
	highest_ratio = 0
	for doc_class in model["class_models"]:
		words_found = 0
		for word in document:
			if word in doc_class["word_cond_probs"]:
				words_found += 1
		find_ratio = words_found / len(document)
		if find_ratio >= highest_ratio:
			highest_ratio = find_ratio
			prob_class = doc_class["class_name"]
	return prob_class

# accepts a document in the form of a list of words and a
# model to test against
def classify(document, model):

	highest_prob = None
	prob_class = ""
	for doc_class in model["class_models"]:
		prior_prob = doc_class["class_prior_prob"]
		running_word_prob = 0

		words_found = 0
		for word in document:
			# Words unseen during training are simply ignored.
			if word in doc_class["word_cond_probs"]:
				words_found += 1
				running_word_prob += doc_class["word_cond_probs"][word]
		total_prob = prior_prob + running_word_prob
		
		# Since probability is being dealt with in log space,
		# the numbers will be negative. Here they are flipped to
		# positive. This is purely cosmetic, as the magnitudes of
		# the probabilities don't actually change.
		total_prob = -total_prob
		if highest_prob is None or total_prob >= highest_prob:
			highest_prob = total_prob
			prob_class = doc_class["class_name"]
	return prob_class

# test_model() is the main driver function for testing
# the accuracy of a given model.
def test_model(model, test_file_directory):

	# tests the accuracy of a given class.
	# "test_docs" are assumed to all be of the given class.
	def test_class(doc_class, test_docs):
		docs_correctly_classified = 0
		total_docs = len(test_docs)
		for doc in test_docs:
			if classify(doc, model) == doc_class:
				docs_correctly_classified += 1
		result_string = "{} / {} or {:.2f}%".format(
						docs_correctly_classified,
						total_docs,
						(docs_correctly_classified / total_docs) * 100)
		return "Accuracy for " + doc_class + ": " + result_string

	def load_test_docs(test_doc_directory):
		ret = []
		for entry in os.scandir(test_doc_directory):
			if entry.is_file():
				with open(entry.path, "r") as doc_file:
					# all files are a collection of space
					# separated words on a single line with no
					# new-line.
					ret.append(doc_file.readline().split())
		return ret

	for document_class in model["class_models"]:
		class_name = document_class["class_name"]
		test_path = os.path.abspath(test_file_directory + class_name)
		result = test_class(class_name, load_test_docs(test_path))
		print(result)

# Setup for a few command line parameters
def parse_flags():
	parser = argparse.ArgumentParser()
	parser.add_argument("-td", "--testing-directory", type=str,
						help="specifies location of test documents",
						default="../data/testing/")

	parser.add_argument("-mf", "--model-file", type=str,
						help="specifies location of model file",
						default="../data/model.json")
	return parser.parse_args()

def main():
	flags = parse_flags()
	try:
		with open(flags.model_file, "r") as mf:
			model = load_models(json.load(mf))
			test_model(model, flags.testing_directory)
	except FileNotFoundError:
		print("no model found at: {}".format(flags.model_file))

main()