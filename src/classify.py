import json
import math
from zipfile import ZipFile

# loads the count models from file, and returns an dictionary
# representation with necessary probabilities calculated in
# log space
def load_models(json_model):

	# print(json.dumps(json_model))

	ret_model = {}
	ret_model["class_models"] = []

	# total_word_count = json_model["total_word_count"]
	total_vocabulary_size = json_model["total_vocabulary_size"]
	total_document_count = json_model["total_document_count"]

	for doc_class in json_model["models"]:
		
		new_class = {}
		new_class["class"] = doc_class["class"]
		floating_prior_prob = doc_class["class_document_count"] / total_document_count
		new_class["class_prior_prob"] = math.log(floating_prior_prob)
		
		new_class["word_cond_probs"] = {}
		for word, count in doc_class["word_counts"].items():
			# Laplace smoothing included
			floating_prob = (count + 1) / (doc_class["total_word_count"] + total_vocabulary_size)
			# print(floating_prob)
			new_class["word_cond_probs"][word] = math.log(floating_prob)

		ret_model["class_models"].append(new_class)
		# print(json.dumps(new_class))

	return ret_model


# accepts a document in the form of a list of words and a
# model to test against
def classify(document, model):
	# print(document)
	highest_prob = None
	prob_class = ""
	for doc_class in model["class_models"]:
		prior_prob = doc_class["class_prior_prob"]
		running_word_prob = 0
		for word in document:
			# choose to ignore unseen words. They could
			# also be represented by a artificial "unkown"
			# token inserted into each model.
			if word in doc_class["word_cond_probs"]:
				running_word_prob += doc_class["word_cond_probs"][word]
		total_prob = prior_prob + running_word_prob
		total_prob = -total_prob # cause of log prob
		# print(json.dumps(doc_class))
		print(doc_class["class"] + " " + str(total_prob))
		if highest_prob is None or total_prob >= highest_prob:
			highest_prob = total_prob
			prob_class = doc_class["class"]
	yield prob_class
	yield highest_prob

def main():

	json_model = {}

	# with ZipFile("model.zip") as zf:
	# 	with zf.open("model.json") as mf:
	# 		# print(json.loads(mf.read()))
	# 		json_model = json.load(mf)

	with open("model.json", "r") as mf:
		json_model = json.load(mf)

	model = load_models(json_model)
	# example document
	# doc_file = "testdoc.txt"
	# doc_file = "../data/nonspam-test/3-391msg1.txt"
	doc_file = "../data/spam-test/spmsga138.txt"
	# doc_file = "test.txt"
	with open(doc_file, "r") as f:
		doc = f.readline().split()
		doc_class, prob = classify(doc, model)
		print("likely class for {}: {} with (log) prob {}".format(doc_file, doc_class, prob))

main()