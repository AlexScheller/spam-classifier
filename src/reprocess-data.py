import os

def process_file(file_name):
	with open(file_name, "r+") as f:
		tokens = f.readline().split()
		# test_string = "e mail x x x hello there heidshfduehfriehfuehfoeheufjeoehfie e mail s d d d d my man e mail"
		# tokens = test_string.split()
		new_tokens = []
		i = 0
		while i < len(tokens):
			if tokens[i] == 'e':
				if (i + 1) < len(tokens) and tokens[i + 1] == "mail":
					new_tokens.append("email")
					i += 1
			else:
				l = len(tokens[i])
				if (l < 15 and l > 1) or tokens[i] == 'x':
					new_tokens.append(tokens[i])
			i += 1
		# clear file for writing
		f.seek(0)
		f.truncate()
		# print(" ".join(new_tokens))
		f.write(" ".join(new_tokens))

def process_files():

	spam_train = "../data/spam-train/"
	spam_test = "../data/spam-test/"
	nonspam_train = "../data/nonspam-train/"
	nonspam_test = "../data/nonspam-test/"

	paths = [spam_train, spam_test, nonspam_train, nonspam_test]

	for path in paths:
		abs_path = os.path.abspath(path)
		for entry in os.scandir(abs_path):
			if entry.is_file():
				process_file(entry.path)

# process_file("test-process.txt")
process_files()