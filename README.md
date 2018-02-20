# spam-classifier

A naive-Bayes classifier derived from chapter 6 of "speech and language processing" by Daniel Jurafsky & James H. Martin. The chapter is included as the file "6.pdf" in the "background" folder.

## data

A .zip file containing the training dataset can be found [here](openclassroom.stanford.edu/MainFolder/courses/MachineLearning/exercises/ex6materials/ex6DataEmails.zip)

Note that there are a few issues with the data set as downloaded. Whatever method was used to preprocess the data ended up creating two tokens out of one, likely from the improper stripping of non alphanumeric characters. For instance, throughout the emails the string "e mail" could be found, as well as instances of possessive "s"s being separated e. g. "america s".

In order to remedy these issues the following steps have been taken: "e mail" has been coalesced into "email". All single letters (except for 'x') have been removed. All tokens longer than 15 (determined by greping around for large strings `'\S\{15,20\}'`) characters have been removed.