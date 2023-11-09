# from jmespath import search
from top2vec import Top2Vec

"""
Print nearest words to each of the <searchWords> items
"""

modelname = "1640_1660_top2vectrained_learn_unlemmatized"
searchWords = ["liberty", "free", "citizen", "subject", "reign", "rule", "sovereign", "God",
               "commonwealth", "govern", "government", "realm", "majesty", "highness", "Hobbes", "Shakespeare", "Cicero"]
model = Top2Vec.load(str("models/") + modelname)


outputFilename = "summaries/words_similarity_" + \
    str(searchWords[0:3]) + "_" + modelname + ".txt"

outputFile = open(outputFilename, "w")

for i in range(0, len(searchWords)):
    outputFile.write("Words similar to: " + searchWords[i] + "\n")
    word = searchWords[i]
    words, word_scores = model.similar_words(
        keywords=[word], keywords_neg=[], num_words=25)
    for word, word_score in zip(words, word_scores):
        outputFile.write(word + " " + str(word_score) + "\n")
    outputFile.write("\n")
