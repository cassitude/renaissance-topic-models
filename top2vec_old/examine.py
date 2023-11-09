from top2vec import Top2Vec

"""
Take saved model, define query parameters, and output log
"""

# settings to tweak output
modelname = "1640_1660_top2vectrained_learn"
maxDocs = 2  # most documents to output
numWords = 10  # words to output to define topic
searchWord = "government"
# can search: for this topic number, which document is most similar to it?
topicNumToDoc = 1
keywords = ["god", "king", "peasant"]

# defaults
model = Top2Vec.load(str("models/") + modelname)
modsumfp = open('summaries/model_summary_' + modelname + '.txt', 'w')

numTopics = model.get_num_topics()

topicSizes = model.get_topic_sizes()
allTopics = model.get_topics()


modsumfp.write("Here is the breakdown of our " +
               modelname + " top2vec model.\n\n")

modsumfp.write("-------------------------------------------------\n")
modsumfp.write("Document and topic distribution\n")
modsumfp.write("-------------------------------------------------\n")
modsumfp.write("\n")

# modsumfp.write("Our latent space dimension is: " +
#                str(model.topic_vectors.shape[1]) + "\n")

modsumfp.write("Number of topics found from our input documents: " +
               str(model.get_num_topics()) + "\n\n")


modsumfp.write("Number of documents assigned to each topic: \n")

for i in range(0, numTopics):
    modsumfp.write("Topic " + str(i) + ": " + str(topicSizes[0][i]) + "\n")

modsumfp.write("\n")

for i in range(0, numTopics):
    modsumfp.write("For topic " + str(i) +
                   " the top " + str(numWords) + " words to define the topic (with the respective similarity to the topic centroid) are:\n")
    for j in range(0, numWords):
        modsumfp.write(allTopics[0][i][j] + " " +
                       str(allTopics[1][i][j]) + "\n")
    modsumfp.write("\n")


modsumfp.write("-------------------------------------------------\n")
modsumfp.write("Queries\n")
modsumfp.write("-------------------------------------------------\n")
modsumfp.write("\n")

modsumfp.write("Topics that are most similar to the word (" +
               searchWord + ") are: \n")


keywordSearch = model.search_topics(
    keywords=[searchWord], num_topics=numTopics)

for i in range(0, numTopics):
    modsumfp.write(
        "Topic " + str(keywordSearch[3][i]) + ": " + str(keywordSearch[2][i]) + " similarity\n")


modsumfp.write("\n\n")

modsumfp.write("Documents that are most similar to topic number (" +
               str(topicNumToDoc) + ") are:\n\n")

documents, document_scores, document_ids = model.search_documents_by_topic(
    topic_num=topicNumToDoc, num_docs=maxDocs)

for i in range(0, maxDocs):
    modsumfp.write(str(i+1) + ":" + " Document ID: " +
                   str(document_ids[i]) + " Similarity Score: " + str(document_scores[i]) + "\n\n")
    modsumfp.write("\"" + documents[i][0:500] + "...\"")
    modsumfp.write("\n\n")

modsumfp.write("\n\n")


modsumfp.write("Documents that are most similar to word (" +
               searchWord + ") are:\n\n")


documents, document_scores, document_ids = model.search_documents_by_keywords(
    keywords=[searchWord], num_docs=maxDocs)

for i in range(0, maxDocs):
    modsumfp.write(str(i+1) + ":" + " Document ID: " +
                   str(document_ids[i]) + " Similarity Score: " + str(document_scores[i]) + "\n\n")
    modsumfp.write("\"" + documents[i][0:500] + "...\"")
    modsumfp.write("\n\n")

modsumfp.write("\n")

modsumfp.write("Words that are most similar to word " +
               searchWord + " are:\n\n")


words, word_scores = model.similar_words(
    keywords=[searchWord], keywords_neg=[], num_words=10)


for word, score in zip(words, word_scores):
    modsumfp.write(f"{word} {score}\n")


modsumfp.write("\n\n")


# modsumfp.write(
#     "The topic most similar to document ( " + str(docNumber) + ") is: \n")
# modsumfp.write(model.get_documents_topics(doc_ids=334))

keywordsTopicQuery = model.search_topics(keywords, num_topics=3)
modsumfp.write(
    "Topics most similar to the keywords (" + str(keywords) + ") are: \n")
for i in range(0, 3):
    modsumfp.write(str(keywordsTopicQuery[3][i]) + " (" + str(keywordsTopicQuery[0]
                   [i][0:5]) + ")" + ": " + str(keywordsTopicQuery[2][i]) + "\n")

    # queryTopic[3][i] + " (" + queryTopic[0][i][0:5] + ")" + ": " + queryTopic[2][i]))

# other possibilities:
# -generate word clouds, collapse topics into smaller set, search for topic or document by a document/sentence, search using multiple
# -keywords, adjust parameters on UMAP dimension reduction and HDB scan for clusteirng
