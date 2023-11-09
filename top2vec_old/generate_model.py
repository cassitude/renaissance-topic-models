import multiprocessing
from top2vec import Top2Vec
from os import listdir
import os
import time

# File selection
from specify_texts import textfileNames, timeperiodStart, timeperiodEnd

# Thread selection
pool = multiprocessing.Pool()
numAvailableThreads = multiprocessing.Pool()._processes
percentageThreadsToUse = 1  # 0-1, "percentage of CPU power to use"
numThreads = int(numAvailableThreads * percentageThreadsToUse)
print("Libraries loaded, number of threads to utilize:  ", numThreads)

# File importing
path = "../../Downloads/tcp_standard/alltexts/"
listOfTexts = list()
for i in range(0, len(textfileNames)):
    filePointer = open(path + textfileNames[i], "r")
    listOfTexts.append(filePointer.read())

numDocs = len(listOfTexts)
print("Documents loaded in, total number of documents:", numDocs)

# Model parameters and save name
trainingSpeed = "fast-learn"  # TODO: deep learn
modelSaveName = (
    "models/"
    + str(timeperiodStart)
    + "_"
    + str(timeperiodEnd)
    + "_top2vectrained_"
    + trainingSpeed
)

# Pre training confirmation
print("Path of files: " + str(path))
print("Time period of files: " + str(timeperiodStart) + "-" + str(timeperiodEnd))
print("Model save name: " + str(modelSaveName))
print("\n")

# Train model
if __name__ == "__main__":
    model = Top2Vec(documents=listOfTexts, speed=trainingSpeed, workers=numThreads)
    model.save(modelSaveName)
    print(trainingSpeed + " speed model trained and saved.")
