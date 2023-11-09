from concurrent.futures import process, thread
import multiprocessing
from re import M
from matplotlib.pyplot import text
import pandas as pd
import numpy as np
import spacy
import time
import os
from os import path
from specify_texts import textfileNames
from multiprocessing import Process
from multiprocessing import Lock

"""
Optional: preprocessing lemmatization of texts, top2vec really doesn't need. 
"""


def info(title):
    print("process id:", Process.pid)


def getRemainingTexts(textfileNames, outputPath):
    trimmedList = list()
    for file in textfileNames:
        if not path.exists(outputPath + file):
            trimmedList.append(file)
    return trimmedList


def lemmatizeText(filenameList, spacyModel, inputPath, outputPath):
    while True:
        print("Length of my list: " + str(len(filenameList)))
        if len(filenameList) == 0:
            print("Length of my file list is 0! Process: " + str(os.getpid()))
            return -1
        filename = filenameList.pop(0)
        if path.exists(outputPath + filename):
            print(filename, "already lemmatized! In process: ", os.getpid())
            continue
        print("To lemmatize: ", filename)
        txt = open(inputPath + filename, "r").read()
        if len(txt) > 5000:
            spacyModel.max_length = len(txt) + 100
            txtList = txt.split("\n\n")
            lemmatizedObject = spacyModel.pipe(txtList)
            contents = ""
            for text in lemmatizedObject:
                contents += " ".join([token.lemma_ for token in text])
        else:
            lemmatizedObject = spacyModel(txt)
            contents = " ".join([x.lemma_ for x in lemmatizedObject])

        outputFile = open(outputPath + "/" + filename, "w")
        outputFile.write(str(contents))
        print(
            "Wrote file: "
            + outputPath
            + filename
            + str(" from process " + str(os.getpid()))
        )


def generatePartitions(numProcesses, filenameList):
    partitions = list()
    partitionSize = int(len(filenameList) / numProcesses)
    for i in range(0, len(filenameList), partitionSize):
        partitions.append(filenameList[i : i + partitionSize])

    if len(partitions) > numProcesses:
        partitions[len(partitions) - 2] += partitions[len(partitions) - 1]
        del partitions[len(partitions) - 1]
    return partitions


if __name__ == "__main__":
    startTime = time.time()
    # python3 -m spacy download en
    model = spacy.load("en_core_web_sm")
    inputPath = "../../Downloads/tcp_standard/alltexts/"
    outputPath = "alltexts_lemmatized"
    numProcesses = int((multiprocessing.Pool()._processes) * 0.75)
    numProcesses = 1

    remainingFiles = getRemainingTexts(textfileNames, outputPath)
    if len(remainingFiles) == 0:
        print("All files lemmatized!")
        exit()

    print(
        "From "
        + str(len(textfileNames))
        + " original texts, we still need to lemmatize: "
        + str(len(remainingFiles))
    )
    time.sleep(1)
    processes = list()
    filenamePartitions = generatePartitions(numProcesses, remainingFiles)
    for i in range(0, numProcesses):
        processes.append(
            Process(
                target=lemmatizeText,
                args=(filenamePartitions[i], model, inputPath, outputPath),
            )
        )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Program finished! Ran in: ", (time.time() - startTime))
