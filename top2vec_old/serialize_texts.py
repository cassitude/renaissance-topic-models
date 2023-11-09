from specify_texts import textfileNames
import pickle
import numpy as np
import pandas as pd
path = "../../../Downloads/tcp_standard/alltexts/"
listOfTexts = list()
for i in range(0, len(textfileNames)):
    filePointer = open(path + textfileNames[i], "r")
    listOfTexts.append(filePointer.read())

df = pd.DataFrame(listOfTexts)
df.to_parquet('serialized_texts.parquet.br',
              compression='brotli', index=False)

#with open('serialized_texts.pkl', 'wb') as f:
    #np.save(f, listOfTexts)
    #pickle.dump(listOfTexts, f)
