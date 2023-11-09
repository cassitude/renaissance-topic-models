# from importlib_metadata import metadata
from os import listdir
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

"""
Output list with textfile names, given filter conditions
"""

# Filter conditions
timeperiodStart = 1640
timeperiodEnd = 1660

# Filtering
metadataOriginal = pd.read_csv("metadata.csv")  # keep original copy
numericFilter = metadataOriginal['Date'].apply(lambda x: (x[0:4].isnumeric()))
metadata = metadataOriginal.loc[numericFilter]
metadata.loc[:, 'Date'] = metadata.Date.apply(lambda x: int(x[0:4]))
metadata = metadata.loc[(metadata.Date >= timeperiodStart)
                        & (metadata.Date <= timeperiodEnd)]
textfileNames = metadata.TCP.apply(lambda x: str(x) + ".headed.txt").tolist()

print("Number of documents after filter: " + str(len(textfileNames)))

# Useful knowledge:
# max and min of date range: 1818 - 1473
