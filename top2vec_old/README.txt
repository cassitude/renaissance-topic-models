Assessing Political Ontologies with Top2Vec Topic Modeling

The repository is an in-depth use of the Top2Vec package. To make its use easiest, there are a few different scripts
with the purpose of pre-processing the data, adjusting hyperparamters of the model itself, and in depth summarizations. 
The descriptions of each file are below, in order of use:

specify_texts.py: end result is to generate list of textfile names (with TCP identification) after filter conditions of metadata
prepare_texts.py (CLEAN UP CONSISTENCY): optional script if lemmatized versions of texts desired, end result is to take in actual textfiles and export
    lemmatized version. 
generate_model.py: end result is to generate model. Imports textfile names from specify_texts.py to know which files to use, 
    and specify parameters with variables declared at top
examine.py: end result is to generate text file with results. Can declare desired queries at top
through_time.ipynb: Jupyter Notebook to output plot of a certain topic through through time 
surroundings.ipynb: More interactive notebook file to demonstrate what the surroudings are of various objects, mainly documents for now
topwords.py: script that outputs only top closest words to a given word.