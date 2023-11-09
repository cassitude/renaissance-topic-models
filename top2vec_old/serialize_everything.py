from specify_texts import textfileNames
import brotli
import codecs
path = "../../../Downloads/tcp_standard/alltexts/"
listOfTexts = list()
for i in range(0, len(textfileNames)):
    filePointer = open(path + textfileNames[i], "r")
    unicode_string = filePointer.read()
    b_string = codecs.encode(unicode_string, 'utf-8')
    compressed = brotli.compress(b_string)
    with open('serialized_texts/'+textfileNames[i]+'.br', 'wb') as f:
        f.write(compressed)