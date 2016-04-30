import os

def loadRaw(directory):
    documents = dict()

    for filename in os.listdir(directory):
        if filename[-3:] == 'txt':
            with open(os.path.join(directory, filename), 'r') as infile:
                documents[filename] = infile.read()

    return documents


def loadCleaned():
    with open('tokenized.txt', 'r') as infile:
        lines = [line.split('|') for line in infile]

    return dict((int(storyId), words.split(',')) for (storyId, words) in lines)
