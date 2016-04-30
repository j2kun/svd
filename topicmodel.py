'''
    A simple topic model using singular value decomposition
    applied to a corpus of cnn stories.
'''
import json
import numpy as np
from collections import Counter

from svd import svd


def makeDocumentTermMatrix(data):
    '''
        Return the document-term matrix for the given list of stories.
        stories is a list of dictionaries {string -> string|[string]}
        of the form
            {
                'filename': string
                'words': [string]
                'text': string
            }

        The list of words include repetition
    '''
    words = allWords(data)
    wordToIndex = dict((word, i) for i, word in enumerate(words))
    indexToWord = dict(enumerate(words))
    indexToDocument = dict(enumerate(data))

    matrix = np.zeros((len(words), len(data)))
    for docID, document in enumerate(data):
        docWords = Counter(document['words'])
        for word, count in docWords.items():
            matrix[wordToIndex[word], docID] = count

    return matrix, (indexToWord, indexToDocument)


def allWords(data):
    words = set()
    for entry in data:
        words |= set(entry['words'])
    return list(sorted(words))


def load():
    with open('all_stories.json', 'r') as infile:
        data = json.loads(infile.read())
    return data


if __name__ == "__main__":
    data = load()
    matrix, (indexToWord, indexToDocument) = makeDocumentTermMatrix(data)

