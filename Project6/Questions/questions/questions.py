import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    D = {}
    directoryList = os.listdir(os.getcwd())
    if directory in directoryList:
        os.chdir(directory)
        currentDir = os.getcwd()
        fileList = os.listdir(currentDir)
        for fileName in fileList:
            with open(os.path.join(currentDir, fileName), "r", encoding="utf8") as f:
                D[fileName] = f.read()
        os.chdir("../")
        return D
    else:
        return None
            
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stopwordsList = list(string.punctuation) + list(nltk.corpus.stopwords.words("english"))
    wordsList = [word.lower() for word in nltk.tokenize.word_tokenize(document) if word not in stopwordsList]
    return wordsList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    length = len(documents)
    D = {}
    # Create a Dictionary with keys are words that appear at least once in the documents
    for name , wordsList in documents.items():
        for word in wordsList:
            if word not in D:
                D[word] = 0
            else:
                continue
    #Update value (inverse document frequency) for each word in D
    for word in D.keys():
        numOfDict = 0
        for name, wordsList in documents.items():
            if word in wordsList:
                numOfDict +=1
            else:
                continue
        D[word] = math.log(length/(numOfDict)) # +1 for smoothing
    return D
    


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    D = {fileName : 0 for fileName in files}
    # for word in query:
    #     for name, wordsList in files.items():
    #         count = wordsList.count(word)
    #         tf_idf = count * idfs[word]
    #         D[name] += tf_idf
    for name , wordsList in files.items():
        for word in query:
            count = wordsList.count(word)
            tf_idf = count * idfs[word]
            D[name] += tf_idf

    sortedD = sorted(D.items(), key= lambda x : x[1], reverse=True)
    topn = list(i for (i,j) in sortedD)[:n]
    return topn

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Calculate the query term density for each sentence in sentences
    frequency = {sentence : 0 for sentence in sentences}
    for sentence, wordsList in sentences.items():
        length = len(wordsList)
        for word in wordsList:
            if word in query:
                frequency[sentence] += 1 / length
    
    D = {sentence : [0, frequency[sentence]] for sentence in sentences}

    #Option 1 : third test is wrong
    for word in query:
        for sentence, wordsList in sentences.items():
            if word in wordsList:
                D[sentence][0] += idfs[word]
            else:
                continue
    #Option 2: second test is wrong
    # for sentence, wordsList in sentences.items():
    #     for word in wordsList:
    #         if word in query:
    #             D[sentence][0] += idfs[word]
    #         else:
    #             continue
    
    sortedD = sorted(D.items() , key= lambda x : (x[1][0], x[1][1]), reverse=True)
    topn = list(i for (i,j) in sortedD)[:n]
    print(sortedD[:3])
    return topn
    

    

if __name__ == "__main__":
    main()
