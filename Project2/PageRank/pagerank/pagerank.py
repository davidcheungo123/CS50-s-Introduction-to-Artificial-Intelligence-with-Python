import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    totalPage  = len(corpus)
    linkedPage = len(corpus[page])
    D = {}
    for ele in list(corpus.keys()):
        prob = (1- damping_factor) / totalPage
        if ele in corpus[page]:
            prob += damping_factor/linkedPage
        D[ele] = prob
    return D


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    totalPage = len(corpus)
    D = dict([(pageName , 1/totalPage) for pageName in corpus.keys()])
    counter = 0
    samplingList = []
    freqDistribution = {}
    while True:
        if counter == n:
            break
        page = random.choices(list(D.keys()), weights=list(D.values()), k=1)
        samplingList.append(page[0])
        D = transition_model(corpus, page[0], damping_factor)
        counter +=1
    for name in list(corpus.keys()):
        freqDistribution[name] = samplingList.count(name) / n
    return freqDistribution


def LinkToName(corpus, name):
    includedList = []
    for dummy, links in corpus.items():
        if name in links:
            includedList.append(dummy)
    return includedList


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    totalPage = len(corpus)
    D = dict([(pageName , 1/totalPage) for pageName in corpus.keys()])
    nameList = D.keys()
    while True:
        probList = list(D.values()).copy()
        for name in nameList:
            LinkToNameList = LinkToName(corpus, name)
            D[name] = (1-damping_factor)/totalPage
            for connection in LinkToNameList:
                D[name] += damping_factor*(D[connection]/len(corpus[connection]))
        nextList = D.values()
        minorList = [abs(i-j) for i,j in zip(probList, nextList)]
        if max(minorList) <= 0.001:
            break
    return D


if __name__ == "__main__":
    main()
