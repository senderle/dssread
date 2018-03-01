import glob
from collections import Counter
from collections import defaultdict
from editdistance import eval as levdist
import re
import os
import sys
import string
import random
import numpy as np
from sklearn.cluster import KMeans

#select most common words; return
#list; every row is file.
#at the end, you want a function that takes a list of counters, and a list of words. The output will be a row of the number of times in each
#counter the word appears. Row 1 of this list will be the number of times the list of words appears in counter 1. Row 2 is number of times the list of words appears
# in counter 2. Column 1 is the number of times word 1 appears in each counter.
#don't use numpy; use python. Much faster. Create a list of lists

#learn how to use sklearn kmeans

#use fit; X is the list of lists.

not_txt = '[^{}]'.format(string.ascii_lowercase)

def filenamereader(file):
    with open(file, 'r') as fileinput:
        return fileinput.read()

def to_ct(doc):
    doc = re.sub(not_txt, ' ', doc.lower())
    words = doc.split()
    words = [w for w in words if len(w) > 1 or w == 'i' or w == 'a']
    return Counter(words)

def files_to_counters(files):
    counterlist = []
    for filename in files:
        doc = filenamereader(filename)
        counterlist.append(to_ct(doc))
    return counterlist

def collapse_counters(counters):
    collapsed = Counter()
    for i in counters:
        collapsed.update(i)
    return collapsed

def wordlist(counters, n):
    most_common = collapse_counters(counters).most_common(n)
    return [w for w, ct in most_common]

def to_features(counters, words):
    features = [[c[w] for w in words] for c in counters]
    return np.array(features)

def kmeans_cluster(features, n_clusters):
    km = KMeans(n_clusters=n_clusters)
    km.fit(features)
    return km

def test_pair(f1, f2):
    with open(f1, encoding='utf-8') as ip:
        f1 = ip.read()

    with open(f2, encoding='utf=8') as ip:
        f2 = ip.read()
    return levdist(f1, f2) / max(len(f1), len(f2))


if __name__ == '__main__':
    repo = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else './deduped'

    n_features = 100
    n_clusters = 100
    n_docs = 30000
    # n_docs = 30000

    datafiles = sorted(glob.glob(os.path.join(repo, 'VA-2016-VHA-0011-*.txt')))
    random.shuffle(datafiles)
    datafiles = datafiles[0:n_docs]
    counters = files_to_counters(datafiles)

    words = wordlist(counters, n_features)
    features = to_features(counters, words)
    km = kmeans_cluster(features, n_clusters)
    labels = km.labels_
    transformed = km.transform(features)

    for cl in range(n_clusters):
        dist = transformed[:, cl]
        cluster_bool_ix = labels == cl
        cluster_ix = cluster_bool_ix.nonzero()[0]

        cluster_dist = dist[cluster_ix]
        closest = [datafiles[i] for i in
                   cluster_ix[cluster_dist.argsort()][:3]]

        n_total = cluster_bool_ix.sum()
        n_close = (cluster_dist < 1).sum()
        std = cluster_dist.std()
        print('---------------------------------------------')
        print('---------------------------------------------')
        print('---------------------------------------------')
        print("Cluster {}".format(cl))
        print()
        print(n_total, n_close, n_close / n_total, std)
        if len(closest) > 1:
            print(test_pair(closest[0], closest[1]))
        for f in closest:
            print()
            print('---------------------------------------------')
            with open(f, encoding='utf-8') as op:
                print(op.read())

        # if test_pair(closest[0], closest[1]) > 0.5:
        #     clusterfiles = [datafiles[i] for i in cluster_ix]
        #     for c in clusterfiles:
        #         in_base, in_name = os.path.split(c)
        #         out_full_path = os.path.join(out_folder, in_name)
        #         with open(c, encoding='utf-8') as ip:
        #             text = ip.read()
        #         with open(out_full_path, 'w', encoding='utf-8') as op:
        #             op.write(text)
        print()
        print()
