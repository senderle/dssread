import glob
from collections import Counter
from collections import defaultdict
import re
import string
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

repo = ('/Users/rbenefo/Library/Mobile Documents/'
 		'com~apple~CloudDocs/Desktop/reg/for-topic-modeling/comments/')
datafiles = sorted(glob.glob(''.join([repo, 'VA-2016-VHA-0011-*.txt'])))
not_txt = '[^{}]'.format(string.ascii_lowercase)

def filenamereader(file):
	with open(file, 'r' ) as fileinput:
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

def topnumbertuples_to_list(topnumber):
	mostcommontuples = collapse_counters(files_to_counters(datafiles[0:10])).most_common(topnumber)
	most_commonlist = []
	for i in mostcommontuples:
		a, b = i
		most_commonlist.append(a)
	return most_commonlist

def listoflists(counterlist, most_commonlist):
	fullcountlist = []
	for i in counterlist:
		listi = list(i.items())
		for p in listi:
			a, b = p
			targetlist = [0]*len(most_commonlist)
			for q, j in enumerate(most_commonlist):
				if a == j:
					targetlist[q] = b
				else:
					targetlist[q] = 0
		fullcountlist.append(targetlist)
	return fullcountlist

def kmeanzer(fullcountlist):
	X = np.array(fullcountlist)
	estimator = [('test1', KMeans(n_clusters=100))]
	for name, est in estimator:
		est.fit(X)
		print(est.labels_)
		est.predict([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	return est

def defaulter(fullcountlist):
	clusters = defaultdict(list)
	cluster_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	print(fullcountlist)
	for p, c in zip(fullcountlist, cluster_labels):
		clusters[c].append(p)
	return clusters.items()
# print(kmeanzer(listoflists(files_to_counters(datafiles),topnumbertuples_to_list(10))))
print(defaulter(collapse_counters(files_to_counters(datafiles[0:10]))))

# print(listoflists(files_to_counters(datafiles),topnumbertuples_to_list(10)))



