#
# Implement a function count_words() in Python
# that takes as input a string s and a number n,
# and returns the n most frequently-occuring words in s.
#
# The return value should be a list of tuples -
# the top n words paired with their respective counts
# [(<word>, <count>), (<word>, <count>), ...],
# sorted in descending count order.
#

def count_words(s, n):
    # compute raw word counts
    counts = {}
    for word in s.split():
        canonical = word.lower().strip()
        if counts.get(canonical):
           counts[canonical] += 1
        else:
           counts[canonical] = 1
    all = {}
    freqs = []

    # gather up the sorted words by frequency
    for word in sorted(counts, key=counts.get, reverse=True):
       freq = counts[word]
       all[freq] = all.get(freq,[])
       all[freq].append(word)
       all[freq].sort()
       freqs.append(freq)

    # iterate through word frequencies, building up our result list
    freqs = set(freqs)
    result = []
    for freq in sorted(freqs,reverse=True):
       words = all[freq]
       while len(words) and n > 0:
          result.append((words.pop(0), freq))
          n += -1

    return result

