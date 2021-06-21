from kmeans_training import get_words

PATH = "test_kmeans.txt"
out = get_words(open(PATH, "r"))
print(out)