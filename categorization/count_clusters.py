training = open("/data/edan/categorization/k_mean_clusters_training.tsv", "r")
predict = open("/data/edan/categorization/k_means_clusters_predict.tsv", "r")


def data(file): 
    tops = {}
    for line in file:
        tabs = line.split("\t")
        tabs[1] = tabs[1][:-1]
        if tabs[1] not in tops:
            tops[tabs[1]] = 1
        else:
            tops[tabs[1]] += 1
    return tops

print(data(training))
print(data(predict))
