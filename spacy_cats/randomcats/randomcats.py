import random

LIST_LENGTH = 1000


# generate list of n random numbers
def generate_random_list(n):
    random_list = random.sample(range(1, n + 1), LIST_LENGTH)
    random_list.sort()
    return random_list

def main(filename):
    file = open(filename, "r")

    count = 0
    prev_span = ""
    for line in file:
        tabs = line.split("\t")
        span = tabs[0]
        if span != prev_span:
            count += 1
            prev_span = span

    random_list = generate_random_list(count)
    index = 0 # index within random list

    results = ["" for i in range(LIST_LENGTH)]

    count = 0  # count of categories
    prev_span = ""

    file.seek(0) # start from beginning
    for line in file:
        tabs = line.split("\t")
        span = tabs[0]
        if span != prev_span:
            if index < len(random_list) and count == random_list[index]:
                index += 1

            count += 1
            prev_span = span
            

        if index < len(random_list) and count == random_list[index]:
            results[index] += line
    
    file.close()

    # write randomly to file
    ans = open("results.txt", "w")
    order = random.sample(range(0, LIST_LENGTH), LIST_LENGTH)
    for my_index in order:
        ans.write(results[my_index] + "\n")
    ans.close()

    return count




if __name__ == "__main__":
    count = main("../../../for_edan/cofie_log.tsv")
    print(count)
