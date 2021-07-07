import randomcats

randomcats.LIST_LENGTH = 5
count = randomcats.main("mock.txt")

assert count == 9
print(count)