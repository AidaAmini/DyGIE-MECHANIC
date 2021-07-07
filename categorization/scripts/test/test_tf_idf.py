from categorization.scripts.tf_idf import scrub

test = scrub(["hi", "\u190c", "my", "name", "200", "is", "2TC", "Josh"])
assert test == ['hi', 'name', '2TC', 'Josh']
print("tf_idf test passed")