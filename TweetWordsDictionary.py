file = open('dictionary.txt', 'r')

input = file.read()

print(input)

transferDataset = open('transfer_learning_modified.csv', 'r')

transferDataset.seek(0)

counts = []

for line in transferDataset:
    words = line.split(' ')
    count = 0
    print(words)
    for word in words:
        if "@" not in word and "http" not in word and "$" not in word and "0" not in word:
            word = word.lower()
            word = word.replace('""",0\n', "")
            word = word.replace('""",1\n', "")
            word = word.replace('"""', "")
            word = word.replace('""', "")
            word = word.replace('"', "")
            word = word.replace('?', "")
            word = word.replace(',', "")
            word = word.replace('!', "")
            word = word.replace('.', "")
            word = word.replace(')', "")
            word = word.replace('(', "")
            word = word.replace('“', "")
            word = word.replace(';', "")
            word = word.replace(':', "")
            word = word.replace("'", "")
            word = word.replace('”', "")
            print(word)
            if word in input:
                count += 1
    counts.append(count)


print(f"counts: {counts}")

# delete the most common words and then see the results as well
