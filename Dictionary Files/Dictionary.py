# File details: This program was to capture specific words from the tweets in the test set
# that were correctly classified by the model.

file = open('TweetVocabsDictionariesTrue.csv', 'r')
content = file.read()

file.seek(0)  # Reset the file pointer to the beginning
dictionary = {}
for line in file:
    #print(line.strip())  # Remove leading/trailing whitespace
    words = line.split(' ')
    for word in words:
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
        if "@" not in word and "http" not in word and "$" not in word and "0" not in word:
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1

print(len(dictionary.keys()))
file2 = open("dictionaryTrueCases.txt", 'w')
file2.write(str(dictionary.keys()))

# print(dictionary["piantino"])