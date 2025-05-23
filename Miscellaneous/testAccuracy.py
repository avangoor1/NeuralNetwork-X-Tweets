# This program was written to find the accuracy of the neural network when tested against a custom test set.

import csv

luck, meritocracy = 0, 0
with open("filteredTweets.csv", "r") as testFile:
    csv_reader = csv.reader(testFile)  # Read all rows
    for row in csv_reader:
        if float(row[4]) > 0.5:
            meritocracy += 1
        else:
            luck += 1

# accuracy = luck / (meritocracy + luck)
accuracy = meritocracy / (meritocracy + luck)
print("meritocratic tweets: ", meritocracy)
print("luck tweets", luck)
print("accuracy: ", accuracy)


