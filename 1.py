import csv

# Load data from CSV file
training_data = []
step=0
with open('ENJOYSPORT.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training_data.append(row)

# Initialize hypothesis with '?'
hypothesis = ['0'] * (len(training_data[0]) - 1)

# Apply Find-S algorithm
for example in training_data:
    if example[-1] == '1':
        for i in range(len(hypothesis)):
            if hypothesis[i] != example[i]:
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                else:
                    hypothesis[i] = '?'
    print('STEP',step,': ')
    print("Specific hypothesis:", hypothesis)
    step+=1
    print('\n')

print("Final hypothesis:", hypothesis)
