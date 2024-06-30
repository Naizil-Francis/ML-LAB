import csv

training_data = []
with open('ENJOYSPORT.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        training_data.append(row)
           
G = ['0'] * (len(training_data[0]) - 1)
S=  []
step=0
# Apply CEA
for example in training_data:
    if example[-1] == '1':
        for i in range(len(G)):
            if G[i] != example[i]:
                if G[i] == '0':
                    G[i] = example[i]
                else:
                    G[i] = '?'
            for h in S[:]:
                if any(h[j] != '?' and G[j] != h[j] for j in range(len(G))):
                    S.remove(h)

    if example[-1] == '0':
        for i in range(len(G)):
            if G[i] != example[i]:
                r=[]
                for z in range(0,6):
                    if z==i:
                        r.append(G[i])
                    else:
                        r.append('?')
                for z in range(0,6):
                    if r[i]!='?':
                        S.append(r)
                        break
    
    print('STEP',step,': ')
    print("Specific hypothesis:", G)
    print("General hypothesis:", S)
    step+=1
    print('\n')

print("Specific hypothesis:", G)
print("General hypothesis:", S)
