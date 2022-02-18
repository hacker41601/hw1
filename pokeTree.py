import math
#import random
#import csv
import pandas as pd
import numpy as np
from pprint import pprint
import pprint
#import matplotlib.pyplot as plt

poke = pd.read_csv('pokemonStats.csv')
legend = pd.read_csv('pokemonLegendary.csv')
#converting the legendary csv to a list to append to new csv file where they're combined
mayhapsLegendary = legend['Legendary'].tolist()
#print('Legendary:', mayhapsLegendary)
poke["Legendary"] = mayhapsLegendary
poke.to_csv("testPKMN.csv", index = False)
pokemon = pd.read_csv('testPKMN.csv', names=['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Type 1_Bug','Type 1_Dark','Type 1_Dragon','Type 1_Electric','Type 1_Fairy','Type 1_Fighting','Type 1_Fire','Type 1_Flying','Type 1_Ghost','Type 1_Grass','Type 1_Ground','Type 1_Ice','Type 1_Normal','Type 1_Poison','Type 1_Psychic','Type 1_Rock','Type 1_Steel','Type 1_Water','Type 2_Bug','Type 2_Dark','Type 2_Dragon','Type 2_Electric','Type 2_Fairy','Type 2_Fighting','Type 2_Fire','Type 2_Flying','Type 2_Ghost','Type 2_Grass','Type 2_Ground','Type 2_Ice','Type 2_Normal','Type 2_Poison','Type 2_Psychic','Type 2_Rock','Type 2_Steel','Type 2_Water', 'Legendary'])

pokemon = pd.read_csv('testPKMN.csv')
#print(pokemon)
#print(pokemon)
#this is for the first 7 since after that when it comes to generation, it gets all messed up
for i in range(len(pokemon)):
    
    max0 = pokemon.iloc[:,0].max()
    max1 = pokemon.iloc[:,1].max()
    min0 = pokemon.iloc[:,0].min()
    min1 = pokemon.iloc[:,1].min()
    #print(max0, max1)
    #print(min0, min1)
    max2 = pokemon.iloc[:,2].max()
    max3 = pokemon.iloc[:,3].max()
    min2 = pokemon.iloc[:,2].min()
    min3 = pokemon.iloc[:,3].min()
    
    max4 = pokemon.iloc[:,4].max()
    max5 = pokemon.iloc[:,5].max()
    min4 = pokemon.iloc[:,4].min()
    min5 = pokemon.iloc[:,5].min()
    
    max6 = pokemon.iloc[:,6].max()
    max7 = pokemon.iloc[:,7].max()
    min6 = pokemon.iloc[:,6].min()
    min7 = pokemon.iloc[:,7].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval1 = np.linspace(min1, max1, 5)
    
    interval2 = np.linspace(min2, max2, 5)
    interval3 = np.linspace(min3, max3, 5)
    
    interval4 = np.linspace(min4, max4, 5)
    interval5 = np.linspace(min5, max5, 5)
    
    interval6 = np.linspace(min6, max6, 5)
    interval7 = np.linspace(min7, max7, 5)
    
    col0 = pokemon.iloc[:,0].values
    col1 = pokemon.iloc[:,1].values
    
    col2 = pokemon.iloc[:,2].values
    col3 = pokemon.iloc[:,3].values
    
    col4 = pokemon.iloc[:,4].values
    col5 = pokemon.iloc[:,5].values
    
    col6 = pokemon.iloc[:,6].values
    col7 = pokemon.iloc[:,7].values
    
    binned0 = np.digitize(col0, interval0)
    binned1 = np.digitize(col1, interval1)
    
    binned2 = np.digitize(col2, interval2)
    binned3 = np.digitize(col3, interval3)
    
    binned4 = np.digitize(col4, interval4)
    binned5 = np.digitize(col5, interval5)
    
    binned6 = np.digitize(col6, interval6)
    binned7 = np.digitize(col7, interval7)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = pokemon.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('pokemon_BINNED.csv'.format(i))
    #data = {
    
    binned = pd.DataFrame({'col1': binned0.tolist(), 'col2': binned1.tolist(), 'col3': binned2.tolist(), 'col4': binned3.tolist(), 'col5': binned4.tolist(), 'col6': binned5.tolist(), 'col7': binned6.tolist(), 'col8': binned7.tolist(), 'Legendary': pokemon.iloc[:,44]})
#print(binned)
#using base 2 b/c we think in bits, could be modified to base 10 or e for checking with the internet for accuracy purposes
def calculate_entropy(column_name):
        entropy = 0
        value, count = np.unique(column_name, return_counts = True)
        probs = count/len(column_name)
        for prob in probs:
            entropy -= prob * math.log(prob,2)
        return entropy
        
#SHOULD NEVER RETURN NEGATIVE NUMBERS
def calculate_information_gain(data,split,target):
    #entropy is needed to calculate information gain and find the difference
    #value, count is reused from the entropy function but the column will be where the data splits
    parent = calculate_entropy(data[target])
    val,count= np.unique(data[split],return_counts=True)
    for i in range(len(val)):
        child = np.sum([(count[i]/np.sum(count))*calculate_entropy(data.where(data[split]==val[i])[target])])
        #weighted entropy is the sum of the corresponding count to values divided by the summation of the counts
        #multiplied by the entropy where data is split
        #within the range of the number of values corresponding to the counts of the split attributes
    IG = parent - child
    return IG
    
def ID3(data,trainData,features,target = "Legendary",rootNode = None):
    #create root node for decTree
    #if ex are pos, return single node root with label = +
    #if ex are neg, return single node root with label = -
    #base cases
    if len(np.unique(data[target])) <= 1:
        #print(len(np.unique(data[target])))
        #print(np.unique(data[target])[0])
        return np.unique(data[target])[0]
    #if number of pred attr is NULL then ret single node root with label = most common val of target attr in ex
    elif len(features) == 0:
        #print(len(features))
        return rootNode
    #else: begin
    else:
        rootNode = np.unique(data[target])[np.argmax(np.unique(data[target],return_counts=True)[1])]
        #print(rootNode)
        for feature in features:
            featVal = [calculate_information_gain(data,feature,target)]
        #print(featVal)
        bestFeat_index = np.argmax(featVal)
        #argmax takes the argument witht emax value
        #print(bestFeatVal_index)
        bestFeat = features[bestFeat_index]
        #print(bestFeatVal)
        decTree = {bestFeat:{}}
        #initialize decTree
        #print(decTree)
        #this part takes out the best featval before recursing so it doesnt build the same exact tree and get a weird run time errors
        #print(features)
        features = [i for i in features if i != bestFeat]
        #print(features)
        #A is the attribute that best classifies examples and = decdecTree attr for root
        #for each val of A:
                #add new branch under root corresponding to test A = val
                #let ex of val be subset of ex that have val for A
        for val in np.unique(data[bestFeat]):
            best_val = val
            #if ex of val is NULL
            #add leaf to this branch with label = most common target val in the ex
            if best_val == None:
                return
            #else:
                    #below this branch add subdecTree ID3(ex(v), target, attr-{A})
            else:
            #kept getting nans at this point had to include .dropna()
            #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
                branch = data.where(data[bestFeat] == best_val).dropna()
                #features at this point is the same as attr-{A}
                subdecTree = ID3(branch,trainData,features,target,rootNode)
                decTree[bestFeat][best_val] = subdecTree
        #end
        return(decTree)
#how to use a dictionary and its keys
#https://stackoverflow.com/questions/37583888/how-to-find-a-particular-key-in-list-of-dictionary-objects-in-python
#https://python-course.eu/machine-learning/regression-trees-in-python.php
#it isnt a regression tree but it's still a predict function that worked with mine!!
def test(data, decTree, target):
    def predict(query,decTree):
            for key in list(query.keys()):
                if key in list(decTree.keys()):
                    try:
                        res = decTree[key][query[key]]
                    except:
                        return default
                    result = decTree[key][query[key]]
                    if isinstance(res,dict):
                        return predict(query,res)
                    else:
                        #print(res)
                        return res
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
    #https://www.geeksforgeeks.org/python-pandas-dataframe-to_dict/
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    prediction = pd.DataFrame(columns=["ans"])
    for i in range(len(data)):
        prediction.loc[i,"ans"] = predict(queries[i],decTree)
        
    pred = np.sum(prediction["ans"] == data[target])/len(data)
    #print(pred)
    #print(prediction)
    print('Accuracy: ',pred*100,'%')

print(("\n"))
print(("                                      |\n"))
print(("                                     /|\n"))
print(("                                   ,' |\n"))
print(("                                  .   |\n"))
print(("                                    | |\n"))
print(("                                 ' '| |\n"))
print(("                                / / | |\n"))
print(("       _,.-\"\"--._              / /  | |\n"))
print(("     ,'          `.           j '   ' '\n"))
print(("   ,'              `.         ||   / ,                         ___..--,\n"))
print(("  /                  \\        ' `.'`.-.,-\".  .       _..---\"\"'' __, ,'\n"))
print((" /                    \\        \\` .\"`      `\"'\\   ,'\"_..--''\"\"\"'.'.'\n"))
print((".                      .      .'-'             \\,' ,'         ,','\n"))
print(("|                      |      ,`               ' .`         .' /\n"))
print(("|                      |     /          ,\"`.  ' `-. _____.-' .'\n"))
print(("'                      |..---.|,\".      | | .  .-'\"\"   __.,-'\n"))
print((" .                   ,'       ||,|      |.' |    |\"\"`'\"\n"))
print(("  `-._   `._.._____  |        || |      `._,'    |\n"))
print(("      `.   .       `\".     ,'\"| \"  `'           ,+.\n"))
print(("        \\  '         |    '   |   .....        .'  `.\n"))
print(("         .'          '     7  \".              ,'     \\\n"))
print(("                   ,'      |    `..        _,'      F\n"))
print(("                  .        |,      `'----''         |\n"))
print(("                  |      ,\"j  /                   | '\n"))
print(("                  `     |  | .                 | `,'\n"))
print(("                   .    |  `.|                 |/\n"))
print(("                    `-..'   ,'                .'\n"))
print(("                            | \\             ,''\n"))
print(("                            |  `,'.      _,' /\n"))
print(("                            |    | ^.  .'   /\n"))
print(("                             `-'.' j` V    /\n"))
print(("                                   |      /\n"))
print(("                                   |     /\n"))
print(("                                   |   ,'\n"))
print(("                                    `\"\"\n"))
print('GOTTA CATCH EM ALL: ')
#decTree = ID3(binned,binned,binned.columns[:-1])
#test(binned,decTree, "Legendary")

for i in range(len(pokemon)):
    
    max0 = pokemon.iloc[:,0].max()
    max4 = pokemon.iloc[:,4].max()
    min0 = pokemon.iloc[:,0].min()
    min4 = pokemon.iloc[:,4].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval4 = np.linspace(min4, max4, 5)
    
    col0 = pokemon.iloc[:,0].values
    col4 = pokemon.iloc[:,4].values
    
    binned0 = np.digitize(col0, interval0)
    binned4 = np.digitize(col4, interval4)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = pokemon.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('pokemon_BINNED.csv'.format(i))
    #data = {
    
    binned = pd.DataFrame({'total': binned0.tolist(), 'sp.attack': binned4.tolist(), 'Legendary': pokemon.iloc[:,44]})

#this is the one with max depth 3 bc there is only two columns
decTree = ID3(binned,binned,binned.columns[:-1])
test(binned,decTree, "Legendary")

#how to make decision surface - ish
#https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html

#def f(x, y):
#   return x*y

#np.linspace is to create numeric sequences
#linspace(start int, end int, number of samples)
#x = np.linspace(np.argmin(trainData1), np.argmax(trainData1), len(trainData1))
#y = np.linspace(np.argmin(trainData1), np.argmax(trainData1), len(trainData1))

#X, Y = np.meshgrid(x, y)
#Z = f(X, Y)
#plt.contour(X, Y, Z, colors='black')
#plt.contour(X, Y, Z, 20, cmap='RdGy')
#plt.legend(loc = "upper left")
#plt.colorbar();
#plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
#            cmap='RdGy')
#plt.colorbar()
#plt.show()
#this didnt work for me
'''def calculate_information_gain(data, col):
    total = 0
    bin1, bin2, bin3, bin4 = [], [], [], []
    attr = data.to_numpy()
    attrSIZE = len(attr)
    for i in range(len(col)):
        if col[i] == 1:
            bin1.append(data[i])
        elif col[i] == 2:
            bin2.append(data[i])
        elif col[i] == 3:
            bin3.append(data[i])
        elif col[i] == 4:
            bin4.append(data[i])
    b1len = len(bin1)
    b2len = len(bin2)
    b3len = len(bin3)
    b4len = len(bin4)
    parentEntropy = calculate_entropy(data)
    bin1ent = calculate_entropy(bin1)
    bin2ent = calculate_entropy(bin2)
    bin3ent = calculate_entropy(bin3)
    bin4ent = calculate_entropy(bin4)
    total += parentEntropy - ((b1len/attrSIZE) *bin1ent) + ((b2len/attrSIZE) *bin2ent) + ((b3len/attrSIZE) *bin3ent) + ((b4len/attrSIZE) *bin4ent)
    #print(total)
    return total
'''
