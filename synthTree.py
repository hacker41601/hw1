import math
import random
import csv
import pandas as pd
import numpy as np
from pprint import pprint
import pprint
import matplotlib.pyplot as plt
import numpy as np

synth1 = pd.read_csv('synthetic-1.csv', header = None)
synth1.rename(columns = {0: 'col1', 1: 'col2', 2: 'ans'}, inplace = True)
#synth1.to_csv('testCol1.csv', index = False)
#synth1 = pd.read_csv('testCol1.csv', names=['col1', 'col2', 'ans'])
synth2 = pd.read_csv('synthetic-2.csv', header = None)
synth2.rename(columns = {0: 'col1', 1: 'col2', 2: 'ans'}, inplace = True)

synth3 = pd.read_csv('synthetic-3.csv', header = None)
synth3.rename(columns = {0: 'col1', 1: 'col2', 2: 'ans'}, inplace = True)

synth4 = pd.read_csv('synthetic-4.csv', header = None)
synth4.rename(columns = {0: 'col1', 1: 'col2', 2: 'ans'}, inplace = True)

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
        child = np.sum([(count[i]/np.sum(count))*calculate_entropy(data.where(data[split]==val[i]).dropna()[target])])
        #weighted entropy is the sum of the corresponding count to values divided by the summation of the counts
        #multiplied by the entropy where data is split
        #within the range of the number of values corresponding to the counts of the split attributes
    IG = parent - child
    return IG

def ID3(data,trainData,features,target = "ans",rootNode = None):
    #create root node for decTree
    #if ex are pos, return single node root with label = +
    #if ex are neg, return single node root with label = -
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    elif len(data)==0:
        return np.unique(trainData[target])[np.argmax(np.unique(trainData[target],return_counts=True)[1])]
    #if number of pred attr is NULL then ret single node root with label = most common val of target attr in ex
    elif len(features) ==0:
        return rootNode
    #else: begin
    else:
        rootNode = np.unique(data[target])[np.argmax(np.unique(data[target],return_counts=True)[1])]
        featVal = [calculate_information_gain(data,feature,target) for feature in features]
        #print(featVal)
        bestFeatVal_index = np.argmax(featVal)
        #print(bestFeatVal_index)
        bestFeatVal = features[bestFeatVal_index]
        #print(bestFeatVal)
        decTree = {bestFeatVal:{}}
        #this part takes out the best featval before recursing so it doesnt build the same exact tree
        features = [i for i in features if i != bestFeatVal]
        #A is the attribute that best classifies examples and = decdecTree attr for root
        #for each val of A:
                #add new branch under root corresponding to test A = val
                #let ex of val be subset of ex that have val for A
        for val in np.unique(data[bestFeatVal]):
            val = val
            #if ex of val is NULL
                    #add leaf to this branch with label = most common target val in the ex
            #else:
                    #below this branch add subdecTree ID3(ex(v), target, attr-{A})
            branch = data.where(data[bestFeatVal] == val).dropna()
            subdecTree = ID3(branch,trainData,features,target,rootNode)
            decTree[bestFeatVal][val] = subdecTree
        #end
        #return Root
        return(decTree)
        
def test(data, decTree, target):
    def predict(query,decTree, default = 1):
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
                        return res
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html
    #https://www.geeksforgeeks.org/python-pandas-dataframe-to_dict/
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["pred"])
    for i in range(len(data)):
        predicted.loc[i,"pred"] = predict(queries[i],decTree,1.0)
    pred = np.sum(predicted["pred"] == data[target])/len(data)
    print('Accuracy: ',pred*100,'%')

synth = pd.read_csv('synthetic-1.csv', header = None)
#print(synth)
for i in range(len(synth)):
    
    max0 = synth.iloc[:,0].max()
    max1 = synth.iloc[:,1].max()
    min0 = synth.iloc[:,0].min()
    min1 = synth.iloc[:,1].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval1 = np.linspace(min1, max1, 5)
    
    col0 = synth.iloc[:,0].values
    col1 = synth.iloc[:,1].values
    
    binned0 = np.digitize(col0, interval0)
    binned1 = np.digitize(col1, interval1)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = synth.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('SYNTH_BINNED.csv'.format(i))
    #data = {
    binned = pd.DataFrame({'col1': binned0.tolist(), 'col2': binned1.tolist(), 'ans': synth.iloc[:,2]})
#print(binned)
    
print('SYNTH - 1: ')
decTree1 = ID3(binned,binned,binned.columns[:-1])
test(binned,decTree1, "ans")
pp = pprint.PrettyPrinter(compact=True)
pp.pprint(decTree1)
print('###################################################################s#')
#print(type(binned))
#print(synth1)
#print(synth2)
#print(synth3)
#print(synth4)
#print(type(synth1))
synth = pd.read_csv('synthetic-2.csv', header = None)
#print(synth)
for i in range(len(synth)):
    
    max0 = synth.iloc[:,0].max()
    max1 = synth.iloc[:,1].max()
    min0 = synth.iloc[:,0].min()
    min1 = synth.iloc[:,1].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval1 = np.linspace(min1, max1, 5)
    
    col0 = synth.iloc[:,0].values
    col1 = synth.iloc[:,1].values
    
    binned0 = np.digitize(col0, interval0)
    binned1 = np.digitize(col1, interval1)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = synth.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('SYNTH_BINNED.csv'.format(i))
    #data = {
    binned = pd.DataFrame({'col1': binned0.tolist(), 'col2': binned1.tolist(), 'ans': synth.iloc[:,2]})
#print(binned)
    
print('SYNTH - 2: ')
decTree2 = ID3(binned,binned,binned.columns[:-1])
test(binned,decTree2, "ans")
pp = pprint.PrettyPrinter(compact=True)
pp.pprint(decTree2)
print('####################################################################')

#decTree1 = ID3(synth1,binned,synth1.columns[:-1])
#test(synth1,decTree1, "ans")
#decTree1 = ID3(binned,synth1,binned.columns[:-1])
#test(binned,decTree1, "ans")
#decTree1 = ID3(binned,synth1,synth1.columns[:-1])
#test(binned,decTree1, "ans")

synth = pd.read_csv('synthetic-3.csv', header = None)
#print(synth)
for i in range(len(synth)):
    
    max0 = synth.iloc[:,0].max()
    max1 = synth.iloc[:,1].max()
    min0 = synth.iloc[:,0].min()
    min1 = synth.iloc[:,1].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval1 = np.linspace(min1, max1, 5)
    
    col0 = synth.iloc[:,0].values
    col1 = synth.iloc[:,1].values
    
    binned0 = np.digitize(col0, interval0)
    binned1 = np.digitize(col1, interval1)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = synth.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('SYNTH_BINNED.csv'.format(i))
    #data = {
    binned = pd.DataFrame({'col1': binned0.tolist(), 'col2': binned1.tolist(), 'ans': synth.iloc[:,2]})
#print(binned)
    
print('SYNTH - 3: ')
decTree3 = ID3(binned,binned,binned.columns[:-1])
test(binned,decTree3, "ans")
pp = pprint.PrettyPrinter(compact=True)
pp.pprint(decTree3)
print('####################################################################')

synth = pd.read_csv('synthetic-4.csv', header = None)
#print(synth)
for i in range(len(synth)):
    
    max0 = synth.iloc[:,0].max()
    max1 = synth.iloc[:,1].max()
    min0 = synth.iloc[:,0].min()
    min1 = synth.iloc[:,1].min()
    
    interval0 = np.linspace(min0, max0, 5)
    interval1 = np.linspace(min1, max1, 5)
    
    col0 = synth.iloc[:,0].values
    col1 = synth.iloc[:,1].values
    
    binned0 = np.digitize(col0, interval0)
    binned1 = np.digitize(col1, interval1)
    
    #binned = pd.DataFrame()
    #binned['col1'] = binned0.tolist()
    #binned['col2'] = binned1.tolist()
    #binned['ans'] = synth.iloc[:,2].to_list()
    #print(binned)
    #binned.to_csv('SYNTH_BINNED.csv'.format(i))
    #data = {
    binned = pd.DataFrame({'col1': binned0.tolist(), 'col2': binned1.tolist(), 'ans': synth.iloc[:,2]})
#print(binned)
    
print('SYNTH - 4: ')
decTree4 = ID3(binned,binned,binned.columns[:-1])
test(binned,decTree4, "ans")
pp = pprint.PrettyPrinter(compact=True)
pp.pprint(decTree4)
print('####################################################################')
