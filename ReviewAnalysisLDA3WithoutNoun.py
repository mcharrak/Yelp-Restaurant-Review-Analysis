import ReadData as rd
# import ReviewAnalysis as ra
import gzip
from collections import defaultdict
import pandas as pd
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()


reviewFile="dataset/reviewShuffled.json"
businessFile="dataset/business.json"
userFile="dataset/user.json"


def getPrediction(alpha,uB,iB,user,item,y_u,y_i,uMap,iMap):
    i=uMap[user] if user in uMap else -1
    j=iMap[item] if item in iMap else -1
    rating=alpha
    rating = np.inner(y_u[uMap[i]],y_i[iMap[j]]) + alpha  + (uB[i] if i in uB else 0)  + (iB[j] if j in iB else 0) 
    return rating  

def getPrediction2(alpha,uB,iB,i,j,y_u,y_i,uMap,iMap):
    rating = alpha  + (uB[i] if i in uB else 0)  + (iB[j] if j in iB else 0)
    if i in uMap and j in iMap:
        rating +=np.inner(y_u[uMap[i]],y_i[iMap[j]]) 
    return rating   

def getWordCleaner():
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    en_stop=set(en_stop)
    en_stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    p_stemmer = PorterStemmer()
    return tokenizer,en_stop,p_stemmer

def trainLDA(rData,latentFactor=50,numOfTokens=15000,iterations=40,savePath="dataset/lda2_"):
    # latentFactor=50
    tokenizer,en_stop,p_stemmer=getWordCleaner()
    texts=[]
    text=rData['text'].values
    nounList=[]
    print("Start lemmatizing")
    for d in text:    
        d = d.lower()
        tokens = tokenizer.tokenize(d)
        stopped_tokens = [i for i in tokens if not i in en_stop]
#         tagged_text = nltk.pos_tag(stopped_tokens)
#         for word, tag in tagged_text:
#             if tag in ["NN", "NNS"]:
#                 nounList.append(word)
#         stopped_tokens=nounList
#     stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        lemm_tokens = [lemm.lemmatize(i) for i in stopped_tokens if(len(i)>2)]
        texts.append(lemm_tokens)
    print("finished lemmatizing")
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(keep_n=numOfTokens)
    corpora.Dictionary.save(dictionary, savePath+str(latentFactor)+"_factor.dict")
    print("Dictionary Created and Saved to: "+savePath)
    corpus = [dictionary.doc2bow(t) for t in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=latentFactor, id2word = dictionary, passes=iterations)
    ldamodel.save(savePath+str(latentFactor)+"_factor.lda")
    print("LDA Model Saved")
    return ldamodel,dictionary

def initMaps(tData,vData):
    uTrainDict = defaultdict(lambda: defaultdict(int))
    iTrainDict = defaultdict(lambda: defaultdict(int))
    uValidDict = defaultdict(lambda: defaultdict(int))
    iValidDict = defaultdict(lambda: defaultdict(int))
    uMap = defaultdict(int)
    uCount=0
    iMap = defaultdict(int)
    iCount=0
    for i in tData:
        user, item, rating = i['user_id'], i['business_id'], i['stars']
        uTrainDict[user][item] = rating
        iTrainDict[item][user] = rating
        if user not in uMap:
            uMap[user]=uCount
            uCount+=1
        if item not in iMap:
            iMap[item]=iCount
            iCount+=1
    for i in vData:
        user, item, rating = i['user_id'], i['business_id'], i['stars']
        uValidDict[user][item] = rating
    return uTrainDict,iTrainDict,uMap,iMap,uValidDict


def learnItemTopics(rData,ldamodel,dictionary,iMap,latentFactor=50):
    tokenizer,en_stop,p_stemmer=getWordCleaner()
    y_i=np.zeros((len(iMap),latentFactor ))
    # print(y_i)
    testbCount=0
    for b in iMap:
        #     print(b)
        nounList=[]
        for d in rData[rData.business_id==b]['text'].values:
            d=d.lower()
            tokens = tokenizer.tokenize(d)
            stopped_tokens = [i for i in tokens if not i in en_stop]
#             tagged_text = nltk.pos_tag(stopped_tokens)
#             for word, tag in tagged_text:
#                 if tag in ["NN", "NNS"]:
#                     nounList.append(word)
#             stopped_tokens=nounList
            lemm_tokens = [lemm.lemmatize(i) for i in stopped_tokens if(len(i)>2)]
            bow = dictionary.doc2bow(lemm_tokens)
            topics=ldamodel.get_document_topics(bow)
            for topic,prob in topics:
                y_i[iMap[b]][topic]=prob
        testbCount+=1
        if(testbCount%1000==0):
            print(testbCount)
    return y_i

def train_LDA_LFM(lam,tData,vData,factor,trials,rData,ldamodel,dictionary):
    print("initializing")
    uTrainDict,iTrainDict,uMap,iMap,uValidDict=initMaps(tData,vData)
    print("initializing done ",len(uTrainDict),len(iTrainDict),len(uMap),len(iMap),len(uValidDict))
    uB = defaultdict(float)
    iB = defaultdict(float)
    latentFactor=factor
    y_u=np.random.normal(scale=1./latentFactor,size=(len(uTrainDict),latentFactor ))
    print("Learning Item Topics "," ",factor," ",trials)
    y_i=learnItemTopics(rData,ldamodel,dictionary,iMap,latentFactor=factor)
    print("Topic Learnt ",len(y_i)," ",factor," ",trials)
    alpha = 0
    totalTrials=trials
    for counter in range(totalTrials):
        alpha=0
        for i in uTrainDict:
            for j in uTrainDict[i]:
                alpha += uTrainDict[i][j] - uB[i] -iB[j] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
        alpha /= len(tData)
        print(alpha)
        for i in uTrainDict:
            uB[i] = 0
            for j in uTrainDict[i]:
                uB[i] += uTrainDict[i][j]  - alpha - iB[j] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
            uB[i] /= (lam + len(uTrainDict[i])) 
        for j in iTrainDict:
            iB[j] = 0
            for i in iTrainDict[j]:
                iB[j] += iTrainDict[j][i]  -alpha - uB[i] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
            iB[j] /= (lam + len(iTrainDict[j]))
    
        for i in uTrainDict:
            for lf in range(latentFactor):
                y_u[uMap[i]][lf] = 0
                for j in uTrainDict[i]:
                    y_u[uMap[i]][lf] += y_i[iMap[j]][lf]*(uTrainDict[i][j]  - alpha - iB[j]  +y_i[iMap[j]][lf]*y_i[iMap[j]][lf]-np.inner(y_u[uMap[i]],y_i[iMap[j]]) )
                    y_u[uMap[i]][lf]  /= (lam + y_i[iMap[j]][lf]*y_i[iMap[j]][lf])
    vMSE = 0
    for i in uValidDict:
        for j in uValidDict[i]:
#             vMSE += ((alpha  + (uB[i] if i in uB else 0)  + (iB[j] if j in iB else 0) - uValidDict[i][j]) **2)
            vMSE += ((getPrediction2(alpha,uB,iB,i,j,y_u,y_i,uMap,iMap) - uValidDict[i][j]) **2)
    vMSE /= len(vData)
    print (vMSE)
    return vMSE,alpha,uB,iB,uMap,iMap

def findLam_Fact(lam,tData,vData,factor,trials):
    uTrainDict,iTrainDict,uMap,iMap,uValidDict=initMaps(tData,vData)
    uB = defaultdict(float)
    iB = defaultdict(float)
    latentFactor=factor
    y_u=np.random.normal(scale=1./latentFactor,size=(len(uTrainDict),latentFactor ))
    y_i=np.random.normal(scale=1./latentFactor,size=(len(iTrainDict),latentFactor ))
    alpha = 0
    totalTrials=trials
    for counter in range(totalTrials):
        alpha=0
        for i in uTrainDict:
            for j in uTrainDict[i]:
                alpha += uTrainDict[i][j] - uB[i] -iB[j] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
        alpha /= len(tData)
        print(alpha)
        for i in uTrainDict:
            uB[i] = 0
            for j in uTrainDict[i]:
                uB[i] += uTrainDict[i][j]  - alpha - iB[j] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
            uB[i] /= (lam + len(uTrainDict[i])) 
        for j in iTrainDict:
            iB[j] = 0
            for i in iTrainDict[j]:
                iB[j] += iTrainDict[j][i]  -alpha - uB[i] - np.inner(y_u[uMap[i]],y_i[iMap[j]])
            iB[j] /= (lam + len(iTrainDict[j]))
    
        for i in uTrainDict:
            for lf in range(latentFactor):
                y_u[uMap[i]][lf] = 0
                for j in uTrainDict[i]:
                    y_u[uMap[i]][lf] += y_i[iMap[j]][lf]*(uTrainDict[i][j]  - alpha - iB[j]  +y_i[iMap[j]][lf]*y_i[iMap[j]][lf]-np.inner(y_u[uMap[i]],y_i[iMap[j]]) )
                    y_u[uMap[i]][lf]  /= (lam + y_i[iMap[j]][lf]*y_i[iMap[j]][lf])
    #     print(y_u)
        for j in iTrainDict:
            for lf in range(latentFactor):
                y_i[iMap[j]][lf] = 0
                for i in iTrainDict[j]:
                    y_i[iMap[j]][lf] += y_u[uMap[i]][lf]*(uTrainDict[i][j]  - alpha - uB[i] - np.inner(y_u[uMap[i]],y_i[iMap[j]]) +y_u[uMap[i]][lf]*y_u[uMap[i]][lf] )
                    y_i[iMap[j]][lf] /= (lam + y_u[uMap[i]][lf]*y_u[uMap[i]][lf])
    vMSE = 0
    for i in uValidDict:
        for j in uValidDict[i]:
#             vMSE += ((alpha  + (uB[i] if i in uB else 0)  + (iB[j] if j in iB else 0) - uValidDict[i][j]) **2)
            vMSE += ((getPrediction2(alpha,uB,iB,i,j,y_u,y_i,uMap,iMap) - uValidDict[i][j]) **2)
    vMSE /= len(vData)
    print (vMSE)
    return vMSE,alpha,uB,iB,uMap,iMap

print("Read Started")
bdata=rd.readData(fileName=businessFile,breakCondition=5000000)

categoryDict=defaultdict(int)
businessSet=set()
for b in bdata:
    bid,categoryList=b['business_id'],b['categories']
    if 'Restaurants' in categoryList:
        businessSet.add(bid)
print(len(businessSet))

rawdata=rd.readData(fileName=reviewFile,breakCondition=5000000)
data=[]
for d in rawdata:
    bid=d['business_id']
    if bid in businessSet:
        data.append(d)
print(len(data))
rawdata=[]


latentfactor=60
trials=40
trainSize=200000
tData=data[:trainSize]
vData=data[trainSize:trainSize+100000]
fileSavePath="dataset/lda2_"
rData=pd.DataFrame(tData)
uTrainDict,iTrainDict,uMap,iMap,uValidDict=initMaps(tData,vData)
# print(len(iMap))

# returns lda model and save the model in the path given by savePath
print("Starting LDA Training")
ldaModel,dictionary=trainLDA(rData,latentFactor=latentfactor,numOfTokens=15000,iterations=trials,savePath=fileSavePath)
print("Done LDA Training")

print(ldaModel.print_topics(num_topics=3, num_words=10))

tData=data[:trainSize]
vData=data[trainSize:]
lamdas=[4.5]
trials=[2]
factors=[latentfactor]
# factors=[1,2,4,6,8,10,15,20,25,30,35,40,45,50]
vMSE = np.iinfo(np.int32).max
bestLam=0.01
bestTrial=1
bestFactor=1
vMSEList=[]
trialList=[]
factorList=[]
lamdaList=[]
for i in lamdas:
    tempvMSE=1
    for t in trials:
        for f in factors:
            tempvMSE,alpha,uB,iB,uMap,iMap=train_LDA_LFM(lam,tData,vData,factor,trials,rData)
#             vMSEList.append(tempvMSE)
            print ("----------lamda: "+str(i)+"-----------Trails: "+str(t)+"-------------Factor: "+str(f)+" MSE: "+str(tempvMSE))
            if(tempvMSE<vMSE):
                vMSE=tempvMSE
                bestLam=i
                bestTrial=t
                bestFactor=f
                bestAlpha=alpha
                bestuB=uB
                bestiB=iB
            vMSEList.append(tempvMSE)
            factorList.append(f)
#     vMSEList.append(tempvMSE)
#     lamdaList.append(i)
        
# import matplotlib.pyplot as plt
# plt.scatter(factorList,vMSEList,color='red',marker='^')
# plt.xlabel('Factors')
# plt.ylabel('MSE')
# plt.show()

print("Best Value for Lamda is: ",bestLam," vMSE: ",vMSE)            
