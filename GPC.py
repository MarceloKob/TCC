#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import pandas as pd
import numpy as np
import statistics
import sklearn
from imblearn.under_sampling import NearMiss
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler


# In[2]:


#Treino e teste
def teste(X,y,n):
    #Essa parte faz o algoritmo rodar 10 vezes
    valor = 10
    accuracy = [None]*valor
    f1=[None]*valor
    asc_roc =[None]*valor

    for i in range(valor):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #Divide o dataset em teste e treino
        
        if n==1:
            X_train,y_train = RandomUnderSampler().fit_resample(X_train,y_train)
        elif n==2:
            X_train,y_train = RandomOverSampler().fit_resample(X_train,y_train)
        elif n==3:
            X_train,y_train = SMOTETomek().fit_resample(X_train,y_train)
        
        kernel = RBF(20.0)
        modelo = GaussianProcessClassifier(kernel=kernel) #Adiciona o modelo de classificador (linha se altera em cada algoritmo)
        modeloTreinado = modelo.fit(X_train,y_train) #Treina o algoritmo classificador
        predicao = modeloTreinado.predict(X_test)#Predição dos resultados
        tn,fp,fn,tp = (confusion_matrix(y_test,predicao,labels=[0,1]).ravel()) #Separa os resultados em Verdadeiro negativo, Verdadeiro positivo, Falso negativo e Falso positivo
        accuracy[i] = (tp+tn)/(tn+fp+fn+tp)#Métrica de validação Acurácia
        f1[i] = tp/(tp+1/2*(fp+fn))#Métrica de validação F1-Score
        asc_roc[i] = roc_auc_score(y_test,predicao)#Métrica de validação Curva ROC
    return accuracy, f1, asc_roc


# In[3]:


#Imprime os resultados
def resultado (accuracy, f1, asc_roc):
    print('Média acurácia: ',statistics.mean(accuracy))
    print('Média f1-Score: ',statistics.mean(f1))
    print('Média Curva ROC: ',statistics.mean(asc_roc), '\n')
    print('Desvio padrão acurácia: ',statistics.stdev(accuracy))
    print('Desvio padrão f1-Score: ',statistics.stdev(f1))
    print('Desvio padrão Curva ROC: ',statistics.stdev(asc_roc)) 


# In[4]:


albert = pd.read_csv("AE-FM.csv")
albert= albert.drop(columns = ['sex'])
X = albert.drop(columns=['y', 'ID'])
y = albert['y']


# In[5]:


#Resultados Original
accuracy, f1, asc_roc = teste(X,y,0)
resultado(accuracy, f1, asc_roc)


# In[6]:


#Resultados Undersampling
accuracy, f1, asc_roc = teste(X,y,1)
resultado(accuracy, f1, asc_roc)


# In[7]:


#Resultados Oversampling
accuracy, f1, asc_roc = teste(X,y,2)
resultado(accuracy, f1, asc_roc)


# In[8]:


#Resultados SMOTE-TOMEK
accuracy, f1, asc_roc = teste(X,y,3)
resultado(accuracy, f1, asc_roc)

