import warnings
from skimage.feature import local_binary_pattern
import numpy as np
import os, glob
import os
import cv2
from sklearn.neighbors import KNeighborsClassifier
from pydicom import dcmread
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import sys
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd          
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
import time
from time import sleep

#######################################################################################
def Lbp(home, features):

  METHOD = 'uniform'
  radius = 3
  n_points = 8 * radius

  #percorre as pastas e subpastas pra achar as imagens
  for (pastaAtual, subpastas, arquivos) in os.walk(home, topdown=True):
    os.chdir(pastaAtual)
    
    #qtd de características, ATENÇÃO: mais que isso ta confundindo alguns classificadores
    #unique = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    unique = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    
    #verificação das LABELS
    if(os.getcwd() == home+'/GRAVES'):
      label = '1' #1 para casos GRAVES
    else:
      label = '2' #2 para casos BMT

    #vai abrir cada imagem
    for img in arquivos:

      #manipulação de imagens DICOM
      dicomImg = dcmread(img,  force=True)
      img = dicomImg.pixel_array
      
      #faz a média dos pixels da imagen usando LBP
      lbp_pred = local_binary_pattern(img, n_points, radius, METHOD)
      lbp_pred = lbp_pred.ravel()

      #criando histograma do LBP que então será usado para extrair a característica
      hist = np.histogram(lbp_pred, bins=unique)
      result = hist[0]/22500 #mudar de acordo com tamanho do patch!!!NÂO ESQUEÇA LINDONA
      result.ravel
      
      #escreve a label
      features.write(str(label)+ " ")
      indice = 0
      
      #escreve as características
      for x in result:
        features.write(str(indice)+":"+str(x)+" ")
        indice = indice +1

      features.write("\n")

  print("\nO arquivo de Características encontra-se no diretório atual\n")


#######################################################################################
def ABC(data):
  
  f1 = 0
  acuracia = 0
  precision = 0
  recall = 0

  #carrega os dados
  x_data, y_data = load_svmlight_file(data)
  
  #para fazer leave one patient out
  cv = LeaveOneOut()
  #separar em teste e treino com a instancia cv
  y_true, y_pred = list(), list()
  for train_ix, test_ix in cv.split(x_data):
    X_train, X_test = x_data[train_ix], x_data[test_ix]
    y_train, y_test = y_data[train_ix], y_data[test_ix]

    #classificação - n_estimators da ŕa variar eu acho, 100 foi bom
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
  
    #ta somando as métricas pra fazer a média
    f1 += f1_score(y_test, yhat,zero_division=0)
    acuracia += accuracy_score(y_test, yhat)
    precision += precision_score(y_test, yhat,zero_division=0)
    recall += recall_score(y_test, yhat,zero_division=0)

    y_true.append(y_test[0])
    y_pred.append(yhat[0])

    print("\n------------------------------------------------------")
    print('Matriz de confusão:\n',confusion_matrix(y_test, yhat))
    #obter resultado dos 12 testes
    print('\nF1', f1_score(y_test, yhat,zero_division=0))
    print('Acuracia', accuracy_score(y_test, yhat))
    print('precision', precision_score(y_test, yhat,zero_division=0))
    print('recall ', recall_score(y_test, yhat,zero_division=0))


  print("\n##################################################")
  print('Matriz de confusão final:\n',confusion_matrix(y_true, y_pred))
  print("\nA média dos resultados para Ada Boost Classifier são:\n")

  #chama metric pra fazer só a média
  metrics(f1,acuracia,precision,recall)


#######################################################################################
def KNN(data):
  
  f1 = 0
  acuracia = 0
  precision = 0
  recall = 0

  x_data, y_data = load_svmlight_file(data)
  
  cv = LeaveOneOut()
  y_true, y_pred = list(), list()
  for train_ix, test_ix in cv.split(x_data):
    X_train, X_test = x_data[train_ix], x_data[test_ix]
    y_train, y_test = y_data[train_ix], y_data[test_ix]
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
  
    #ta somando as métricas pra fazer a média
    f1 += f1_score(y_test, yhat,zero_division=0)
    acuracia += accuracy_score(y_test, yhat)
    precision += precision_score(y_test, yhat,zero_division=0)
    recall += recall_score(y_test, yhat,zero_division=0)

    y_true.append(y_test[0])
    y_pred.append(yhat[0])

    print("\n------------------------------------------------------")
    print('Matriz de confusão:\n',confusion_matrix(y_test, yhat))
    #obter resultado dos 12 testes
    print('\nF1', f1_score(y_test, yhat,zero_division=0))
    print('Acuracia', accuracy_score(y_test, yhat))
    print('precision', precision_score(y_test, yhat,zero_division=0))
    print('recall ', recall_score(y_test, yhat,zero_division=0))

  print("\n##################################################")
  print('Matriz de confusão final:\n',confusion_matrix(y_true, y_pred))
  print("\nA média dos resultados para K-Neighbors Classifier são:\n")

  #chama metric pra fazer só a média
  metrics(f1,acuracia,precision,recall)


#######################################################################################

def DTC(data):
  
  f1 = 0
  acuracia = 0
  precision = 0
  recall = 0

  x_data, y_data = load_svmlight_file(data)
  
  cv = LeaveOneOut()
  y_true, y_pred = list(), list()
  for train_ix, test_ix in cv.split(x_data):
    X_train, X_test = x_data[train_ix], x_data[test_ix]
    y_train, y_test = y_data[train_ix], y_data[test_ix]
    model = DecisionTreeClassifier(random_state=0,max_depth=None, min_samples_split=2)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
  
    #ta somando as métricas pra fazer a média
    f1 += f1_score(y_test, yhat,zero_division=0)
    acuracia += accuracy_score(y_test, yhat)
    precision += precision_score(y_test, yhat,zero_division=0)
    recall += recall_score(y_test, yhat,zero_division=0)

    y_true.append(y_test[0])
    y_pred.append(yhat[0])

    print("\n------------------------------------------------------")
    print('Matriz de confusão:\n',confusion_matrix(y_test, yhat))
    #obter resultado dos 12 testes
    print('\nF1', f1_score(y_test, yhat,zero_division=0))
    print('Acuracia', accuracy_score(y_test, yhat))
    print('precision', precision_score(y_test, yhat,zero_division=0))
    print('recall ', recall_score(y_test, yhat,zero_division=0))

  print("\n##################################################")
  print('Matriz de confusão final:\n',confusion_matrix(y_true, y_pred))
  print("\nA média dos resultados para Decision Tree Classifier são:\n")

  #chama metric pra fazer só a média
  metrics(f1,acuracia,precision,recall)

#######################################################################################
def RFC(data):

  x_data, y_data = load_svmlight_file(data)

  f1 = 0
  acuracia = 0
  precision = 0
  recall = 0

  cv = LeaveOneOut()
  y_true, y_pred = list(), list()
  for train_ix, test_ix in cv.split(x_data):
    X_train, X_test = x_data[train_ix], x_data[test_ix]
    y_train, y_test = y_data[train_ix], y_data[test_ix]
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
  
    #ta somando as métricas pra fazer a média
    f1 += f1_score(y_test, yhat,zero_division=0)
    acuracia += accuracy_score(y_test, yhat)
    precision += precision_score(y_test, yhat,zero_division=0)
    recall += recall_score(y_test, yhat,zero_division=0)

    y_true.append(y_test[0])
    y_pred.append(yhat[0])

    print("\n------------------------------------------------------")
    print('Matriz de confusão:\n',confusion_matrix(y_test, yhat))
    #obter resultado dos 12 testes
    print('\nF1', f1_score(y_test, yhat,zero_division=0))
    print('Acuracia', accuracy_score(y_test, yhat))
    print('precision', precision_score(y_test, yhat,zero_division=0))
    print('recall ', recall_score(y_test, yhat,zero_division=0))

  print("\n##################################################")
  print('Matriz de confusão final:\n',confusion_matrix(y_true, y_pred))
  print("\nA média dos resultados para Random Forest Classifier são:\n")

  #chama metric pra fazer só a média
  metrics(f1,acuracia,precision,recall)

#######################################################################################
def metrics(f1,acuracia,precision,recall):

  #f1 score
  print('F1Score:  ',f1/12)

  #acurácia
  print( 'Acurácia:  ',acuracia/12)

  #precisão
  print('Precisão:  ',precision/12)

  #recall
  print('Recall:  ',recall/12)
  

#######################################################################################
if __name__ == '__main__':

  #pega o diretório atual
  home1 = os.getcwd()
  home = home1+'/Bancos'

  #verifica se rodou alguma vez e já tem as característica
  if(os.path.isfile('features.txt')):
    print("O arquivo de Características encontra-se no diretório atual\n")
  else:
    features = open("features.txt", "a+")
    #extraindo característica com LBP - 
    Lbp(home,features)
    os.chdir(home1)

  
  print("Dentre as opções de classificadores temos: \n 1:RandomForestClassifier\n 2:DecisionTreeClassifier\n 3:AdaBoostClassifier\n 4:KNeighborsClassifier \n 5:Para sair\n")
  classifier = input("\nEscolha uma das opções (1 2 3 4 ou 5 para sair):")
  while (classifier != '5'):
    print('\n\nSerão apresentadas métricas e matriz de confusão para cada uma das 12 execuções, seguidas da média dos resultados!')
    sleep(5)
    if(classifier == '1'):
      inicio1 = time.time()
      RFC('features.txt')
      fim1 = time.time()
      print('Tempo de classificação: ', fim1-inicio1)
      print("\n##################################################")

    if(classifier == '2'):
      inicio2 = time.time()
      DTC('features.txt')
      fim2 = time.time()
      print('Tempo de classificação: ', fim2-inicio2)
      print("\n##################################################")

    if(classifier == '3'):
      inicio3 = time.time()
      ABC('features.txt')
      fim3 = time.time()
      print('Tempo de classificação: ', fim3-inicio3)
      print("\n##################################################")

    if(classifier == '4'):
      inicio4 = time.time()
      KNN('features.txt')
      fim4 = time.time()
      print('Tempo de classificação: ', fim4-inicio4)
      print("\n##################################################")

    classifier = input("Escolha uma das opções (1 2 3 4 ou 5 para sair):")
