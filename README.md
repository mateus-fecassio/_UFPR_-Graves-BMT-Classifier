# GRAVES / BMT Classifier

## Extração de Características
Para a extração de características foi utilizado o histograma da extração LBP.


## Classificadores
Os classificadores utilizados foram: 
- RandomForestClassifier
- DecisionTreeClassifier
- AdaBoostClassifier
- KNeighborsClassifier 

A técnica de divisão dos dados utilizada foi LeaveOneOut.


## Execução do Experimento
- é necessário que o arquivo "Bancos" e o código piBiomedicas.py estejam no mesmo diretório.
- linha de comando : python3 piBiomedicas.py