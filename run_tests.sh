#!/bin/bash
cat AdaBoost.txt | tr "." "," >> "./resultados/AdaBoost.txt"
cat decisionTree.txt | tr "." "," >> "./resultados/decisionTree.txt"
cat knn.txt | tr "." "," >> "./resultados/knn.txt"
cat randomForest.txt | tr "." "," >> "./resultados/randomForest.txt"