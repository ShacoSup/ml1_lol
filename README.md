<img width="1496" alt="181875297-0143e21e-eced-4bea-ba5a-a66a11b944ba" src="https://user-images.githubusercontent.com/92352445/183236897-f3e1a5b3-9d82-4f67-aaef-ae71d58ab5fc.jpeg">

<div align="Right">
    Project Period: 2021.05.03-2021.06.23
    </br>
    </br>
    한창훈
    </br>
</div>

## Index

1. [Introduction](#1-introduction)

    1.1. [Subject](#11-subject)

    1.2. [Objective](#12-objective)

2. [Data](#2-data)

    2.1. [Table Specification](#21-table-specification)
    
    2.2. [Data Preprocessing](#22-data-preprocessing)

3. [Model](#3-model)

    3.1. [Model: Logistic Regression](#31-model-logistic-regression)
    
    3.2. [Model: KNN](#32-model-knn)
    
    3.3. [Model: Decision Tree](#33-model-decision-tree)
    
    3.4. [Model: SVM](#34-model-svm)

4. [Trial](#4-trial)

    4.1. [Trial: Logistic Regression](#41-trial-logistic-regression)
    
    4.2. [Trial: KNN](#42-trial-knn)
    
    4.3. [Trial: Decision Tree](#43-trial-decision-tree)
    
    4.4. [Trial: SVM](#44-trial-svm)

5. [Conclusion](#5-conclusion)

6. [Refernce](#refernce)

7. [Source](#source)

## 1. Introduction
### 1.1. Subject
predicting the winner of LCK 

### 1.2. Objective
By using various models 어쩌고 저쩌고
Design predict models and analyze results by using League of Legends game data. 

## 2. Data
I've crawled the data from this [site](https://gol.gg/tournament/tournament-stats/LCK%20Spring%202021/)

<p align="Center">
<img width="770" alt="스크린샷 2022-08-06 오후 5 25 59" src="https://user-images.githubusercontent.com/92352445/183241188-68a8c00e-7563-4a2f-b91d-0034474b8abd.png">
</p>

### 2.1. Table Specification

</br>
<div align="Center">
  
|Feature|Explaination|Feature|Explaination|
|:--------:|:-----------:|:-------:|:-------:|
|Team|Team Name|Total damage to Champion|Total damage to Champion|
|Player|Player Name|Physical Damage|Physical Damage|
|Role|Role|Magic Damage|Magic Damage|
|Kills|Kills|True Damage|True Damage|
|Deaths|Deaths|DPM|Damage per minute|
|Assists|Assists|DMG%|DMG%|
|KDA|(K+A)/D|K+A Per Minute|K+A Per Minute|
|Perfect Score|(K+A)*1.2|KP%|Percentage of team's kill that is yours|
|CS|Killed CS|Solo kills|Solo kills|
|CS in Team's Jungle|Killed CS in ally's jungle|Double kills|Double kills|
|CS in Enemy Jungle|Killed CS in enemy's jungle|Triple kills|Triple kills|
|CSM|CS per minute|Quadra kills|Quadra kills|
|Golds|Golds|Penta kills|Penta kills|
|GPM|Golds per minute|GD@15|Gold Difference at 15 minutes between you and the enemy also in your role|
|GOLD%|Percentage of team's gold that is yours|CSD@15|CS Difference at 15 minutes between you and the enemy also in your role|
|Vision Score|Vision Score|XPD@15|Exp Difference at 15 minutes between you and the enemy also in your role|
|Wards placed|Wards placed|LVLD@15|Level Difference at 15 minutes between you and the enemy also in your role|
|Wards destroyed|Wards destroyed|Damage dealt to turrets|Damage dealt to turrets|
|Control Wards Purchased|Control Wards Purchased|Total heal|Total heal|
|VSPM|Vision Score per minutes|Damage self mitigated|Damage self mitigated|
|WPM|Wards placed per minute|Time ccing others|Time ccing others|
|VWPM|Control Wards placed per minute|Total damage taken|Total damage taken|
|WCPM|Wards cleared per minute|Win|Win|![image](https://user-images.githubusercontent.com/92352445/183241109-d926c5a0-0070-481f-94ed-a550ff876a1a.png)

</div>
</br>

### 2.2. Data Preprocessing
- Used features are `Win`, `Team`, `Player`, `Role`, `Kills`, `Deaths`, `Assists`, `KDA`, `K+A Per Minute`, `KP%`, `Total damage taken`, `Total heal`, `Time ccing others`, `Damage self mitigated`, `Total damage to Champion`, `Damage dealt to turrets`, `DPM`, `Total_CS`, `CSM`, `Golds`, `GPM`, 'Vision Score', 'GD@15', 'CSD@15', 'XPD@15', 'LVLD@15'

- Made a new feature called `Total_CS` which combines 'CS in Team's Jungle", "CS in Enemy Jungle", "CS"
- Made a new feature called `winner`
- The observation in KDA which is `Perfect KDA` are replaced by this formula `KDA = (K+A)*1.2`
Scaling
PCA

## 3. Model
### 3.1. Model: Logistic Regression

### 3.2. Model: KNN

### 3.3. Model: Decision Tree

### 3.4. Model: SVM


## 4. Trial
### 4.1. Trial: Logistic Regression

### 4.2. Trial: KNN

### 4.3. Trial: Decision Tree

### 4.4. Trial: SVM

## 5. Conclusion

## Refernce
[1] https://medium.com/@bjmoon.korea/ai-x-%EB%94%A5%EB%9F%AC%EB%8B%9D-fianl-assignment-84e66d7e451d

[2] https://github.com/HojinHwang/FIFA-Online4

[3] https://hojjimin-statistic.tistory.com/10

[4] http://www.gameinsight.co.kr/news/articleView.html?idxno=16078

## Source
[1] Pic Source: https://github.com/ultralytics/yolov5

[2] 
