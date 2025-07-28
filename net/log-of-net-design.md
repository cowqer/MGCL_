## test日记

| Method      | Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|MGCL-office  |  30.37 |  50.57 |  43.01 |  41.32 |
|MGCL-ours-(1)|  29.38 |  54.86 |  43.70 |  39.31 |
|MGCL-ours-(2)|  30.38 |  46.98 |  41.93 |  39.76 |
|MGCL-ours-(3)|  29.38 |  50.78 |  41.33 |  40.02 |

| Method-FBC  | Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  FBC-(1)    |  26.00 |  48.20 | 43.97  |  39.39 |
|  FBC-(2)    |  29.61 |  47.49 | 43.97  |  40.14 |
|  FBC-(3)    |  26.00 |  49.40 | 43.32  |  40.79 |
|  FBC-(4)    |  - |  - | -  |  - |

| Method-FBC_1| Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  FBC_1      |   |   |   |   |
|  FBC_1      |   |   |   |   |
|  FBC_1      |   |   |   |   |


### 1. MGA: mask guided attention 
### 2. ECM：Error Correction Module
### 3. FBC: cat(foreground - backgroud , fore , back)
1. FBC

选择第三层的特征基于mask average pooling 生成bg和fg的prior，bg-fg得到先验 加在了support和query的第三层特征上

2. FBC_1:

每层的特征用于每层的prior生成 其余相同 alpha也相同

2. FBC_2:



### 4. HGG：Hyper Graph Guided


---
## Notes:

**zip**

query_feats = [A1, A2, A3]
_query_feats = [B1, B2, B3]
zip(query_feats, _query_feats) => [(A1, B1), (A2, B2), (A3, B3)]

