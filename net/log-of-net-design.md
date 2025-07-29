## test，调参日志
### MGCl
<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">1-shot</th>
      <th colspan="4">5-shot</th>
    </tr>
    <tr>
      <th>Fold 0</th><th>Fold 1</th><th>Fold 2</th><th>mean</th>
      <th>Fold 0</th><th>Fold 1</th><th>Fold 2</th><th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MGCL-office</td><td>30.37</td><td>50.57</td><td>43.01</td><td>41.32</td><td>33.64</td><td>54.84</td><td>52.34</td><td>46.94</td>
    </tr>
    <tr>
      <td>MGCL-ours-(1)</td><td>29.38</td><td>44.86</td><td>43.70</td><td>39.31</td><td>32.34</td><td>53.99</td><td>50.02</td><td>45.45</td>
    </tr>
    <tr>
      <td>MGCL-ours-(2)</td><td>30.39</td><td>46.98</td><td>41.93</td><td>39.77</td><td>33.60</td><td>55.10</td><td>51.42</td><td>46.71</td>
    </tr>
    <tr>
      <td>MGCL-ours-(3)</td><td>27.96</td><td>50.78</td><td>41.34</td><td>40.49</td><td>32.71</td><td>56.67</td><td>49.06</td><td>46.15</td>
    </tr>
  </tbody>
</table>


### FBC

####  A 最高维度产生用于区分前景背景的prior，引导第三层的前景背景区分

| Method-FBC  | Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  FBC-(1)    |  26.00 |  48.20 | 43.97  |  39.39 |
|  FBC-(2)    |  29.61 |  47.49 | 43.32  |  40.14 |
|  FBC-(3)    |  31.00 |  49.40 | 41.98  |  40.79 |
|  FBC-(4)    |  30.49 |  47.42 | 44.34  |  40.75 |

#### B 每层的特征用于每层的prior生成,其余相同,alpha也相同
| Method-**a.** FBC_1| Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  ~~FBC_1~~  | ~~25.36~~ | ~~44.20~~  | ~~38.53~~  |  ~~36.03~~ |
|  ~~FBC_1~~  | ~~27.35~~ | ~~46.38~~  |    -       |      -     |

---
#### C 只用当前层信息生成prior，仅用到低维的两个层级特征上
| Method-**b.** FBC_1| Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  ~~FBC_1~~  | ~~20.24~~ | ~~44.03~~  | ~~33.17~~  |      -     |

#### D 加了temperate参数和两个dropout对应两个特征
| Method-**b.** FBC_1| Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  FBC_1      | 31.95  | 43.37  | 41.62  | 38.98  |

#### F 基于A ,将alpha系数改为基于self-gating获得
| Method-FBC_2| Fold 0 | Fold 1 | Fold 2 |  mean  |
|-------------|--------|--------|--------|--------|
|  ~~FBC_2-(1)~~  | ~~30.00~~  | ~~46.08~~  | ~~43.09~~  |  ~~39.72~~ |
|  ~~FBC_2-(2)~~  | ~~30.77~~  | ~~44.26~~  | ~~41.74~~  |  ~~38.92~~ |
|  ~~FBC_2-(3)~~  | ~~28.44~~  | ~~46.18~~  | ~~39.99~~  |  ~~38.20~~ |

### 1. MGA: mask guided attention   
### 2. ECM：Error Correction Module
### 3. FBC: cat(foreground - backgroud , fore , back)
### 4. Mamba:

1. FBC

选择第三层的特征基于mask average pooling 生成bg和fg的prior，bg-fg得到先验 加在了support和query的第三层特征上

2. FBC_1:

- ~~a. 每层的特征用于每层的prior生成,其余相同,alpha也相同~~
- ~~b. 只用当前层信息生成prior，仅用到低维的两个层级特征上~~


3. FBC_2:

- 基于FBC,将alpha系数改为基于self-gating获得
- 
### 4. HGG：Hyper Graph Guided


---
## Notes:

**zip**

query_feats = [A1, A2, A3]
_query_feats = [B1, B2, B3]
zip(query_feats, _query_feats) => [(A1, B1), (A2, B2), (A3, B3)]

