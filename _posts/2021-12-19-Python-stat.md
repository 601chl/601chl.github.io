---
layout: single
title:  "Python_stat_정리"
header:
      teaser: "/assets/images/python.png"
categories: python
tag: [python, stat, summarize]
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---





```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy as sp
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm  #일원배치분산분석
from statsmodels.stats.multicomp import pairwise_tukeyhsd  #사후검정
from sklearn.linear_model import LinearRegression  # 사이킷런에서 선형회귀 알고리즘
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures #다항식변환메소드
from sklearn.preprocessing import MinMaxScaler   #스케일러, 단위맞추기
from sklearn.preprocessing import StandardScaler
```

---------------------------

## 1. stat1

### 1) 통계수치 계산 문법
sum, mean, var, std, **stats.scoreatpercentile 분위수**, median, describe,     
공분산: sp.cov(cov_data.x, cov_data.y, ddof=1) ,    
상관계수: np.corrcoef(cov_data.x,cov_data.y)  == df.corr()

[통계량 선정시 평가기준]
- 불편성
- 효율성
- 일치성
- 충분성


<br>


```python
fish_data=np.array([2,3,3,4,4,4,4,5,5,6])
np.sum(fish_data)  #합계
fish_data.sum()
```




    40




```python
len(fish_data) #표본개수
#평균
N = len(fish_data)
sum_value = np.sum(fish_data)
mu = sum_value/N

print('평균:',mu)
np.mean(fish_data)  # == mu
```

    평균: 4.0
    

    4.0


```python
#표본분산
sigma_2_sample = np.sum((fish_data-mu)**2)/N  #분산: 편차제곱의 평균
np.var(fish_data, ddof=0) # == sigma_2_sample
```


```python
#표준편차
np.std(fish_data, ddof=0) #루트분산
```


```python
#불편분산, 편차 (분산이 과소평과되는 문제가 있어 보정된 표본통계량)
np.var(fish_data, ddof=1')
np.std(fish_data, ddof=1).round(2) # == np.round(np.std(fish_data, ddof=1),2)
```


```python
from scipy import stats
#분위수
stats.scoreatpercentile(fish_data,[25,75]) 
```


    array([3.25, 4.75])


```python
#중앙값
np.median(fish_data2)
```

그룹별 통계량 : mean(), std(), describe()      
교차분석표 : pivot_table(), crosstab()
       
공분산 : 2개의 연속형 변수의 관계성을 확인하는 통계량       
피어슨 상관계수 : 공분산을 최대값 1, 최소값 -1 사이가 되도록 표준화         
cov / sp.sqrt(sigma_2_x * sigma_2_y)


```python
fish_multi = pd.read_csv('dataset/5_2_fm.csv')
fish_multi.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#그룹별 통계량 계산
grouped = fish_multi.groupby('species')
grouped.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="8" halign="left">length</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>species</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>3.0</td>
      <td>3.5</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>3.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#공분산행렬
cov_data = pd.read_csv('dataset/4_cov.csv')
cov_data.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.5</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.7</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
#과제
#01.공분산을 풀어서 구하세요.

#공분산
#(편차를 곱한것의 합을 표본개수(표본이라 N-1, 모집단 전체이면 N)로 나눠줌)
mu_x = cov_data.x.mean()  #x 평균
mu_y = cov_data.y.mean()  #y 평균
N = len(cov_data) # 개수
df = N-1 # 자유도

cov = sum((cov_data.x-mu_x)*(cov_data.y-mu_y))/df
# 공분산 = (x값-x평균값) 곱하기 (y값-y평균값) 의 합 / 자유도
```


```python
#공분산 행렬
np.cov(cov_data.x, cov_data.y, ddof=1) 
```

    <ipython-input-19-8a9b479634fe>:2: DeprecationWarning: scipy.cov is deprecated and will be removed in SciPy 2.0.0, use numpy.cov instead
      sp.cov(cov_data.x, cov_data.y, ddof=1)
    




    array([[ 3.64622222,  7.67333333],
           [ 7.67333333, 28.01111111]])




```python
cov # sp.cov 공분산 계산한것과 같다.
```




    7.673333333333336




```python
#02.상관계수를 풀어서 구하세요.
# 상관관계(공분산을 표준편차로 나눔)
sigma_x = cov_data.x.std()  #ddof=1 디폴트값
sigma_y = cov_data.y.std()
cor = cov/(cov_data.x.std()*cov_data.y.std())
# 상관계수 = 공분산 / sp.sqrt( 데이터 x의 표준편차 곱사기 데이터 y의 표준편차)
```


```python
#상관 행렬: df.corr() 도 가능
np.corrcoef(cov_data.x,cov_data.y)  #ddof 영향없음.
```




    array([[1.       , 0.7592719],
           [0.7592719, 1.       ]])




```python
cor #np.corrcoef() 상관계수 계산한것과 같다.
```




    0.7592719041137088



### 2) 정규분포
norm = stats.norm()
- 많은 통계방법에서 모수는 정규분포를 따를거라는 가정.
- 정규분포의 표본분포를 통한 확인
- 정규분포의 표분의 개수가 많을수록 표본의 평균은 정규분포의 평균과 가까워진다.


```python
#평균 4, 표준편차 0.8인 정규분포에서 10개 샘플 추출
population = stats.norm(loc=4,scale=0.8)
population.rvs(size= 10)  #랜덤샘플
```




    array([4.57961807, 3.56129294, 4.307118  , 4.05207123, 3.77510296,
           2.51604812, 3.75124001, 3.63449545, 4.1361518 , 3.21819278])




```python
#0으로 빈 배열 만들기 : np.zeros
sample_mean_array = np.zeros(10000)
```


```python
# 집어넣어준다. -> 반복적인 행동
# 표본의 평균을 구하는 법 
# population.rvs(size= 10).mean() 
# 표본평균 (샘플사이즈가 10개) 
# 만개 구해서 빈행렬에 넣어준다
sample_mean_array = np.zeros(10000) 
for i in range(0,10000):
    sample_mean_array[i] = population.rvs(size= 10).mean()
sample_mean_array[:10]  #sample에 들어간 것들은 표본평균들(평균 4, 표준편차 0.8)
```




    array([4.14453047, 4.27935323, 3.91459474, 4.14200547, 4.11233808,
           4.10969124, 3.99576557, 4.23773869, 4.14007356, 4.00650778])




```python
#표본평균의 평균, 표준편차 구하기.
print(sample_mean_array.mean().round(2)) #평균
print(np.std(sample_mean_array, ddof=1).round(2)) 
# 표준편차 #ddof 신경써주기
#- 준비된 데이터가 모집단(전체)일때 0
#- 준비된 데이터가 불편분산, 불편표준편차일 경우 1
```

    4.0
    0.25
    


```python
#확률밀도함수 구하기
sns.histplot(sample_mean_array,color='black', kde=True)
#표본평균 만개의 분포 히스토그램
#아래 그래프 면적 1
#평균 4, 표준편차 0.8인 정규분포
```




    <AxesSubplot:ylabel='Count'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_27_1.png?raw=true)
    



```python
#샘플사이즈 변화( 많아질수록 표준편차가 작아진다, 평균으로 좁아진다. 
# -> 학습 추정) 
#그림으로 확인하기

#샘플사이즈변화에 따라 평균변화
#샘플사이즈를 변화시키면서 표본평균을 도출하는 시뮬레이션

#샘플사이즈변화
size_array = np.arange(10,100100,100)

np.random.seed(1)
sample_mean_array_size= np.zeros(len(size_array))  #빈 array 그릇 만들어주기!!!
for i in range(0, len(size_array)):
    sample = population.rvs(size=size_array[i]) #10개부터 100개씩 증가하는(100100-1까지) 정규분포 랜덤샘플
    sample_mean_array_size[i] = np.mean(sample) #샘플사이즈 10개 평균, 110개평균, 210개 평균...

#그림그리기    
plt.plot(size_array, sample_mean_array_size, color = 'skyblue') 
#표본의 수가 증가함에따라 표본평균이 어떻게 변화하는지 그래프

plt.xlabel('sample size')  #표본 개수
plt.ylabel('sample mean')  #표본의 평균 (설정했던 평균 4)
plt.axhline(y=4, xmin=0, xmax=1, color = 'b')
plt.show()
#샘플사이즈가 커질수록 평균(4)으로 가까워지는 것을 확인할 수 있다.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_28_0.png?raw=true)
    


### 3) 샘플사이즈(표본 개수)에 따른 평균비교
샘플 사이즈가 커질수록 표본평균이 모평균에 가까워지고 모평균으로 밀집도가 높아진다.


```python
np.random.seed(1) #랜덤고정

#표본평균 계산하는 사용자함수 만들기.
def cal_sample_mean(norm, size, n_trial):  #사이즈, 표본개수
    sample_mean_array = np.zeros(n_trial)  #빈그릇 생성(표본의 수 만큼)
    for i in range(0,n_trial):  #그릇에 담기는 동안
        sample_mean_array[i] = norm.rvs(size = size).mean() #모집단의 랜덤표본 size개의 평균
    return sample_mean_array

population = stats.norm(loc=4,scale=0.8)  
#이 정규분포에 대해 표본개수에 따른 표본평균의 변화를 관찰.

# 샘플 사이즈가 10
size_10 = cal_sample_mean(population,10, 10000)
# 샘플 사이즈가 20
size_20 = cal_sample_mean(population,20, 10000)
# 샘플 사이즈가 30
size_30 = cal_sample_mean(population,30, 10000)
```


```python
#표본개수에 따른 평균값들 가지고 df 만들기
df = pd.DataFrame([size_10,size_20,size_30],
                  index=['size_10_sample_mean','size_20_sample_mean',
                         'size_30_sample_mean'])
df.T.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size_10_sample_mean</th>
      <th>size_20_sample_mean</th>
      <th>size_30_sample_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.922287</td>
      <td>4.021685</td>
      <td>4.129223</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.864329</td>
      <td>3.899744</td>
      <td>3.772253</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T.plot(kind='density')
plt.show() #표본수가 30이 가장 평균 4 에 가까운것을 확인할 수 있다.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_32_0.png?raw=true)
    



```python
# 다른방법

# 표본샘플 사이즈가 10
size_10 = cal_sample_mean(population,10, 10000)
size_10_df = pd.DataFrame({'sample_mean':size_10,  #딕셔너리 키-값으로 df만들기
                          'size' : np.tile('size_10',10000)}) #np.tile(str,num) str을 num개 붙이라는 의미
# 표본샘플 사이즈가 20
size_20 = cal_sample_mean(population, 20, 10000)
size_20_df = pd.DataFrame({'sample_mean':size_20,
                          'size' : np.tile('size_20',10000)})
# 표본샘플 사이즈가 30
size_30 = cal_sample_mean(population, 30, 10000)
size_30_df = pd.DataFrame({'sample_mean':size_30,
                          'size' : np.tile('size_30',10000)})

#위 3개의 df 합치기
sim_result = pd.concat([size_10_df,size_20_df,size_30_df], ignore_index=True) #행방향 합치기
sim_result.head() #3만개


# 바이올린 플롯을 그리고, 3개 그룹에 대한 인사이트를 기술하세요.
sns.violinplot(x='size', y='sample_mean',data=sim_result)
#분포와 밀집도를 둘다 알 수 있음.
```




    <AxesSubplot:xlabel='size', ylabel='sample_mean'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_33_1.png?raw=true)
    


- 추출횟수가 같더라도, 표본샘플의 사이즈가 커질수록 모평균으로 밀집도가 높아지는 것을 알 수 있다.
- 추출횟수가 같더라도, 표본샘플의 사이즈가 커질수록 표준편차가 작아지는 것을 확인할 수 있다.       
샘플사이즈에 의미가 특별하다.

[다른분들의 인사이트]       
- 세 분포 모두 평균 4.0을 중심으로 종모양의 분포 형태를 갖는다. 
- size=10일때는 size=20,30일때보다 분포가 넓게 퍼져있고, 
- 샘플사이즈가 커질수록 분포가 평균값에 몰려있다는 것을 확인할 수 있다. 
- 즉, 샘플사이즈가 클수록 표본에서 구한 평균이 모평균과 가까운 값일 확률이 높아진다는 의미이다.

- 추출횟수가 같더라도 샘플의 사이즈가 커질수록 표본평균이 모평균에 근사함
- 추출횟수가 같더라도 샘플의 사이즈가 커질수록 표준편차가 작아짐

1. 샘플 사이즈가 커질수록 표본평균에 근접한 값의 밀도가 높아짐.
2. 즉 샘플 사이즈가 커질수록 표본평균이 모평균에 가까워짐.
3. 샘플 사이즈의 크기가 중요해진다.

### 4) 샘플사이즈(표본 개수)에 따른 표준편차의 변화
키워드 : 표본평균의 표준편차, 표준오차


```python
#샘플사이즈가 커짐
size_array = np.arange(2,102,2)
#표본평균의 표준편차를 저장할 배열(그릇)
sample_mean_std_array = np.zeros(len(size_array))   

#시뮬레이션 : 샘플사이즈가 커질수록 표본평균의 표준편차가 작아짐.
for i in range(len(size_array)):
    sample_mean = cal_sample_mean(population, size=size_array[i],n_trial=100)  
    #101까지 2씩 증가하는 array -> 샘플 2씩 증가 #n_trial = 표본개수 100개 고정
    sample_mean_std_array[i] = np.std(sample_mean, ddof=1)
plt.plot(size_array, sample_mean_std_array,color='black')
plt.xlabel('sample size')
plt.ylabel('mean std value')  
plt.show() #샘플사이즈가 커짐에 따라서 표본평균의 표준편차(표준오차)가 줄어드는 것을 확인할 수 있다.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_36_0.png?raw=true)
    



```python
# 표준오차 : 표본평균과 모평균과의 표준적인 차이,
#(샘플사이즈가 커지면 표본평균이 모평균에 근사해지니까 차이는 작아짐)
standard_error = 0.8/np.sqrt(size_array) #(모집단의 표준편차)/(루트 샘플사이즈)

#시뮬레이션 결과(표준평균의 표준편차)와 표준오차간 비교
plt.plot(size_array, sample_mean_std_array,color='black')
plt.plot(size_array, standard_error,color='red', linestyle='dotted')
plt.xlabel('sample size')
plt.ylabel('mean std value')

#거의 일치
#표본평균의 표준편차로 표준오차를 쓰기도 한다.
```




    Text(0, 0.5, 'mean std value')




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_37_1.png?raw=true)
    


### 5) 정규분포의 함수
확률밀도함수(pdf), 누적분포함수(cdf), 누적분포함수의 역함수(ppf)


```python
#확률밀도함수(pdf, probability density function)
stats.norm.pdf(loc= 평균, scale=표준편차, x = 배열이나 숫자)

#확률밀도함수 그래프
x_plot = np.arange(1,7.1,0.1)
plt.plot(x_plot, stats.norm.pdf(x=x_plot,loc=4,scale=0.8),color='black')
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_39_0.png?raw=true)
    



```python
norm_dist = stats.norm(loc=4,scale=0.8) #정규분포
norm_dist.pdf(x=4) #정규분포의 확률밀도함수에서 x가 4일때의 값 출력
```




    0.49867785050179086




```python
#누적분포함수 (cdf, cumulative distribution function)
stats.norm.cdf(loc=4,scale=0.8,x=7)  #정규분포의 누적분포함수에서 x가 7일때의 값 출력
```




    0.9999115827147992




```python
x_plot = np.arange(1,7.1,0.1)
plt.plot(x_plot,stats.norm.cdf(x=x_plot,loc=4,scale=0.8),color='midnightblue')
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_42_0.png?raw=true)
    



```python
stats.norm.cdf(loc=4,scale=0.8,x=4) #평균값은 딱 중간(0.5)에 있다.
```




    0.5




```python
# 누적분포함수의 역함수(inverse cumulative distribution function) ppf
# 퍼센트 포인트 - 하측확률 대응됨
# 하측확률 q 에 해당되는 퍼센트포인트(x값) 구할 때 이용
stats.norm.ppf(loc=4,scale=0.8,q=0.5) #하측확률이 q=0.5 일 x값(= 퍼센트포인트)
```




    4.0




```python
#하측확률 구할때, x= (퍼센트포인트) 이용
stats.norm.cdf(loc=4,scale=0.8, x=4) #하측확률은? 0.5(분위수퍼센트)
```




    0.5



### 6) t 분포
stats.**t**.pdf(x=x,df=5)  
-  자유도가 5인 t 분포의 확률밀도함수

[t분포]      
t 분포를 쓰는 이유       
- 모분산을 모르는 상황       
- 표본수가 30개 미만             
→ t분포는 자유도 만큼 정규분포와 차이가 난다



```python
#t 값의 표본분포
np.random.seed(1)
t_value_array = np.zeros(10000) #빈그릇

norm_dist = stats.norm(loc=4, scale=0.8) #정규분포

for i in range(0,10000):
    sample = norm_dist.rvs(size=10)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample,ddof=1)
    sample_se = sample_std/np.sqrt(len(sample))  
    #se: 표준오차(모평균과 표본 차이)  
    #se = 랜덤표본의 표준편차(표본평균과 모평균의 차이)/루트 샘플수
    t_value_array[i] = (sample_mean - 4)/sample_se  
    #t값 = (표본평균 - (4 : 모평균))/표준오차
    # => 크다 : 표준오차 작다, 작다 :표준오차 크다
```


```python
# t분포
sns.distplot(t_value_array, color = 'skyblue', kde=True)

#표준정규분포의 확률밀도 stats.norm.pdf()
x = np.arange(-8,8.1,0.1)
plt.plot(x, stats.norm.pdf(x=x),
        color = 'black', linestyle='dotted') #점선
plt.show()
#하늘색 t분포는 점선인 정규분포보다 살짝 넓게 퍼져있다.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_48_0.png?raw=true)
    



```python
# 자유도 n 이 커짐에 따라 표준정규분포 N(0,1)에 수렴
plt.plot(x, stats.norm.pdf(x=x),
         color = 'black', linestyle = 'dotted') 
plt.plot(x, stats.t.pdf(x=x,df=5),  #자유도가 5인 t분포
        color = 'blue')
plt.plot(x, stats.t.pdf(x=x, df=1),  #자유도가 1인 t분포
        color= 'red')
plt.show()
#자유도가 커질수록 표준정규분포에 근사한다.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_49_0.png?raw=true)
    


### 7) 신뢰구간
신뢰구간계산 : 신뢰구간이란 특정 신뢰계수를 만족하는 구간        
**(표본평균-모평균 ;=표준편차)/표준오차**로 계산한 t값. 


```python
df = pd.read_csv('dataset/5_7_fl.csv')
fish = df['length']  #시리즈로 가져오기
# 모평균의 점추정
mu = np.mean(fish)
# 모분산의 점추정
sigma_2 = np.var(fish, ddof=1) #ddof=1 불편추정량의 성질을 만족하기 위함
```


```python
# 표준편차
sigma = np.std(fish, ddof=1)
#표준오차
se = (sigma)/np.sqrt(len(fish)) #표준평균과 모평균의 차이 = 표준편차/루트 개수
#자유도 degree of freedom
df = len(fish) -1
```


```python
#신뢰구간 stats.t.interval
Interval = stats.t.interval(alpha = 0.95, df=df, loc=mu, scale=se) 
#alpha = 신뢰수준, df=자유도, loc = 평균, scale = 표준오차(t분포여서)
Interval #신뢰구간의 (하한가, 상한가)
#(3.5970100568358245 ~ 4.777068592173221) 이 신뢰구간이다.
```




    (3.5970100568358245, 4.777068592173221)




```python
# 샘플사이즈를 10배로 늘려서 신뢰구간 계산 - 신뢰구간 줄어듬
df_10 = (len(fish*10)) -1
se_10 = (sigma)/np.sqrt(len(fish)*10)
stats.t.interval(alpha=0.95, df=df_10, loc=mu, scale=se_10)

# 신뢰계수가 커질수록 신뢰구간의 폭이 넓어지고 안전해진다고 볼 수 있음.
```




    (4.0004556873051, 4.373622961703947)



-------------
## 2. stat2
가설 - 귀무가설,대립가설 

### 1) 가설검정 t-검정(t 통계량)
t 값 : 정규분포와 유사한  t분포의 값으로 t값이 크면 유의미한 차이가 있음.
- 대응표본 t-검정
- 독립표본 t-검정


```python
# t검정 : 귀무가설 _ 과자의 무게는 50g 이다.
junk_food = pd.read_csv('dataset/5_8_jfw.csv')
#표본평균 뽑기
jfood = junk_food.weight.copy()
mu = np.mean(jfood)
# 자유도
df = len(jfood)-1
# 표준편차
sigma = np.std(jfood, ddof=1)
# 표준오차
se = sigma/np.sqrt(len(jfood))
```


```python
# t 값
t_value = (mu-50)/se
t_value
```




    2.7503396831713434




```python
# p값(유의확률)을 구해서 p값이 유의수준 0.05보다 작으면 귀무가설 기각
# t 분포의 누적밀도함수(cdf) , x = t 값, df = 자유도
alpha = stats.t.cdf(t_value,df=df)
alpha #p값은 양쪽끝에 있음. #양측검정이니까 *2 필요
```




    0.9936372049937379




```python
(1-alpha)*2  #0.05보다 작다 -> 귀무가설 기각, 대립가설 채택
```




    0.012725590012524268



[대응표본 t검정]


```python
# 대응표본(약을 먹기 전과 후의 체온의 차이)
paired_test_data = pd.read_csv('dataset/5_9_ptt.csv')
ptdata = paired_test_data.copy()
ptdata.head(2)
# 귀무가설 : 약을 먹기 전고 후의 체온에 차이가 없다. 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>medicine</th>
      <th>body_temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>before</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>before</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 대응표본 t검정
# 데이터 전처리, 쿼리 이용하여 before 와 after을 구분하기
before = ptdata.query('medicine=="before"')['body_temperature']
after = ptdata.query('medicine=="after"')['body_temperature']
#배열로 변환
before = np.array(before)
after = np.array(after)
# 차이구하기
diff = after - before 
```


```python
# 차이가 평균값이 0과 다른지 검정
stats.ttest_1samp(diff,0)
# t검정 귀무가설에서 p값이 0.05보다 작으니 귀무가설 기각, 대립가설 채택됨
# 약을 먹기 전고 후의 체온에 유의미한 차이가 있다고 주장할 수 있다.
```




    Ttest_1sampResult(statistic=2.901693483620596, pvalue=0.044043109730074276)



[독립표본 t검정]


```python
# 독립표본으로 각각의 표본 평균 구해줌
mean_bf = np.mean(before)
mean_af = np.mean(after)
#분산
sigma_bf = np.var(before, ddof=1)
sigma_af = np.var(after, ddof=1)
#샘플사이즈
m = len(before)
n = len(after)
# t값
t_value = (mean_af-mean_bf)/np.sqrt((sigma_bf/m + sigma_af/n))
t_value # == Ttest_indResult의 statistic = 3.1557282344421034
```




    3.1557282344421034




```python
stats.ttest_ind(after,before,equal_var=False) 
# equal_val : 분산이 같다, 다르다를 가정한 t검정(다르다_False -> welch 검정)
# 독립표본 indepandent
# p값이 0.05보다 작다-> 귀무가설 기각(우연아님) 
# 귀무가설을 기각하고 유의미한 차이가 있다고 판단
# -> after, before 유의미한 차이가 잇다고 주장할 수 있음.
```




    Ttest_indResult(statistic=3.1557282344421034, pvalue=0.013484775682079892)



### 2) 카이제곱검정(x² 카이스퀘어검정)
- **독립성 검정**의 분할표 및 가설 설정
- 귀무가설 : 두 범주형 변수 사이에 연관이 없다.(독립이다.)
- 대립가설 : 두 범주형 변수 사이에 연관이 있다.(종속이다.)

stats.**chi2_contingency**(분할표, correction = False)


```python
# 분할표
click_data = pd.read_csv('dataset/5_10_cd.csv')
cross = pd.pivot_table(data=click_data, values = 'freq', 
                       aggfunc='sum',index='color',columns='click')
cross
#색이 클릭과 독립성이 있는지 분석 _ 독립성 분석
# 카이스퀘어 통계량으로 t값을 구함.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>click</th>
      <th>click</th>
      <th>not</th>
    </tr>
    <tr>
      <th>color</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>blue</th>
      <td>20</td>
      <td>230</td>
    </tr>
    <tr>
      <th>red</th>
      <td>10</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 카이스퀘어 통계량 , p값, 자유도, 기대도수표 출력
stats.chi2_contingency(cross, correction = False)
```




    (6.666666666666666,
     0.009823274507519247,
     1,
     array([[ 25., 225.],
            [  5.,  45.]]))



correction = False 파라미터 보정 안함.      
카이스퀘어 검정 시 기대도수가 전부 5 이상이어야 함.      
p값 0.05보다 작으므로 귀무가설(색에 따라 클릭버튼 관계 없음) 기각       
색에 따라 버튼을 클릭하는 것이 유의미하게 변한다고 판단. -> 독립성 없음.      

### 3) 수리모델 : 현상을 수식으로 표현한 모델. 
맥주 매상 = 20 + 4 * 기온

[모델]       
lm_model = smf.**ols** (formula= , data= )       
lm_model.fit() : 모델 학습시키는 문법.         
OLS Regression Results 이해하기.

Q-Q플롯 


```python
beer = pd.read_csv('dataset/7_1_beer.csv')
sns.jointplot(x=beer.temperature, y=beer.beer, kind='reg',
             color='skyblue')
plt.show() # 온도가 올라감에 따라 beer 가 올라감
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_73_0.png?raw=true)
    



```python
#모델구축
lm_model = smf.ols(formula="beer ~ temperature",
                  data = beer).fit()  #fit 은 학습시킴
lm_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>beer</td>       <th>  R-squared:         </th> <td>   0.504</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.486</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   28.45</td>
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 18 Aug 2021</td> <th>  Prob (F-statistic):</th> <td>1.11e-05</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:45:55</td>     <th>  Log-Likelihood:    </th> <td> -102.45</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    30</td>      <th>  AIC:               </th> <td>   208.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    28</td>      <th>  BIC:               </th> <td>   211.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>   34.6102</td> <td>    3.235</td> <td>   10.699</td> <td> 0.000</td> <td>   27.984</td> <td>   41.237</td>
</tr>
<tr>
  <th>temperature</th> <td>    0.7654</td> <td>    0.144</td> <td>    5.334</td> <td> 0.000</td> <td>    0.471</td> <td>    1.059</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.587</td> <th>  Durbin-Watson:     </th> <td>   1.960</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.746</td> <th>  Jarque-Bera (JB):  </th> <td>   0.290</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.240</td> <th>  Prob(JB):          </th> <td>   0.865</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.951</td> <th>  Cond. No.          </th> <td>    52.5</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



=> 해석 : 기온이 맥주 매상에 영향을 미친다는 것을 알 수 있음     
https://blog.naver.com/ehdanf1219 항목의 자세한 설명 참고.


```python
# 기온이 0도일 때 맥주 매상의 기댓값
# 방법1
lm_model.predict(pd.DataFrame({'temperature':[0]}))

# 방법2
beta0 = lm_model.params[0]
beta1 = lm_model.params[1]
temperature = 0
beta0 + (beta1 * temperature)  #y = a(기울기)x + b(편차)
```




    34.610215255741466




```python
lm_model.params
```




    Intercept      34.610215
    temperature     0.765428
    dtype: float64




```python
# 잔차계산
# 방법1
resid = lm_model.resid

#방법2
# 잔차 = 실제값 - 예측값
y_hat = beta0 + (beta1 * beer.temperature) # -> 예측값
beer.beer - y_hat  # 실제값 - 예측값
```


```python
# 잔차의 산포도 : x축 적합도(fittedvalues), y축 잔차(resid)
sns.jointplot(lm_model.fittedvalues,lm_model.resid, color = 'skyblue') #아래 모양 잘 되어있음.
plt.show()
# 잔차 플롯의 패턴 또는 절대 값이 큰 잔차는 회귀모형에 문제가 있음을 나타낸다.
# 패턴없는게 문제 없음.
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_79_0.png?raw=true)
    



```python
# 결정계수 R-squared
# Regression model의 성능을 측정하기 위해 mean value로 예측하는 단순 모델(Zero-R 모델)과
# 비교하여 상대적으로 얼마나 성능이 나오는지를 측정한 지표.
mu = np.mean(beer.beer)
y = beer.beer
yhat = lm_model.predict()

# np.sum((yhat-mu)**2) 모듈에의한 변화, 
# np.sum((y-mu)**2)전체변화 -> 모델이 전체를 설명할 수 있는 정도 (설명력)
np.sum((yhat-mu)**2)/np.sum((y-mu)**2)
```




    0.503959323061188




```python
# 수정결정계수 : 독립변수의 수가 늘어나면 
# 결정계수가 커지는 경향을 조정하기 위하여 
# 독립변수가 늘어나는 것에 대하여 패널티 부여
#방법1
n = len(beer.beer)
s = 1 # 독립변수의 개수
1 - ((np.sum(resid**2))/(n-s-1))/(np.sum((y-mu)**2)/(n-1))

# 방법2
lm_model.rsquared_adj
```




    0.4862435845990851




```python
# 종속변수의 변동 크기는 모델로 설명 가능한 변동과 설명 못하는 
# 잔차제곱합으로 분해할 수 있다.

# 종속변수의 변동
# 방법1
np.sum((y-mu)**2)

# 방법2
# np.sum((yhat-mu)**2)  모델이 설명하는 크기, 
# sum(resid**2) 잔차제곱합(모델이 설명하지 못하는)
np.sum((yhat-mu)**2) + sum(resid**2)
```




    3277.1146666666727




```python
# Q-Q플롯 : 정규화를 검토하기 위한 그래프
# 이론상의 분위점과 실제 데이터의 분위점(1,2,3,4분위)을 산포도 그래프로 그린 것.
#- 점선과 직선이 일치할수록, 데이터는 정규분포를 따른다.
#- 점선과 직선이 일치하지 않으면 데이터는 정규분포를 따르지 않는다.
import statsmodels.api as sm
fig = sm.qqplot(resid, line='s')
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_83_0.png?raw=true)
    


### 4) 분산분석
[분산분석 모델]      
anova_model = smf.ols('beer~weather', data = w_beer).fit()        
sm.stats.**anova_lm**(anova_model,typ=2)  
- F 비 이용한 분석
- F 분포에 근거함.
- F 비 : 효과의 분선 크기/ 오차의 분산 크기
- F 비가 크면 오차에 비해 효과의 영향이 클 것.
    - 검정, 분석 둘다 쓰임


```python
# 샘플데이터
weather = [
    "cloudy","cloudy",
    "rainy","rainy",
    "sunny","sunny"
]
beer = [6,8,2,4,10,12]
w_beer = pd.DataFrame({"beer":beer,
                      "weather" : weather})
```


```python
# 날씨에 따른 맥주판매
sns.boxplot(w_beer.weather,w_beer.beer, color ='skyblue') #군간변동이 있는것을 확인할 수 있다.
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_86_0.png?raw=true)
    



```python
# 날씨별 매상의 평균치
w_beer.groupby('weather').mean()
```


```python
# 날씨에 의한 영향 : 2일씩 6일에 대한 날씨별 매상의 평균치
effect = [7,7,3,3,11,11] #각 군의 평균을 넣어줌, 오차를 빼버림. 

# 군간변동 구하기
# 군간변동 => effect의 흩어진 정도
# effect의 흩어진 정도를 구함으로써 군간변동을 구할 수 있음.
mu_effect = np.mean(effect)
sq_model = np.sum((effect-mu_effect)**2)
sq_model
```




    64.0




```python
# 오차(Error(Residual = 잔차))는 beer 와 평균과의 차이
resid = w_beer.beer - effect

# 군내변동: 오차 제곱의 합. 오차의 평균값은 0
sq_resid = np.sum(resid**2)

# 자유도 구하기
df_model = 2 # 군간변동의 자유도(수준의 종류 수에 따라 좌우됨 : 수준(3) - 1)
df_resid = 3 # 군내변동의 자유도(샘플사이즈와 수준의 종류 수 : 샘플(6) - 수준(3))

# 군간 평균 제곱(분산)
v_model = sq_model/df_model

# 군내 평균 제곱(분산)
v_resid = sq_resid/df_resid

# F비를 구할 수 있다.
# F비는 군간분산과 군내분산의 비
f_ratio = v_model/v_resid
f_ratio
```




    16.0




```python
# p값 구하기
1- stats.f.cdf(x=f_ratio, dfn=df_model, dfd = df_resid) 
```




    0.02509457330439091




```python
# p값  - 0.05보다 작아 귀무가설 기각.-> 우연이 아니고 
# 날씨에 의해 맥주매상이 유의미하게 변화한다고 판단할 수 있다.
```


```python
# 날씨에 따른 맥주판매, 분산분석 모델
anova_model = smf.ols('beer~weather',
                     data = w_beer).fit()  #아노바모델만들어줌
sm.stats.anova_lm(anova_model,typ=2) #분산분석표 : 군간, 군내편차제곱합, 자유도, F비, p값
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>64.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>0.025095</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
anova_model.params  #파라미터값, 변수들의 회귀계수들
```




    Intercept           7.0
    weather[T.rainy]   -4.0
    weather[T.sunny]    4.0
    dtype: float64



### 5) 독립변수가 여럿인 모델


> 독립변수가 여러개 일 때, 분산분석이 필요한 이유,       
> 검정을 이용하여 모델 선택하는 방법


```python
sales = pd.read_csv('dataset/7_3_lmm.csv')
# pairplot 그려보기
sns.pairplot(data = sales, hue = 'weather', palette = 'Blues')
```




    <seaborn.axisgrid.PairGrid at 0x2a66d5641f0>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_96_1.png?raw=true)
    



```python
lm = smf.ols('sales ~ price', data = sales).fit()
lm.params #sales가 price와 양의상관관계라고 나옴. 그래프에서는 잘 안보임.
```




    Intercept    113.645406
    price          0.332812
    dtype: float64




```python
# 분산분석표 보기
sm.stats.anova_lm(lm, typ=2)
# p값이 0.05보다 작아서 유의미하다고 나옴.
# 잘못된 분석으로 보임, 가격이 오르면 매상도 증가한다.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1398.392322</td>
      <td>1.0</td>
      <td>4.970685</td>
      <td>0.028064</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>27570.133578</td>
      <td>98.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 매상을 상품가격만으로 분석하여 문제의 소지 발생.
# 그림을 그려서 원인 발견, 다른변수 무시, 가격과 매상만 비교하는 그림
sns.lmplot(x='price', y='sales',data = sales, palette='gray')
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_99_0.png?raw=true)
    



```python
# weather을 넣으니 그래프 변화 있음.
# 변수선택의 중요성이 보여짐
sns.lmplot(x='price', y='sales',data = sales, hue='weather', palette='gray')
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_100_0.png?raw=true)
    



```python
sales.groupby('weather').mean()
# rainy, sunny의 sales 차이가 보임.
# 매상을 가격만으로 분석하면 문제의 소지가 발생
# 날씨별로 보면 가격이 높아질 경우 매상이 줄어든다는 것을 알 수 있음.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>humidity</th>
      <th>price</th>
      <th>sales</th>
      <th>temperature</th>
    </tr>
    <tr>
      <th>weather</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rainy</th>
      <td>32.126</td>
      <td>295.5</td>
      <td>205.924</td>
      <td>20.422</td>
    </tr>
    <tr>
      <th>sunny</th>
      <td>30.852</td>
      <td>309.5</td>
      <td>222.718</td>
      <td>21.102</td>
    </tr>
  </tbody>
</table>
</div>




```python
#독립변수가 4개인 모델 추정
lm_sales = smf.ols(
"sales ~ weather+humidity+temperature+price", data = sales).fit()
lm_sales.params
# 교차효과가 없다고 가정하고 +  // 교차효과가 있다고 가정하면 * 사용해야 함.
# 위에서와는 다르게 price의 기울기 음수로 나옴.
# 2개 이원분산, 3개부터 다원분산분석.
```




    Intercept           278.627722
    weather[T.sunny]     19.989119
    humidity             -0.254055
    temperature           1.603115
    price                -0.329207
    dtype: float64




```python
#분산분석표
sm.stats.anova_lm(lm_sales).round(3) #소수점이 많아서 반올림처리
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>1.0</td>
      <td>7050.961</td>
      <td>7050.961</td>
      <td>38.848</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>1.0</td>
      <td>1779.601</td>
      <td>1779.601</td>
      <td>9.805</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>1.0</td>
      <td>2076.845</td>
      <td>2076.845</td>
      <td>11.443</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1.0</td>
      <td>818.402</td>
      <td>818.402</td>
      <td>4.509</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>95.0</td>
      <td>17242.717</td>
      <td>181.502</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



=> 모든 독립변수가 유의미하다고 도출되어 '잘못된 검정 결과'라 의심됨

- F검정(분산분석) 대신 회귀계수의 T검정을 수행하면 문제는 발생되지 않음,     
- 그러나 검정 다중성의 문제가 발생할 수 있음     
         

[검정 다중성의 문제]      
: 귀무가설이 기각되기 쉬어지고 1종오류가 발생할 가능성이 커짐.      
- "다중성은 여러 개의 검정을 동시에 실시함에 따라 그 중에 하나라도 우연히 기각할 확률이 점차 늘어남을 의미한다(김권현, 193)."     
- 귀무가설이 과소평가되는 것 - > 귀무가설이 참인데 기각되는 오류(1종오류)      
- 그래서 분산분석을 이용하여 한번에 분석.      


```python
lm_sales.summary().tables[1]
# humidity t검정의 p값 0.578, 0.05이상으로 귀무가설 채택 유의미하지 않음.
# 하지만 검색 다중성의 문제가 있음.
```




<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>  278.6277</td> <td>   46.335</td> <td>    6.013</td> <td> 0.000</td> <td>  186.641</td> <td>  370.615</td>
</tr>
<tr>
  <th>weather[T.sunny]</th> <td>   19.9891</td> <td>    3.522</td> <td>    5.675</td> <td> 0.000</td> <td>   12.997</td> <td>   26.982</td>
</tr>
<tr>
  <th>humidity</th>         <td>   -0.2541</td> <td>    0.456</td> <td>   -0.558</td> <td> 0.578</td> <td>   -1.159</td> <td>    0.651</td>
</tr>
<tr>
  <th>temperature</th>      <td>    1.6031</td> <td>    0.443</td> <td>    3.620</td> <td> 0.000</td> <td>    0.724</td> <td>    2.482</td>
</tr>
<tr>
  <th>price</th>            <td>   -0.3292</td> <td>    0.155</td> <td>   -2.123</td> <td> 0.036</td> <td>   -0.637</td> <td>   -0.021</td>
</tr>
</table>




```python
lm_sales2 = smf.ols(
"sales ~ weather+temperature+humidity+price", data = sales).fit()
sm.stats.anova_lm(lm_sales2).round(3)
# humidity, temperature 순서를 바꿨더니 humidity의 p값이 0.05보다 큼.
# 유의미한 변수 3개로 바뀜.- > 뭔가 잘못됨을 감지할 수 있음.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>1.0</td>
      <td>7050.961</td>
      <td>7050.961</td>
      <td>38.848</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>1.0</td>
      <td>3814.779</td>
      <td>3814.779</td>
      <td>21.018</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>1.0</td>
      <td>41.667</td>
      <td>41.667</td>
      <td>0.230</td>
      <td>0.633</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1.0</td>
      <td>818.402</td>
      <td>818.402</td>
      <td>4.509</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>95.0</td>
      <td>17242.717</td>
      <td>181.502</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
lm_sales2.summary().tables[1]  
#t검정 결과 값이 humidity, temperature 순서를 바꾸기 전과 같음을 알 수 있음.
```




<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>  278.6277</td> <td>   46.335</td> <td>    6.013</td> <td> 0.000</td> <td>  186.641</td> <td>  370.615</td>
</tr>
<tr>
  <th>weather[T.sunny]</th> <td>   19.9891</td> <td>    3.522</td> <td>    5.675</td> <td> 0.000</td> <td>   12.997</td> <td>   26.982</td>
</tr>
<tr>
  <th>temperature</th>      <td>    1.6031</td> <td>    0.443</td> <td>    3.620</td> <td> 0.000</td> <td>    0.724</td> <td>    2.482</td>
</tr>
<tr>
  <th>humidity</th>         <td>   -0.2541</td> <td>    0.456</td> <td>   -0.558</td> <td> 0.578</td> <td>   -1.159</td> <td>    0.651</td>
</tr>
<tr>
  <th>price</th>            <td>   -0.3292</td> <td>    0.155</td> <td>   -2.123</td> <td> 0.036</td> <td>   -0.637</td> <td>   -0.021</td>
</tr>
</table>



**위의 검정에서 주요 문제 ; 검색다중성의 문제**


```python
# 그래서 변수 개수가 다수이면 Type 2 ANOVA 사용
# 모든 변수가 포함된 
mod_full = smf.ols(
"sales ~ weather+humidity+temperature+price", sales).fit()
sm.stats.anova_lm(mod_full, typ=2).round(3)
# F검정의 p값이 T검정 p값과 같은 결과로 나옴.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>5845.878</td>
      <td>1.0</td>
      <td>32.208</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>56.425</td>
      <td>1.0</td>
      <td>0.311</td>
      <td>0.578</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>2378.017</td>
      <td>1.0</td>
      <td>13.102</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>price</th>
      <td>818.402</td>
      <td>1.0</td>
      <td>4.509</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>17242.717</td>
      <td>95.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#유의미하지 않은 변수(humidity) 제거
mod_non_humi = smf.ols(
"sales ~ weather+temperature+price", sales).fit()
sm.stats.anova_lm(mod_non_humi, typ=2).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>6354.966</td>
      <td>1.0</td>
      <td>35.266</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>4254.736</td>
      <td>1.0</td>
      <td>23.611</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>price</th>
      <td>803.644</td>
      <td>1.0</td>
      <td>4.460</td>
      <td>0.037</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>17299.142</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 습도는 기온과 강한 상관관계가 있으며 기온이라는 독립변수가 포함되어 있으면
# 습도는 매상에 영향을 끼친다고 볼 수 없음.
```

**문제 해결 :     
※ 독립변수 여러개일 때 Type 2 ANOVA 모델 선택.**

**독립변수가 여러개일 때는 아노바(F검정이용하는, Type 2 ANOVA)이용**
- t검정일 때는 검색 다중의 문제가 발생

![image.png](attachment:image.png)

### 6) 독립변수들간의 교우작용
- 결과에서 교우관계를 확인한 결과(2가지가 종속변수에 미치는 영향) 독립변수들끼리 p값이 0.05미만인 것은 확인되지 않는다.
- 이원분산분석 부터는 교우작용도 따져줘야한다.
- 시험문제에 아노바분석은 * 로, 교우작용이 유의미한지 확인해줘야 한다.


```python
# 변수간의 교우작용 확인
mod_non_humi = smf.ols(
"sales ~ weather*temperature*price", sales).fit()
sm.stats.anova_lm(mod_non_humi, typ=2).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>weather</th>
      <td>6425.428</td>
      <td>1.0</td>
      <td>36.086</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>temperature</th>
      <td>4278.061</td>
      <td>1.0</td>
      <td>24.026</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>weather:temperature</th>
      <td>71.995</td>
      <td>1.0</td>
      <td>0.404</td>
      <td>0.526</td>
    </tr>
    <tr>
      <th>price</th>
      <td>803.336</td>
      <td>1.0</td>
      <td>4.512</td>
      <td>0.036</td>
    </tr>
    <tr>
      <th>weather:price</th>
      <td>96.890</td>
      <td>1.0</td>
      <td>0.544</td>
      <td>0.463</td>
    </tr>
    <tr>
      <th>temperature:price</th>
      <td>390.204</td>
      <td>1.0</td>
      <td>2.191</td>
      <td>0.142</td>
    </tr>
    <tr>
      <th>weather:temperature:price</th>
      <td>357.056</td>
      <td>1.0</td>
      <td>2.005</td>
      <td>0.160</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>16381.404</td>
      <td>92.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



---------------------------

## 3. stat 과제

[과제]    
Q. 'dataset/5_2_shoes.csv' 을 데이터프레임으로 불러와서 아래 작업을 수행하세요.
- 4행 3열을 복사하여 수직으로 결합하여 8행 3열의 데이터프레임 만드세요.
- 교차분석표를 만드세요.(values='sales',aggfunc='sum',index='store',columns='color')
- 독립성 검정을 수행하세요.


```python
shoes = pd.read_csv('dataset/5_2_shoes.csv')
shoes2 = shoes.copy()
shoes = pd.concat([shoes,shoes2])
shoes.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store</th>
      <th>color</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tokyo</td>
      <td>blue</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tokyo</td>
      <td>red</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>osaka</td>
      <td>blue</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
shoes_cross = pd.pivot_table(data = shoes, values='sales',
                             aggfunc='sum',index='store',columns='color')
shoes_cross
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>color</th>
      <th>blue</th>
      <th>red</th>
    </tr>
    <tr>
      <th>store</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>osaka</th>
      <td>26</td>
      <td>18</td>
    </tr>
    <tr>
      <th>tokyo</th>
      <td>20</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 독립성 검정 -> 카이제곱검정(chi2_contingency)

stats.chi2_contingency(shoes_cross, correction = False)
```




    (3.413537549407115,
     0.06466368573255789,
     1,
     array([[21.53191489, 22.46808511],
            [24.46808511, 25.53191489]]))




```python
#통계량 3.413537549407115,
#p값 0.06466368573255789,
#자유도 1,
#기대도수표  [[21.53191489, 22.46808511],
#            [24.46808511, 25.53191489]]))
```

[과제]    
Q. lm_model 선형모델을 생성하고 summary 를 출력한 후 모델에 대해 해석하세요.


```python
df = pd.read_csv('C:\workspace\cakd3\programming\dataset/auto-mpg.csv')
df_mw = df[['mpg', 'weight']]
df_mw.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>3693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>3436</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 선형모델을 생성(smf.ols)
df_lm_model = smf.ols(formula="mpg ~ weight",
                  data = df).fit()
# summary 를 출력
df_lm_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.692</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.691</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   888.9</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 18 Aug 2021</td> <th>  Prob (F-statistic):</th> <td>2.97e-103</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:49:07</td>     <th>  Log-Likelihood:    </th> <td> -1148.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   398</td>      <th>  AIC:               </th> <td>   2301.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   396</td>      <th>  BIC:               </th> <td>   2309.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   46.3174</td> <td>    0.795</td> <td>   58.243</td> <td> 0.000</td> <td>   44.754</td> <td>   47.881</td>
</tr>
<tr>
  <th>weight</th>    <td>   -0.0077</td> <td>    0.000</td> <td>  -29.814</td> <td> 0.000</td> <td>   -0.008</td> <td>   -0.007</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>40.423</td> <th>  Durbin-Watson:     </th> <td>   0.797</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  56.695</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.713</td> <th>  Prob(JB):          </th> <td>4.89e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.176</td> <th>  Cond. No.          </th> <td>1.13e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.13e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



[모델해석]
       
Dep. Variable: mpg - **종속변수 : mpg**      
Model:OLS	- 최소자승법(OLS: Ordinary Least Squares), 잔차(Residual) 제곱의 합을 최소로 하는 방법이 최소자승법이며, 최소자승법을 활용하여 데이터를 가장 잘 표현하는 선형 회귀선을 그릴 수 있다. (근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법)         
        
Method:	Least Squares - 최소자승근사법(method of least squares)        
No. Observations:398	- Number of observations, 관찰표본 수, 즉 총 표본 수 Df Residuals:396	 - DF는 Degree of Freedom으로 자유도를 뜻하는데, DF Residuals는 전체 표본 수에서 측정되는 변수들(종속변수 및 독립변수)의 개수를 빼서 구한다.        
        
Df Model:1 - **독립변수의 개수가 한개**        
Covariance Type:nonrobust -  공분산 타입. 특별히 지정하지 않으면 onrobust가 됨        
        
R-squared:0.692 - 결정계수, 모델 설명력, **mpg의 분산이 weight를 약 69%를 설명한다.** 1이면 모델이 완벽하게 데이터를 100% 설명해주는 상태이다.     
Adj. R-squared:	0.691 - 독립변수의 개수와 표본의 크기를 고려하여 R-squared를 보정한 값.     
- 종속 변인과 독립변인 사이에 상관관계가 높을수록 1에 가까워진다.     
- 결정계수는 전체 변동폭의 크기에 대한 모델로 설명 가능한 변동폭의 비율이라고 할 수 있음
      
      
      
F-statistic:888.9 - F통계량, 검정 통계량이 귀무 가설 하에서 F- 분포를 갖는 통계 검정      
Prob (F-statistic):	2.97e-103 - 회귀모형에 대한 (통계적) 유의미성 검증 결과, 유의미함 (p < 0.05)     
**p값이 0.05보다 작으므로 귀무가설 기각됨, mpg가 weight와 유의미하다고 판단할 수 있음.**

      
Log-Likelihood:	-1148.4 - 최대로그우도, 특정 사건이 일어날 가능성을 비교할 수는 없을까?: 가능도(Likelihood)           
- 연속 사건: 가능도 ≠ 확률, 가능도 = PDF값
- 셀 수 있는 사건(이산사건): 가능도 = 확률  
- 확률변수의 표집값과 일관되는 정도를 나타내는 값.(주어진 표집값에 대한 모수의 가능도는 이 모수를 따르는 분포가 주어진 관측값에 대하여 부여하는 확률이다.)          

AIC:2301, BIC:2309 - AIC, BIC: 로그우도를 독립변수의 수로 보정한 값 (작을 수록 좋다)        
     
Intercept.coef : 46.3174  - 절편값     
weight.coef : 	-0.0077	 -"weight"의 회귀계수. 즉, 기울기     
     
**y = -0.0077\*x + 46.3174**     
     
P>|t|  0 : **0.05보다 작기 때문에 유의하다고 판단할 수 있다.**    
     
Skew(왜도): 0.713 - 데이터가 치우친것을 의미함.      
- 오른쪽으로 치우침 = 왜도 < 0
- 왼쪽으로 치우침 = 왜도 > 0

**왜도가 0보다 크기 때문에 왼쪽으로 데이터가 치우친 것을 알 수 있다.**  

Kurtosis(첨도): 4.176 -  자료 분포가 뾰족한 정도를 나타내는 척도      
- 정규분포 = 첨도 0 (Pearson 첨도 = 3)
- 위로 뾰족함 = 첨도 > 0 (Pearson 첨도 >3)
- 아래로 뾰족함 = 첨도 < 0 (Pearson 첨도 < 3)
     
**첨도가 3보다 크기 때문에 위로 뾰족한 분포인것을 알 수 있다.**     

Durbin-Watson:	0.797 - 시계열데이터일때 체크 필수!
- 잔차의 자기상관을 체크하는 지표.
- 보통 1.5 ~ 2.5사이이면 독립으로 판단하고 회귀모형이 적합하다는 것을 의미
- 시계열데이터를 분석하는 경우 반드시 이 지표를 체크. 잔차에 자기상관이 있으면 계수의 t검정 결과 신뢰 못함
- Durbin-Watson 통계량이 2보다 크거나 차이가 난다면 일반화 제곱법 등의 사용 검토 필요

**[도전 과제]**       
FIFA 데이터는 가상의 온라인 축구게임에 등장하는 축구 선수의 주요 특징과 신체 정보에 대한 데이터이다.     
변수는 'ID', 'Name', 'Age', 'Nationality', 'Overall', 'Club', 'Preferred_Foot', 'Work_Rate', 'Position', 'Jersey_Number', 'Contract_Valid_Until','Height', 'Weight_lb', 'Release_Clause', 'Value', 'Wage'와 같다.


```python
fifa = pd.read_csv('C:\workspace\cakd3\programming\dataset/FIFA.csv')
fifa.columns
```




    Index(['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Club', 'Preferred_Foot',
           'Work_Rate', 'Position', 'Jersey_Number', 'Contract_Valid_Until',
           'Height', 'Weight_lb', 'Release_Clause', 'Value', 'Wage'],
          dtype='object')



Q1. FIFA데이터에서 각 선수의 키는 Heghit변수에 피트와 인치로 입력되어 있습니다. 이를 cm로 변환하여 새로운 변수 Height_cm을 생성하시오. ( “ ' ” 앞의 숫자는 피트이며, “ ' ” 뒤의 숫자는 인치, 1피트 = 30cm, 1인치 = 2.5cm)


```python
cm = fifa['Height'].str.split('\'', expand=True)
cm[[0, 1]] = cm[[0, 1]].astype(int)
cm['cm'] = cm[0] * 30+ cm[1]* 2.5
fifa['Height_cm'] = cm['cm']
```


```python
fifa.Height_cm.head()
```




    0    167.5
    1    185.0
    2    172.5
    3    190.0
    4    177.5
    Name: Height_cm, dtype: float64



 Q2. 포지션을 의미하는 Position변수를 아래 표를 참고하여 “Forward”, “Midfielder”,“Defender”, “GoalKeeper”로 재범주화하고, 변환하여 Position_Class 라는 변수를 생성하고 저장하시오.

Forward : ['LS','ST','RS','LW','LF','CF','RF','RW']       
Midfielder : ['LAM','CAM','RAM','LM','LCM','CM','RCM','RM']      
Defender : ['LBW','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']      
GoalKeeper 


```python
def position(x):
    if x in ['LS','ST','RS','LW','LF','CF','RF','RW']:
        return 'Forward'
    elif x in ['LAM','CAM','RAM','LM','LCM','CM','RCM','RM']:
        return 'Midfielder'
    elif x in ['LBW','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']:
        return 'Defender'
    else : return 'GoalKeeper'
        
fifa[['Position_Class']] = fifa.Position.map(position)
```


```python
fifa.Position.head()
```




    0     RF
    1     ST
    2     LW
    3     GK
    4    RCM
    Name: Position, dtype: object




```python
fifa.Position_Class.head()
```




    0       Forward
    1       Forward
    2       Forward
    3    GoalKeeper
    4    Midfielder
    Name: Position_Class, dtype: object



Q3. 새로 생성한 Position_Class 변수의 각 범주에 따른 Value(선수의 시장가치)의 평균값의 차이를 비교하는 일원배치 분산분석을 수행하고 결과를 해석하시오.


```python
# 귀무가설 : 각 범주에 따른 Value(선수의 시장가치)의 평균값의 차이가 없다.
from statsmodels.stats.anova import anova_lm
# 일원배치 분산분석(anova_lm)
model = smf.ols('Value~Position_Class', data = fifa).fit()
anova_lm(model)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df</th>
      <th>sum_sq</th>
      <th>mean_sq</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Position_Class</th>
      <td>3.0</td>
      <td>4.062654e+09</td>
      <td>1.354218e+09</td>
      <td>41.682393</td>
      <td>7.946657e-27</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>16638.0</td>
      <td>5.405515e+11</td>
      <td>3.248897e+07</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



- F값 41.682396
- P값 0.05보다 작기 때문에 귀무가설 기각
- 따라서 Position_Class(4가지 포지션)와 Value(선수의 시장가치)는 통계적으로 유의미한 차이가 있다고 판단할 수 있다.


```python
# 특히 네 가지 포지션들 중 특히나 어떠한 포지션들 간에 선수의 시장가치에 
# 차이가 있는지 파악하기 위해 사후검정을 수행한다. 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
posthoc = pairwise_tukeyhsd(fifa.Value, fifa.Position_Class, alpha=0.05)
print(posthoc)
```

            Multiple Comparison of Means - Tukey HSD, FWER=0.05        
    ===================================================================
      group1     group2    meandiff  p-adj    lower      upper   reject
    -------------------------------------------------------------------
      Defender    Forward   930.3291  0.001   610.1302  1250.528   True
      Defender GoalKeeper  -488.0626 0.0046  -863.0391  -113.086   True
      Defender Midfielder   760.8347  0.001   486.0072 1035.6623   True
       Forward GoalKeeper -1418.3917  0.001 -1841.4645 -995.3189   True
       Forward Midfielder  -169.4944 0.5609  -507.0049  168.0162  False
    GoalKeeper Midfielder  1248.8973  0.001   859.0339 1638.7607   True
    -------------------------------------------------------------------
    

post-hoc을 사용하는 이유 :  t-test를 시행한후에 **집단간 차이가 있는지**를 통계수치를 통해 확인이 가능하기 때문.

- 포지션 Forward-Midfielder는 유의수준이 0.05보다 크므로 통계적으로 시장가치에 대해서 차이가 없다고 볼 수 없다.
- 반면 다른 모든 포지션에 대해서는 유의수준이 0.05보다 작으므로 통계적으로 시장가치에 대해 차이가 있다고 판단할 수 있다.

Q4. Preferred Foot(주로 사용하는 발)과 Position_Class(재범주화 된 포지션)변수에 따라 Value(이적료)의 차이가 있는지를 알아보기 위해 이원배치분산분석을 수행하고 결과를 해석하시오.


```python
# 귀무가설: 선수의 발에 따른 선수의 가치에는 차이가 없다. 
# 선수의 포지션에 따른 선수의 가치에는 차이가 없다. 발과 포지션간의 상호작용 효과가 없다.
# 대립가설: 선수의 발에 따른 선수의 가치에는 차이가 있다. 
# 선수의 포지션에 따른 선수의 가치에는 차이가 있다. 발과 포지션간의 상호작용 효과가 있다.

#이원배치분산분석
model = smf.ols('Value ~ Position_Class * Preferred_Foot', fifa).fit()
anova_lm(model, typ=2).round(3) #독립변수 여러개이리 때, 아노바 typ=2 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Position_Class</th>
      <td>4.081002e+09</td>
      <td>3.0</td>
      <td>41.910</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>Preferred_Foot</th>
      <td>1.644328e+08</td>
      <td>1.0</td>
      <td>5.066</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>Position_Class:Preferred_Foot</th>
      <td>4.727252e+08</td>
      <td>3.0</td>
      <td>4.855</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>5.399144e+11</td>
      <td>16634.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



- 포지션과 발에 대해 p-value값이 유의주순 0.05보다 작다 -> 귀무가설 기각
- 즉, 선수의 포지션에 따른 선수의 가치에 차이가 있고 선수의 발에 따른 선수 가치의 차이가 있다.
- 또한, 발과 포지션 변수의 p값도 0.05보다 작으므로 귀무가설이 기각되며 선수의 발과 포지션의 상호작용에 의한 효과가 있다고 판단할 수 있다.

Q5.Age, Overall, Wage, Height_cm, Weight_lb 가 Value에 영향을 미치는지 알아보는 회귀분석을 단계적 선택법을 사용하여 수행하고 결과를 해석하시오.

**변수 선택법(Variable Selection)**은     

1.전진선택법(Forward Selection) / 2.후진소거법(Backward Elimination) / 3.단계적선택법(Stepwise Selection) 이 있다.


```python
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

def stepwise_selection(data, target,SL_in=0.05,SL_out = 0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<SL_in):
            best_features.append(new_pval.idxmin())
            while(len(best_features)>0):
                best_features_with_constant = sm.add_constant(data[best_features])
                p_values = sm.OLS(target, best_features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()
                if(max_p_value >= SL_out):
                    excluded_feature = p_values.idxmax()
                    best_features.remove(excluded_feature)
                else:
                    break 
        else:
            break
    return best_features
    
    
X = fifa[['Wage','Overall','Age','Height_cm','Weight_lb']]
y = fifa['Value'].values 

stepwise_selection(X,y)
```




    ['Wage', 'Overall', 'Age', 'Height_cm']



**선택된 변수는 ['Wage', 'Overall', 'Age', 'Height_cm']** 이다.


```python
# 선택된 변수들을 이용하여 분산분석
model = smf.ols('Value~Wage+Overall+Age+Height_cm', fifa).fit()
anova_lm(model, typ=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Wage</th>
      <td>1.826353e+11</td>
      <td>1.0</td>
      <td>26672.280256</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Overall</th>
      <td>2.490562e+10</td>
      <td>1.0</td>
      <td>3637.246280</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>1.147466e+10</td>
      <td>1.0</td>
      <td>1675.772760</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Height_cm</th>
      <td>5.140396e+07</td>
      <td>1.0</td>
      <td>7.507096</td>
      <td>0.006152</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>1.139199e+11</td>
      <td>16637.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
model.summary2()
```




<table class="simpletable">
<tr>
        <td>Model:</td>               <td>OLS</td>         <td>Adj. R-squared:</td>      <td>0.791</td>   
</tr>
<tr>
  <td>Dependent Variable:</td>       <td>Value</td>             <td>AIC:</td>         <td>309167.6615</td>
</tr>
<tr>
         <td>Date:</td>        <td>2021-08-18 19:03</td>        <td>BIC:</td>         <td>309206.2600</td>
</tr>
<tr>
   <td>No. Observations:</td>        <td>16642</td>        <td>Log-Likelihood:</td>   <td>-1.5458e+05</td>
</tr>
<tr>
       <td>Df Model:</td>              <td>4</td>           <td>F-statistic:</td>      <td>1.572e+04</td> 
</tr>
<tr>
     <td>Df Residuals:</td>          <td>16637</td>      <td>Prob (F-statistic):</td>    <td>0.00</td>    
</tr>
<tr>
      <td>R-squared:</td>            <td>0.791</td>            <td>Scale:</td>        <td>6.8474e+06</td> 
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>Coef.</th>   <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th>   <th>[0.025</th>     <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>-8690.8178</td> <td>588.2795</td> <td>-14.7733</td> <td>0.0000</td> <td>-9843.9084</td> <td>-7537.7273</td>
</tr>
<tr>
  <th>Wage</th>       <td>184.1837</td>   <td>1.1278</td>  <td>163.3165</td> <td>0.0000</td>  <td>181.9731</td>   <td>186.3942</td> 
</tr>
<tr>
  <th>Overall</th>    <td>241.3450</td>   <td>4.0018</td>   <td>60.3096</td> <td>0.0000</td>  <td>233.5011</td>   <td>249.1889</td> 
</tr>
<tr>
  <th>Age</th>        <td>-202.1603</td>  <td>4.9384</td>  <td>-40.9362</td> <td>0.0000</td>  <td>-211.8401</td>  <td>-192.4805</td>
</tr>
<tr>
  <th>Height_cm</th>   <td>-8.4446</td>   <td>3.0821</td>   <td>-2.7399</td> <td>0.0062</td>  <td>-14.4858</td>    <td>-2.4034</td> 
</tr>
</table>
<table class="simpletable">
<tr>
     <td>Omnibus:</td>    <td>17089.038</td>  <td>Durbin-Watson:</td>      <td>1.407</td>   
</tr>
<tr>
  <td>Prob(Omnibus):</td>   <td>0.000</td>   <td>Jarque-Bera (JB):</td> <td>3525378.426</td>
</tr>
<tr>
       <td>Skew:</td>       <td>4.665</td>       <td>Prob(JB):</td>        <td>0.000</td>   
</tr>
<tr>
     <td>Kurtosis:</td>    <td>73.690</td>    <td>Condition No.:</td>      <td>5577</td>    
</tr>
</table>



모형 방정식은     
y = -8690.818 + 184.184\*Wage + 241.345\*Overall -202.160\*Age -8.446\*Height_cm


모형의 결정계수(R-squared)는 0.79 이다.     
즉, 다변량 회귀식은 전체 데이터의 80%를 설명하고있다고 판단할 수 있다.    
    
F통계량의 p값이 0.05보다 작음으로 귀무가설이 기각되며     
모형이 통계적으로 유의하다고 판단할 수 있다.  

---------

## 4. 회귀
- 소득이 증가하면 소비도 증가. 어떤 변수가 다른 변수에 영향을 준다면 두 변수 사이에 선형관계가 있다고 할 수 있음
- 두 변수 사이에 일대일로 대응되는 확률적, 통계적 상관성을 찾는 알고리즘을 Simple Linear Regression이라고 함. 지도학습
- 변수 X와 Y에 대한 정보를 가지고 일차 방정식의 계수 a,b를 찾는 과정이 단순회귀분석 알고리즘

### 1) 회귀객체
- 사이킷런      
from sklearn.linear_model import LinearRegression       
lr = LinearRegression()  


```python
df = pd.read_excel('C:\workspace\cakd3\programming\dataset/auto-mpg.xlsx')
df.head()
# 안될때는 옵션에 engine = 'openpyxl'
ndf = df[['mpg','cylinders','horsepower','weight']].copy()
```


```python
ndf.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>horsepower</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>130</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>165</td>
      <td>3693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>150</td>
      <td>3436</td>
    </tr>
  </tbody>
</table>
</div>




```python
#horsepower 물음표 처리
ndf.horsepower.replace('?', np.nan, inplace=True)
ndf.dropna(subset=['horsepower'], axis=0,inplace=True)
```


```python
ndf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 392 entries, 0 to 397
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   mpg         392 non-null    float64
     1   cylinders   392 non-null    int64  
     2   horsepower  392 non-null    float64
     3   weight      392 non-null    int64  
    dtypes: float64(2), int64(2)
    memory usage: 15.3 KB
    


```python
sns.pairplot(ndf)
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_160_0.png?raw=true)
    



```python
# 변수(속성) 선택
X = ndf[['weight']] #독립변수 데이터프레임  -> 밑에 메소드에 넣어야할 형식
y = ndf['mpg'] #종속변수 시리즈

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,
                                                   random_state=11) 
# test_size, 검증용데이터 30%
```


```python
lr = LinearRegression()  #객체생성
```


```python
lr.fit(X_train, y_train)   
# 데이터 넣어서 선형회귀 알고리짐 학습시키기
```




    LinearRegression()




```python
y_preds = lr.predict(X_test)  #X_test 에서 도출되는 y값과 y_test와 비교
y_preds[:5]
```




    array([29.27985295, 25.65957977, 27.90795996, 24.97363328, 15.02740907])



### 2) 회귀 평가 지표
- MAE : 실제값과 예측값의 차이를 절대값으로 변환해 평균한 것
- MSE : 실제값과 예측값의 차이를 제곱해 평균한 것
- RMSE : MSE에 루트를 씌운 것(실제 오류 평균보다 커지는 것 보정)_ 사이킷런에는 없어서 MSE 구해서 sqrt 루트 씌워줌.
- R square(설명력): 분산 기반으로 예측 성능을 평가. 실제값의 분산 대비 예측값의 분산 비율을 지표로 함.



```python
mse = mean_squared_error(y_test,y_preds) 
# 실제값과 예측값의 차이
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_preds)
print(f'MSE :{mse:.3f} ,RMSE :{rmse:.3f}, R2 :{r2:.3f}')
```

    MSE :22.169 ,RMSE :4.708, R2 :0.641
    

설명력 64% 정도              
weight 에 따른 mpg(종속변수)              
예측값과 실제값 차이가 4정도 _ 이거는 상대적(단위에 따라 변동가능)         


```python
# 회귀식 기울기 a
lr.coef_  
#왜 그림과는 다르기 기울기가 작을까 
# : 종속변수와 독립변수간의 수 차이가 커서
```




    array([-0.00762163])




```python
# 절편 b
lr.intercept_
```




    45.971217704757684




```python
y.plot(kind='hist') #정규본포와 많이 거리가 있다.
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_170_1.png?raw=true)
    



```python
y_hat = lr.predict(X) #독립변수에 대한 종속변수 예측값
pd.DataFrame(y_hat).plot(kind='hist')  #정규본포와 많이 거리가 있다.
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_171_1.png?raw=true)
    



```python
import warnings
warnings.filterwarnings('ignore')
# 선형회귀(회귀계수가 선형/비선형 인지를 보고 선형이라함.)
plt.figure(figsize=(10,5))
ax1 = sns.distplot(y, hist=False, label="y")
ax2 = sns.distplot(y_hat, hist=False, label = "y_hat")
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_172_0.png?raw=true)
    


### 3) 다항회귀분석
- 직선보다 곡선으로 설명하는 것이 적합할 경우 다항 함수를 사용하면 복잡한 곡선 형태의 회귀선을 표현할 수 있음
- 2차 함수 이상의 다항함수를 이용하여 두 변수 간의 선형관계를 설명하는 알고리즘
- 다항 회귀도 선형회귀임. 선형/비선형 회귀를 나누는 기준은 회귀계수가 선형/비선형 인지에 따르며 독립변수의 선형/비선형 여부와는 무관


```python
# 변수(속성) 선택
X = ndf[['weight']] #독립변수 데이터프레임  -> 밑에 메소드에 넣어야할 형식
y = ndf['mpg'] #종속변수 시리즈

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,
                                                   random_state=11) 
# 여기서 X 1차 -> 다항식으로 변환
# PloynomialFeatures # 다항식 변환 메소드
```


```python
poly = PolynomialFeatures(degree=2) #차수설정필요, 2차항 적용
X_train_poly = poly.fit_transform(X_train) #X_train 1차항 데이터를 2차항으로 변환
print(X_train.shape)
print(X_train_poly.shape)
```

    (274, 1)
    (274, 3)
    


```python
pr = LinearRegression()
pr.fit(X_train_poly,y_train) #다항식으로 회귀 학습 -> 다항회귀

X_test_poly = poly.fit_transform(X_test)
#인스턴스.score 사용
r_score = pr.score(X_test_poly, y_test) #r_score 구하는 또다른 방법 
r_score 
# 직선보다 약간의 곡선으로써 좀더 잘 반영되어 설명력이 올라간다. 
# 샘플보정없애면 잘 보임.
```




    0.6368479947378759




```python
#r2_score사용
y_preds2 = pr.predict(X_test_poly) # y 예측값을 구해서
r2_poly = r2_score(y_test, y_preds2) # y실체값, y예측값
r2_poly
```




    0.6368479947378759




```python
# 그림으로 표시
y_hat_test = pr.predict(X_test_poly) #다항식의 x에 대한 y 예측값

fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(1,1,1)
ax.plot(X_train,y_train,'o',label='Train Data') #기본 데이터 분포
ax.plot(X_test, y_hat_test, 'r+', label= 'Predicted Value') # 다항식으로 변환했을 때
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_178_0.png?raw=true)
    



```python
# 예측값 실제값 비교
X_poly = poly.fit_transform(X)
y_hat = pr.predict(X_poly)

plt.figure(figsize=(10,5))
ax1 = sns.distplot(y,hist=False,label = 'y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat')

```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_179_0.png?raw=true)
    


아까와 비교하면 예측값과 실제값이 가까워진것을 확인할 수 있다.           
다항 회귀를 하는 이유 -> 실제값에 가까워짐        
곡선이 직선보다 더 적합시킬 수 있다. - 독립변수가 하나일때!!!            

------------------------

==> 설명하기 위한 내용            
실제로는 여러개의 독립변수가 있는 것을 가지고 분석함       
-> 다중회귀분석          

### 4) 다중 회귀 분석
- 여러 개의 독립 변수가 종속 변수에 영향을 주고 선형 관계를 갖는 경우에 다중회귀분석을 사용한다.
- 다중 회귀 분석 알고리즘은 각 독립변수의 계수와 상수항에 적절한 값들을 찾아서(학습을 통해) 모형을 완성. 지도학습


```python
# 변수가 여러개일 때의 다중회귀.
X = ndf[['cylinders','horsepower','weight']]
y = ndf['mpg']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,
                                                    random_state = 0)

lr = LinearRegression()
lr.fit(X_train,y_train) #학습
y_preds = lr.predict(X_test)   #학습 후 예측

# r_square = lr.score(X_test, y_test)
r_square = r2_score(y_test, y_preds)
mse = mean_squared_error(y_test, y_preds) #실제값 예측값의 평균 차이
rmse = np.sqrt(mse)
print(f'MSE :{mse:.3f} ,RMSE :{rmse:.3f}, R2 :{r_square:.3f}')

# 회귀식 기울기
print('X변수의 회귀계수 :', lr.coef_)  #변수 3개에 대한 회귀계수
print('절편 :', lr.intercept_)
```

    MSE :19.674 ,RMSE :4.436, R2 :0.680
    X변수의 회귀계수 : [-0.57598375 -0.03393439 -0.00537578]
    절편 : 45.99088694107769
    


```python
y_hat = lr.predict(X_test)
plt.figure(figsize=(10,5))
ax1 = sns.distplot(y_test, hist= False, label = 'y_test')
ax2 = sns.distplot(y_hat, hist = False, label = 'y_hat')
```


    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_183_0.png?raw=true)
    



```python
ndf.head(3) 
#단위가 다다른것을 볼 수있다 -> 스케일링을 통해 단위맞춰주기
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>cylinders</th>
      <th>horsepower</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>130.0</td>
      <td>3504</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>165.0</td>
      <td>3693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>150.0</td>
      <td>3436</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = MinMaxScaler() #객체생성 
ndf_ms = scaler.fit_transform(ndf) #(0~1사이)바꿔줌, 배열로 반환
ndf_ms
```




    array([[0.2393617 , 1.        , 0.45652174, 0.5361497 ],
           [0.15957447, 1.        , 0.64673913, 0.58973632],
           [0.2393617 , 1.        , 0.56521739, 0.51686986],
           ...,
           [0.61170213, 0.2       , 0.20652174, 0.19336547],
           [0.50531915, 0.2       , 0.17934783, 0.2869294 ],
           [0.58510638, 0.2       , 0.19565217, 0.31386447]])




```python
ndf_ms_df = pd.DataFrame(data = ndf_ms, columns = ndf.columns)
X = ndf_ms_df[['cylinders','horsepower','weight']]
y = ndf_ms_df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,
                                                   random_state = 0)
lr = LinearRegression()
lr.fit(X_train,y_train)
re_score = lr.score(X_test, y_test)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test,y_preds)  # 숫자가 작아져서 mse도 작아짐(상대적)
rmse = np.sqrt(mse)
print(f'MSE :{mse:.3f} ,RMSE :{rmse:.3f}, R2 :{re_score:.3f}')
```

    MSE :0.014 ,RMSE :0.118, R2 :0.680
    


```python
# 이번에는 종속변수 부터 보기
sns.distplot(ndf.mpg)  #정규분포에 약간 못미치는 형태
# 정규분포가 최대한 되게 하려면 어떻게 해야할가?
# 스탠다스 스케일러 이용해보기
```




    <AxesSubplot:xlabel='mpg', ylabel='Density'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_187_1.png?raw=true)
    



```python
# Standardization 평균 0 / 분산 1
scaler = StandardScaler()   
scaler = scaler.fit_transform(ndf)
#scaler = scaler.fit_transform(ndf[['mpg']]) mpg만 뽑아두됨
ndf_mpg_df = pd.DataFrame(data = scaler, columns = ndf.columns)
sns.distplot(ndf_mpg_df.mpg) 
```




    <AxesSubplot:xlabel='mpg', ylabel='Density'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_188_1.png?raw=true)
    



```python
# 스케일러해도 변화 별로 없음 -> 로그변환 시켜주기
log_mpb = np.log1p(ndf['mpg']) 
#원본+1에 자연로그나, 원본에 log1p나 결과가 같은 것을 볼 수 있다.
sns.distplot(log_mpb) #왜곡됬던게 바뀐것을 확인할 수 있다.
```




    <AxesSubplot:xlabel='mpg', ylabel='Density'>




    
![png](https://github.com/601chl/601chl.github.io/blob/master/_posts/output_189_1.png?raw=true)
    


[과제]             
Q. 로그변환된 종속변수를 적용하여 다중회귀분석결과를 평가하세요.         
**로그변환된** r_square, mse, rmse 구하기.       


```python
X = ndf[['cylinders','horsepower','weight']]
y = np.log1p(ndf['mpg'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,
                                                    random_state = 0)

lr = LinearRegression()
lr.fit(X_train,y_train)  #로그변환된 y값으로 학습
y_preds = lr.predict(X_test) #예측값또한 로그변환으로 학습되어 도출된 값.
r_square = r2_score(y_test, y_preds)
# r_square = lr.score(X_test, y_test)
mse = mean_squared_error(y_test, y_preds) #실제값 예측값의 평균 차이
rmse = np.sqrt(mse) #실제값도 로그변환, 예측값도 로그변환
# 변환되지 않은것이 있으면 다시 로그변환전으로 돌려서 계산해야함.

print(f'MSE :{mse:.3f} ,RMSE :{rmse:.3f}, R2 :{r_square:.3f}')
# 로그변환된 종속변수를 적용한 다중회귀분석에서
# r_square (결정계수, 설명력)의 값이 가장 높다.
```

    MSE :0.023 ,RMSE :0.153, R2 :0.771
    
