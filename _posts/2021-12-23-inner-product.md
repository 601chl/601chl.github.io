---
layout: single
title:  "programmers_coding_문제1_내적"
categories: coding_practice
tag: [python, coding, study, programmers, practice, level1]
toc: true
toc_sticky: false
author_profile: false
sidebar:
    nav: "docs"
search: true
---


## 목표 : a와 b의 내적을 return 하도록 solution 함수완성하기.

### 내적이란?

[ 내적 | 內積 | inner product ]      
쌓을 적(積). '곱'의 구용어로, 여기서는 '곱한다'는 뜻    
벡터의 곱하기는 두 가지 정의가 있는데, 내적은 벡터를 마치 수처럼 곱하는 개념이다.

내적은 여러가지 연산 중 하나로, 벡터와 벡터의연산이다. 결과가 스칼라 라는 점이 특이한 점을 가진다.      
‘벡터+벡터=벡터’, ‘스칼라+스칼라=스칼라’와 같은 형태처럼 인풋과 아웃풋의 형태가 같지만, 이 내적이라는 연산은 신기하게도 벡터와 벡터를 연산 하는데 스칼라라는 아웃풋이 도출된다.

간단한 식 (내적은 어떤 연산인가?)      
    
$$<u,v> = u*v = u_1v_1 + u_2v_2 + ⋯ + u_nv_n$$

<br>
<br>

         
출처1 : [위키독스, 수학 용어를 알면 개념이 보인다, 042. 내적 vs 외적](https://wikidocs.net/22384)           
출처2 : [로스카츠의 AI 머신러닝, 선형대수_내적(inner product) 의미](https://losskatsu.github.io/linear-algebra/innerproduct/#1%EB%82%B4%EC%A0%81%EC%9D%98-%EC%A0%95%EC%9D%98)


<br>
<br>

### python 풀이 모음


```python
# 풀이 1
def solution(a, b):
    return sum([x*y for x, y in zip(a, b)])
```

#### 내장함수 zip           
여러 개의 순회 가능한(iterable)객체를 인자로 받고, 각 객체가 담고 있는 원소를 튜플 형태로 차례로 반환함.           
동일한 개수로 이루어진 자료형을 묶어 주는 역할.
예를들어 같은 길이의 리스트를 같은 인덱스끼리 잘라 튜플로 반환함.    
만약 배열의 길이가 다를 경우 같은 인덱스끼리만 짝지어주고, zip 객체에서 나머지 인덱스는 제외된다.


```python
# zip 이해하기 쉬운 예시
a = [1,2,3]   
b = [4,5,6]  
for z in zip(a, b):    
    print(z)

out:(1, 4)
    (2, 5)
    (3, 6)
```


```python
# 풀이 2
def solution(a, b):
    return sum(map(lambda i: a[i]*b[i], range(len(a))))
```

len(a) : a의 길이, 개수 반환         
range(len(a)) : 0 ~ len(a)-1 까지의 정수      

<br>

#### lambda           
lambda 인자 : 리턴값           
람다함수는 결과부분된 부분을 return키워드 없이 자동으로 return해줍니다.              
<img src="https://wikidocs.net/images/page/22804/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2018-11-07_05.56.24.png" align="left"  width="40%"/>
               
<br>
<br>
<br>
<br>
<br>

lambda i: a[i]×b[i]           
i를 인자로 받아서 a[i]×b[i]를 리턴함.          



#### map       
map(함수, 리스트)     
이 함수는 함수와 리스트를 인자로 받음.         
리스트로부터 원소를 하나씩 꺼내서 함수를 적용시킨 다음, 그 결과를 새로운 리스트에 담아줍니다.     

sum(리스트) : 리스트 숫자들의 합.


```python
# 풀이 3
def solution(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]*b[i])
    return sum(c)
```

for 문을 이용하여 a[i]×b[i] 을 리스트에 담고 그 합을 return 함.


```python
# 풀이 4
solution = lambda x, y: sum(a*b for a, b in zip(x, y))
```

x, y를 입력했을때, sum(a×b for a, b in zip(x, y)) 을 return 함.    
a×b for a, b in zip(x, y) : x,y를 같은 인덱스끼리 잘라 튜플로 반환. a, b는 반환된 튜플(a, b) 값


```python
# 풀이 5
solution = lambda a,b: sum([a[i]*b[i] for i in range(len(a))])
```

a,b를 입력했을때, sum([a[i]×b[i] for i in range(len(a))]) 을 return 함.    
- [_] 리스트화     
- a[i]×b[i] :  a 인덱스의 i번째 자리 와 b 인덱스의 i번째 자리의 곱      
- 즉, 같은 자리의 수끼리 곱. i는 range(len(a))     
    
=> for 문이 돌아가는 동안 a[i]×b[i]가 계산되어 [_] 리스트에 담김.     
sum([_]) : 리스트에 담긴 수들의 합.    


```python
# 풀이 6
def solution(a, b):
    answer = 0
    for x, y in zip(a,b):
        answer += x*y
    return answer
```


```python
# 풀이 7
def solution(a, b):
    answer = 0
    for i in range(len(a)):
        answer += a[i]*b[i]
    return answer
```


```python
# 풀이 8
def solution(a, b):
    answer = 0
    for idx, an in enumerate(a) : 
        answer += (an * b[idx])
    return answer
```

#### enumerate       
리스트가 있는 경우 순서와 리스트의 값을 반환.        
순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체(자료형의 현재 순서(index), 그 값)를 리턴.      
보통 enumerate 함수는 for문과 함께 자주 사용됩니다.     

enumerate(a) : a를 받아 (자료형의 현재 순서(index), 그 값)을 리턴함.

for 문이 돌아갈 때마다 answer += (an * b[idx])     
a 의 현재 순서(index)에 있는 값과, b에서 같은 순서(index)에 있는 값을 곱하여 answer에 더함.


```python
# 풀이 9
def solution(a, b):
    import numpy as np
    return int(np.dot(a, b))
```

#### np.dot
numpy 라이브러리에서 numpy.dot은 행렬의 곱을 표현 할 때 사용합니다.
- numpy array a와 b가 있을 때, 이 둘이 각각 1차원 행렬(vectro)라면 각 자리 수끼리 곱한 후 전부 더합니다. 내적과 같은 연산입니다.
- 행렬 a와 그의 전치 행렬 aT 에 대한 dot은 a 행렬의 요소들의 제곱의 합(스칼라와 같다.)
- 만약 a가 N차원 행렬이고, b가 1차원 행렬이라면, a의 마지막 축에 b를 곱하여 더한 값을 나타낸다.

<br>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd3vdNn%2FbtqZiUf5DJv%2FXa1V7fed221qzeyOXTCpmk%2Fimg.png" align="left" width="40%"/>


<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
    
출처 : [개발자비행일지, numpy.dot()](https://cyber0946.tistory.com/99)
