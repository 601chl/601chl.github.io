---
layout: single
title:  "programmers_coding_문제2_소수만들기"
header:
      teaser: "/assets/images/programmers/level1.png"
categories: coding_practice
tag: [python, coding, study, programmers, practice, level1]
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---


## 목표     
: 주어진 숫자 중 3개의 수를 더했을 때 소수가 되는 경우의 개수구하기.
- nums에 들어있는 숫자의 개수는 3개 이상 50개 이하.
- nums의 각 원소는 1 이상 1,000 이하의 자연수이며, 중복된 숫자가 들어있지 않음.

### 소수는?
**1보다 큰 자연수 중 1과 자기 자신만을 약수로 가지는 수.**    
예를 들어, 5는 1×5 또는 5×1로 수를 곱한 결과를 적는 유일한 방법이 그 수 자신을 포함하기 때문에 5는 소수이다. 그러나 6은 자신보다 작은 두 숫자의 곱(2×3)이므로 소수가 아닌데, 이렇듯 1보다 큰 자연수 중 소수가 아닌 것은 합성수라고 한다.


소수 예시 (1부터 100까지 소수)      
    
![image](https://www.lgsl.kr/contents/sl_image/ALMA/ALMA2018/ALMA201803/ALMA2018030003017.jpg)

<br>
<br>

         
출처1 : [위키백과, 소수](https://ko.wikipedia.org/wiki/%EC%86%8C%EC%88%98_(%EC%88%98%EB%A1%A0))    


<br>
<br>

### 소수 판별하기
1) 나머지가 0인 수 검사    
어떤 수 A가 소수인지 판별하기 위해서는 2부터 A-1 까지 나누어서 나머지가 0인 경우가 있는지 검사    
"%" : 나눗셈의 나머지 - 나눗셈 결과의 '나머지'를 가져옴
```python
# 1부터 100 사이의 소수 구하기
n=100
def Prime_num(a):
    if(a<2):
        return False
    for i in range(2,a):
        if(a%i==0):
            return False
    return True

for i in range(n+1):
    if(Prime_num(i)):
        print(i)  # 소수이면(True)프린트
```

2) 제곱근 이용    
어떤 수 A가 소수인지 판별하기 위해서는 2부터 루트A까지의 수 중 한 개의 수에 대해서라도 나누어 떨어지면 소수가 아님.    
파이썬에서 제곱근(Square Root) 계산 방법 : 숫자 ** (1/2), math.sqrt(숫자), cmath.sqrt(숫자)  _ cmath : 복소수에 대한 제곱근도 구할 수 있음

```python
def prime_number(x):
    for i in range(1,int(x**0.5)+1):
        if x%i==0:
            print(i) # 소수이면(True)프린트
```


3) 아리스토텔리스의 체 반환
```python
n=1000
a = [False,False] + [True]*(n-1)
primes=[]

for i in range(2,n+1):
  if a[i]:
    primes.append(i)
    for j in range(2*i, n+1, i):
        a[j] = False
print(primes)
```


<br>
<br>

### python 풀이 모음


```python
# 풀이 1
from itertools import combinations

def prime_num(a):
    if (a<2):
        return False
    for i in range(2,(a//2)+1):
        if(a%i==0):
            return False
    return True

def solution(nums):
    combi = list(combinations(nums, 3))
    li=[]
    for case in combi:
        if(prime_num(sum(case))):
            li.append(sum(case))
    return len(li)

```

#### combinations, 조합           
- 서로 다른 n개의 원소를 가지는 어떤 집합에서 순서에 상관없이 r개의 원소를 선택하는 것.           
- itertools 라이브러리에서 순열(permutations)과 조합(combinations)을 구하는 함수를 이용할 수 있음.      

#### itertools 라이브러리
- 효율적인 루핑을 위한 이터레이터를 만드는 함수 라이브러리.
  - combinations(iterable, r) : iterable에서 원소 개수가 r개인 조합 만들기, 조합은 튜플로 입력 iterable의 순서에 따라 사전식 순서로 output됨.    
  - combinations_with_replacement(iterable, r) : iterable에서 원소 개수가 r개인 중복 조합 만들기.
  - product(iterable, r=None) : iterable에서 원소 개수가 r개인 순열 만들기, r을 지정하지 않거나 r=None으로 하면 최대 길이의 순열을 리턴함.
  - permutations(*iterables, repeat=1) : 여러 iterable의 데카르트곱 리턴, product는 다른 함수와 달리 인자로 여러 iterable을 넣어줄 수 있고 그 친구들간의 모든 짝을 지어서 리턴함.

<br>

출처 : [파이썬 색인 표준 라이브러리, 함수형 프로그래밍 모듈
itertools — 효율적인 루핑을 위한 이터레이터를 만드는 함수](https://docs.python.org/ko/3.8/library/itertools.html)

<br>
<br>

```python
# 풀이 2 (제곱근 이용)
from itertools import combinations
def prime_number(x):
    answer = 0
    for i in range(1,int(x**0.5)+1):
        if x%i==0:
            answer+=1
    return 1 if answer==1 else 0

def solution(nums):
    return sum([prime_number(sum(c)) for c in combinations(nums,3)])

```
solution 함수를 바로 리턴값에 리스트 안에 for 문을 넣어 간결하게 표현한것이 인상깊음.