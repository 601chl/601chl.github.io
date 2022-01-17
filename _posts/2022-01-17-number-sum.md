---
layout: single
title:  "programmers_coding_문제3_없는숫자더하기"
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
: numbers에서 찾을 수 없는 0부터 9까지의 숫자를 모두 찾아 더한 수를 return 하기.



### python 풀이 모음


```python
# 풀이 1
def solution(numbers):
    answer =  [0,1,2,3,4,5,6,7,8,9]
    for number in numbers:
        answer.remove(number)
    return sum(answer)
```

<br>

```python
# 간단 풀이 1
def solution(numbers):
    return 45 - sum(numbers)
```


```python
# 간단 풀이 2
solution = lambda x: sum(range(10)) - sum(x)
```
#### lambda           
lambda 인자 : 리턴값           
람다함수는 결과부분된 부분을 return키워드 없이 자동으로 return해줍니다.              
<img src="https://wikidocs.net/images/page/22804/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2018-11-07_05.56.24.png" align="left"  width="80%"/>
               
<br>
<br>
<br>
<br>
<br>
<br>
lambda i: a[i]×b[i]           
i를 인자로 받아서 a[i]×b[i]를 리턴함.          

```python
# 간단 풀이 3
def solution(numbers):
    return sum([i for i in [1,2,3,4,5,6,7,8,9,0] if i not in numbers])
```


```python
from collections import Counter
def solution(numbers):
    cnt = Counter(numbers)
    return sum([n for n in range(1,10) if n not in cnt.keys()])
```

#### Counter
```python
Counter('hello world') 
# Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})
```
- dictionary를 이용한 카운팅
- Counter 클래스는 이와 같은 작업을 좀 더 쉽게 할 수 있도록, 데이터의 개수가 많은 순으로 정렬된 배열을 리턴하는 most_common이라는 메서드를 제공
```python
Counter('hello world').most_common(1) 
# [('l', 3)]
```

(출처 : 프로그래머스, [코딩테스트 연습](https://programmers.co.kr/learn/challenges) )     
(출처 : https://www.daleseo.com/python-collections-counter/)
