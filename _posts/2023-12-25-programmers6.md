---
layout: single
title:  "programmers_문제6_나머지가 1이 되는 수 찾기"
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
: 나머지가 1이 되는 수 찾기.

<br>

### python 풀이 모음

- 핵심 <br>
%, 나눴을 때 나오는 나머지 반환하는 기호

- 문제설명 <br>
자연수 n이 매개변수로 주어집니다. n을 x로 나눈 나머지가 1이 되도록 하는 가장 작은 자연수 x를 return 하도록 solution 함수를 완성해주세요. 답이 항상 존재함은 증명될 수 있습니다.



- 제한사항 <br>
3 ≤ n ≤ 1,000,000
<br>

```python
# 통과완료
def solution(n):
    if n >= 3:
        for i in range(1,1000000):
            if n % i == 1:
                return i
            else : pass
```

<br>
<br>

```python
# 다른 풀이 1
def solution(n):
    return [x for x in range(1,n+1) if n%x==1][0]
```


<br>


```python
# 다른 풀이 2 (while 이용)
def solution(n):

    if not 3 <= n <= 1000000 :
        return False

    answer = 2
    while True :
        if n % answer == 1 :
            break
        else :
            answer += 1

    return answer
```



<br>
<br>
<br>




(출처 : 프로그래머스, [코딩테스트 연습](https://programmers.co.kr/learn/challenges) )     

