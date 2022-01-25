---
layout: single
title:  "Django 시작하기"
categories: python-django
header:
      teaser: "/assets/images/django.png"
tag: [study, python, django, tutorial]
toc: true
toc_sticky: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---


![django](../../../assets/images/django.png)

<br>

< 장고 시작하기 >      
출처 : [Django 4.0 documentation, 장고에 대해 알아야할 모든것](https://docs.djangoproject.com/en/4.0/)

<br>
<br>

# Django 훑어보기
- 파이썬으로 만들어진 무료 오픈소스 웹 애플리케이션 프레임워크(web application framework)     
  출처 및 추천: [장고란 무엇인가요? 왜 프레임워크가 필요한가요?](https://tutorial.djangogirls.org/ko/django/)

- 장고는 공통적인 엡 개발 업무를 빠르고 쉽게 만들어주도록 설계됨. 데이터베이스 기반 웹앱을 만들 수 있음.     


<br>
<br>

## 모델설계
- 장고를 데이터베이스 없이 쓸 수 있지만, 데이터베이스 레이아웃을 파이썬 코드로 표현하는 object-relational mapper 가 같이 따라옴.      
  ORM(Object Relational Mapping, 객체-관계 매핑) : 객체와 관계형 데이터베이스의 데이터를 자동으로 연결해주는 것, 객체 지향 프로그래밍은 class를 사용하고 관계형 데이터베이스는 테이블을 사용함.       
  출처 : [OMR이란](https://gmlwjd9405.github.io/2019/02/01/orm.html)
       
-  model.py 는 설계할 db모델을 표현하는 풍부한 방법을 제공해줌. 
      
```python
# model.py 예시
from django.db import models

class Article(models.Model):
    pub_date = models.DateField()
    headline = models.CharField(max_length=200)
    content = models.TextField()
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

    def __str__(self):
        return self.headline

```

<br>
<br>

## 모델설치(데이터베이스 생성)
django command-line 유틸리티를 실행, 백엔드 지원 : PostgreSQL, MySQL, SQLite 

```bash
$ python manage.py makemigrations
$ python manage.py migrate

```

- makemigrations : 생성 가능한 모델 찾아서 migrations 생성. 모델을 변경한 것에 기반한 새로운 마이그레이션을 만들 때 사용.           
  마이그레이션은 당신의 모델에 생긴 변화(필드를 추가했다던가 모델을 삭제했다던가 등등)를 반영하는 Django의 방식   
  makemigrations --name NAME, -n NAME 옵션 사용하여 이름지정 가능.   

  > django-admin makemigrations [app_label [app_label ...]]  
  > migrations디렉터리 가 없는 앱에 마이그레이션을 추가

- migrate : 마이그레이션을 실행하고 사용자의 데이터베이스에 테이블을 생성, 마이그레이션을 반영하거나 반영하지 않기 위해 사용
  --database DATABASE 옵션을 이용하여 데이터베이스를 지정      
  --fake-initial : 데이터베이스 테이블이 이미 존재하는 경우 앱 생성 초기 마이그레이션을 건너뛸 수 있도록        
  --run-syncdb : 마이그레이션 없이 앱에 대한 테이블 생성을 허용      

- sqlmigrate: 마이그레이션을 위한 SQL 구문을 보기 위해 사용
- showmigrations: 프로젝트 전체의 마이그레이션과 각각의 상태를 나열하기 위해 사용

<br>

> 이미 설계되어있는 데이터베이스 테이블을 django와 연결하여 사용할 때, 모델파일 생성하기.(마이그레이션 불필요, 관리되지 않는 모델)       
> python manage.py inspectdb > models.py
> class에 managed = False를 삭제 또는 변경해줘야 db모델관리 가능

<br>

[마이크레이션 스키마 문서 바로가기](https://docs.djangoproject.com/en/4.0/topics/migrations/)

<br>
<br>

## 자유로운 API 
자료에 접근할 수 있는 자유롭고 풍부한 Python API 이용 가능, API는 즉시 생성되며, 코드 생성이 필요없다.

- models을 만들면 Django는 자동으로 객체들을 만들고 읽고 수정하고 지울 수 있는 데이터베이스-추상화 API를 만들어냄.

<br>

### 객체 만들기
(모델은 ``mysite/app/models.py``에 있다고 가정)

- 모델 클래스가 데이터베이스의 테이블을 나타내고, 그 클래스의 인스턴스가 데이터베이스 테이블의 각 레코드를 표현하는 것
  
```python
# 객체 생성 예시
from app.models import Blog

b = Blog(name='Beatles Blog', tagline='All the latest Beatles news.')
b.save()

```
- 이 명령어는 SQL 명령어 중 **INSERT 구문을 작동**
- Django에서는 save().를 실행할 때까지 데이터베이스를 수정하지 않는다.         
  > 객체를 만들고 저장하는 것을 한 번에 하려면 create() 명령어를 이용      

<br>

### 객체 변경사항 저장하기
```python
# 객체를 변경한 후 저장
from app.models import Blog

b.name = 'New name'
b.save()

```
- SQL의 UPDATE 구문을 실행

<br>

### ForeignKey와 ManyToManyField 필드 저장
```python
#  Blog, Entry 모델이 있다고 가정.
#  This example updates the blog
from app.models import Blog, Entry

entry = Entry.objects.get(pk=1)
cheese_blog = Blog.objects.get(name="Cheddar Talk")
entry.blog = cheese_blog
entry.save()
```

ManyToManyField를 업데이트 하려면  add() 명령어를 사용
```python
# Author 인스턴스 joe를 entry 객체에 추가
from blog.models import Author
joe = Author.objects.create(name="Joe")
entry.authors.add(joe)
```

여러 레코드를 한 번에 추가
```python
john = Author.objects.create(name="John")
paul = Author.objects.create(name="Paul")
paul = Author.objects.create(name="Paul")
ringo = Author.objects.create(name="Ringo")
entry.authors.add(john, paul, george, ringo))
```

### 객체 조회
데이터베이스에 있는 객체들을 조회, QuerySet 만들어 이용
```python
Blog.objects
# <django.db.models.manager.Manager object at ...>

b = Blog(name='Foo', tagline='Bar')
b.objects
# error : "Manager isn't accessible via Blog instances."
# The Manager is the main source of QuerySets for a model.
# use the all() method on a Manager:
all_entries = Entry.objects.all()
```
- all() method는 데이터베이스 테이블 objects의 QuerySet을 리턴함.     
- filter(**kwargs)      
  ex) Entry.objects.filter(pub_date__year=2006)
- exclude(**kwargs)         
  ex) Entry.objects.filter(headline__startswith='What').exclude(pub_date__gte=datetime.date.today()
- get() method는 하나의 객채만 리턴              
  ex) one_entry = Entry.objects.get(pk=1)      
- objects의 QuerySet return에 limit 설정 가능        
  ex) Entry.objects.all()[:5]      
  ex) Entry.objects.all()[5:10]      
  ex) Entry.objects.all()[:10:2] - (every second object of the first 10)        
  ex) Entry.objects.order_by('headline')[0] - (this returns the first Entry in the database, after ordering entries alphabetically by headline)

<br>

> filter(), exclude() and get() 에서 필드 조회 가능.       
> field__lookuptype=value 형식               
> ex) Entry.objects.filter(pub_date__lte='2006-01-01')         
> SQL의 SELECT * FROM blog_entry WHERE pub_date <= '2006-01-01';         
> ex) Entry.objects.filter(blog_id=4)  __id로 ForeignKey 조회
>            
> 정확한 매치는 exact 이용
> ex) Entry.objects.get(headline__exact="Cat bites dog")      
> SQL의 SELECT ... WHERE headline = 'Cat bites dog';       
>          
> 대소문자 구분없이, iexact
> ex) Blog.objects.get(name__iexact="beatles blog")
> "Beatles Blog", "beatles blog", "BeAtlES blOG" 등 매치됨.
>          
> 대소문자를 구분, contains
> ex) Entry.objects.get(headline__contains='Lennon')
> SQL은 SELECT ... WHERE headline LIKE '%Lennon%';


[QuerySet API 더 자세히](https://docs.djangoproject.com/ko/4.0/ref/models/querysets/#queryset-api)        
[객체조회 더 자세히](https://docs.djangoproject.com/ko/4.0/topics/db/queries/)


<br>
<br>

## 동적인 관리자 인터페이스
admin site에 모델 객체를 등록
```python
# mysite/app/admin.py
from django.contrib import admin
from .models import Article

admin.site.register(Article)

```
- Django 앱을 생성하는 하나의 전형적인 작업 흐름은 일단 모델을 만들고 관리자 사이트를 올려서 가능한 빨리 작동할 수 있게 만드는 것

<br>
<br>

## URL 설계
Reporter/Article 예제
```python
# mysite/app/urls.py
from django.urls import path
from .views import year_archive, month_archive, article_detail

urlpatterns = [
    path('articles/<int:year>/', year_archive, name='year_archive'),
    path('articles/<int:year>/<int:month>/', month_archive, name='month_archive'),
    path('articles/<int:year>/<int:month>/<int:pk>/', article_detail, name='article_detail'),
]
```
- URL 경로들을 파이썬 콜백 함수들(views)로 연결       
- 만약 아무것도 매치하는 것이 없다면, Django 는 특수한 사례인 404 view 를 호출      
- ex) 사용자가 URL 《/articles/2005/05/39323/》로 요청 하면,       
  app.views.article_detail(request, year=2005, month=5, pk=39323) 함수 호출.
  
<br>
<br>

## 뷰 작성
요청된 페이지의 내용을 담고 있는 HttpResponse 객체를 반환하거나, Http404 같은 예외를 발생        
일반적으로 파라미터들에 따라 데이터를 가져오며, 템플릿을 로드하고 템플릿을 가져온 데이터로 렌더링함.         
```python
# mysite/app/views.py
from django.shortcuts import render
from .models import Article

def year_archive(request, year):
    a_list = Article.objects.filter(pub_date__year=year)
    context = {'year': year, 'article_list': a_list}
    return render(request, 'app/year_archive.html', context)

```

[장고 템플릿 시스템 자세히 보기](https://docs.djangoproject.com/ko/4.0/topics/templates/)

<br>
<br>

## 템플릿 작성
위의 코드는 app/year_archive.html 템플릿을 로드
- 장고는 템플릿들중 중복을 최소화할 수 있게 하는 템플릿 검색 경로를 가지고 있음. project에서 settings.py에 DIRS 템플릿을 확인하기 위한 디렉토리의 목록을 명시함.
- 만약 첫번째 디렉토리에 템플릿이 존재하지 않으면, 두번째 디렉토리, 그 외 디렉토리를 점검.

"base.html" 기본 템플릿 예시
```html
{% raw %} {% load static %} {% endraw %}
<html>
<head>
    <title> {% raw %} {% block title %}{% endblock %} {% endraw %} </title>
</head>
<body>
    {% raw %} <img src="{% static 'images/sitelogo.png' %}" alt="Logo"> {% endraw %}
    {% raw %} {% block content %}{% endblock %} {% endraw %}
</body>
</html>
```
- settings.py에서 템플릿과 마찬가지로 static DIR 경로 설정 가능.     
- 간단하게 사이트를 정의하고 하위 템플릿이 채울 수 있는 **구멍( {% raw %} {% block title %}{% endblock %}, {% block content %}{% endblock %} {% endraw %} )** 들을 제공함.   
- {% raw %} {% extends "base.html" %} {% endraw %} 이용하여 다른 템플릿파일에 상속 가능.      
- {% raw %} {% include "base.html" %} {% endraw %} 이용하여 다른 템플릿파일에 포함 가능.        