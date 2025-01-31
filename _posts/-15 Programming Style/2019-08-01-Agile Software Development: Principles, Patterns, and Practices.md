---
layout: post
title: Agile Software Development：Principles, Patterns, and Practices
categories: [-15 Programming Style]
tags: [Agile]
number: [-2.1]
fullview: false
shortinfo: Agile Software Development：Principles, Patterns, and Practices
---
目录
{:.article_content_title}


* TOC
{:toc}

---
{:.hr-short-left}

## 1 Agile Development

### CH1 Agile Practices

Deliver software early and often results a higher quality at the end.

Agile development has several implementations:

- SCRUM
- Extreme Programming
- Crystal
- Feature Driven Development
- Adaptive Software Development(ADP)

### CH4 Test

Unit test are white box to ensure each unit works as expected;
Accpetance(Integertion) test are black box to ensure the whole works as expected;

## 2 Agile Design

## 3 The Payroll Case Study

### 3.1 Design Patterns

#### CH13 Command

> Command Pattern: a behavioural design pattern in which an object is used to represent and encapsulate all the information needed to call a method at a later time. The information includes the method name, the object that owns the method and values for the method paramters. `Objective C`'s [performSelector](https://developer.apple.com/documentation/objectivec/nsobject/1411637-performselector) method is an implementation of command pattern. Command pattern 的本质是封装了方法调用的所有信息(object, method, paramters)到一个Command Object, 然后其`do()`或者`undo()`方法调用就执行`object.method(paramters)`，这样使得`command.do()`的使用者无需知道`do()`里面的细节就可以调用。

{: .img_middle_hg}
![NodeJS]({{site.url}}/assets/images/posts/-15_Programming_Style/Agile Software Development/Command_DesignPattern.png)

#### CH14 Template and Strategy Pattern

> Template and Strategy Patterns: both Template and Strategy pattern are used to separate high level algo from its detail implementation.
Strategy adds another layer of indirection, which allow detail implmentation to be reused for other high level algo

#### CH15 Facade and Mediator Pattern

> Facade: a simplified API to encapsulate complicated class

> Mediator: encapsulate various objects interaction so the caller of Mediator only needs to know Mediator without need to know the various objects within the mediator.

#### CH16 Singleton and Monostate Pattern

> Singleton: always return the same private instance via static factory method

> Monostate: turn instance var to static var.

#### CH17 Null Object Pattern

> Null Object: eliminate the null check in  `if(e != null && e.shouldPay())` to `if(e.shouldPay())` by have a `Employee` interface, with two implementations `EmployeeNull` and `EmployeeImplementation`, the former implements all the method defined in `Employee` interface, mostly by doing nothing.



## 4 Packaging the Payroll System

## 5 The Weather Station Case Study

## 6 The ETS Case Study


## A1 UML Notation

## A2 Miscellaneous


{: .img_middle_hg}
![NodeJS]({{site.url}}/assets/images/posts/-14_Backend/2015-10-09-Backend：Server Architecture/NodeJS.png)

## 2 参考资料 ##

- [The Architecture of Open Source Applications: Nginx](http://www.aosabook.org/en/nginx.html);


