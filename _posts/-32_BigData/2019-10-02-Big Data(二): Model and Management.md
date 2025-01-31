---
layout: post
title: Text Retrieval and Search Engines(一)： Overview
categories: [Data Mining]
tags: [Data Mining, Text Retrieval, Serach Engine]
number: [-3.2]
fullview: false
shortinfo: 文本检索是搜索引擎最核心的技术。本系列通过 Coursera上UIUC Prof. ChengXiang Zhai的《Text Retrieval and Search Engines》课程，对文本检索进行全面的了解和学习。该课程是Data Mining系列课程的第二门课。本文是第一周的一个Overview。
---
目录
{:.article_content_title}


* TOC
{:toc}

---
{:.hr-short-left}

## 1. NLP：Natural Language Processing，自然语言处理 ##

### 1.1 什么是NLP ###

> **自然语言处理(NLP)**：计算机对自然语言的处理。

由于自然语言在人类沟通过程中既省略了很多**常识(common sense)**，又保留了很多**模糊性(ambiguities)**，使得NLP变得异常困难。

### 1.2 NLP技术前沿 ###

NLP技术分以下四个方面：

1. Lexical Analysis (词性分析：是名词还是动词？)。
2. Syntactic Analysis (语法分析：介词短语还是谓语？)。
3. Semantics Analysis (语义分析：名字之间的关系，去模糊性，情绪)。
4. Pragmatic Analysis (语用分析：说这句话的目的)。

{: .img_middle_lg}
![state of art NLP]({{site.url}}/assets/images/posts/2015-11-01/state of art NLP.png)

### 1.3 NLP for Text Retrieval ###

> **Bag of Words(词袋)**： sufficient for most search tasks.

## 2. Text Access，文本访问 ##

> **Text Access(文本访问)**：the proccess of accessing **Small Releavant Text Data** from **Big Text Data**.


### 2.1 Push Mode：推送模式 ###

> **Push Mode**: **系统**拥有足够的用户信息，主动推荐

### 2.2 Pull Mode：牵引模式 ###

> **Pull Mode**: **用户**主动搜索。

牵引模式又可分为：

1. **Querying(请求)**：用户输入关键词，
2. **Browsing(浏览)**：用户根据文本结构导航进来，通常在不知道关键词的情况下发生。

## 3. Text Retrieval Problem，文本检索问题 ##

### 3.1 什么是Text Retrieval ###

> **Text Retrieval**：Big Text Data exists， user gives a Query， Search Engine return relevant Small Text Data，即上面所说的Push Mode中的Querying.

### 3.2 Text Retrieval VS Database Retrieval###


{: .img_middle_mid·}
![state of art NLP]({{site.url}}/assets/images/posts/2015-11-01/TF VS DF.png)

### 3.3 Foundamental Model of Text Retrieval ###

{: .img_middle_mid}
![state of art NLP]({{site.url}}/assets/images/posts/2015-11-01/Formulation of TR.png)




### 3.4 Document Selection VS Document Ranking ###


{: .img_middle_hg}
![state of art NLP]({{site.url}}/assets/images/posts/2015-11-01/Document Selection VS Document Ranking.png)





## 4. Text Retrieval Methods，文本检索方法 ##


## 5. VSM：Vector Space Model，向量空间模型 ##


### 5.1 VSM simple instantiation； 向量空间模型简单实例 ###

{: .img_middle_lg}
![Programming Paradigm]({{site.url}}/assets/images/posts/2015-10-01/Programming Paradigm.png)


{% highlight scala linenos %}
 def factorial(n:Int):Int = {
    def factLoop(n:Int, acc:Int):Int = {
        if (n==0) acc
        else factLoop(n-1, acc * n)
    }
    factLoop(n,1)
 }                                                //> factorial: (n: Int)Int

 factorial(4)                                     //> res2: Int = 24
{% endhighlight %}





{: .img_middle_lg}
![Assignment]({{site.url}}/assets/images/posts/2015-10-01/assignment.png)


具体代码见[这里](https://github.com/shunmian/-2_Functional-Programming-in-Scala)。


## 总结 ##



## 参考资料 ##
- [《Structure and Interpretation of Computer Programs》](https://mitpress.mit.edu/sicp/full-text/book/book.html);
- [Martin Odersky: Scala with Style](https://www.youtube.com/watch?v=kkTFx3-duc8);
- [SF Scala: Martin Odersky, Scala -- the Simple Parts](https://www.youtube.com/watch?v=ecekSCX3B4Q);


