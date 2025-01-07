---
layout: post
title: BALLMFS(三) :Coding Attention Mechanisms
categories: [-07 Machine Learning]
tags: [LLM]
number: [-11.1]
fullview: false
shortinfo: BALLMFS(三) :Coding Attention Mechanisms

---
目录
{:.article_content_title}


* TOC
{:toc}

---
{:.hr-short-left}

## 1 Understanding Large Language Models ##

### 1.1 What is transformer
> **Transformer**: consists of two main parts:
1. Encoder: Processes the input sequence.
2. Decoder: Generates the output sequence (used in tasks like translation).

However, in models like GPT (Generative Pre-trained Transformer), only the decoder is used, while in models like BERT (Bidirectional Encoder Representations from Transformers), only the encoder is used.

It is introduced in papar [Attention is All You Need](https://arxiv.org/abs/1706.03762).

{: .img_middle_mid}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-01-BALLMFS(一) :Understanding Large Language Models/Transformer.jpg)


### 1.2 What is GPT

> **GPT**: Generative Pretrained Transformer, introduced in [improving Language Understanding by Generative Pre-Training (2018) by Radford et al.](http://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), good at simple next-word prediction task.

> **GPT-3**: GPT-3 is a scaled-up version of this model that has more parameters and was trained on a larger dataset

> **InstructGPT**: Finetuning with Human Feedback To Follow Instructions based upon GPT-3, can carry out other tasks such as spelling correction, classification, or language translation.

> **Emergent Behavior**: GPT is not trained and designed for translation but only for next-word prediction. It turns out that the fact that GPT is good at translation task. This capability isn't explicitly taught during training but emerges as a natural consequence of the model's exposure to vast quantities of multilingual data in diverse context

{: .img_middle_mid}
![GPT]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-01-BALLMFS(一) :Understanding Large Language Models/GPT.jpg)


### 1.3 Let's build GPT from scrach

{: .img_middle_lg}
![Steps]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-01-BALLMFS(一) :Understanding Large Language Models/Steps.jpg)





## 6 参考资料 ##
- [Deep Learning](https://book.douban.com/subject/26883982/);