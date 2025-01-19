---
layout: post
title: BALLMFS(二) :Working with Text Data
categories: [-07 Machine Learning]
tags: [LLM]
number: [-11.1]
fullview: false
shortinfo: BALLMFS(二) :Working with Text Data

---
目录
{:.article_content_title}


* TOC
{:toc}

---
{:.hr-short-left}

## 1. Tokenize and Embedding ##


Work Flow

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-02-BALLMFS(二) :Working with Text Data/WorkFlow.jpg)


Detail

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-02-BALLMFS(二) :Working with Text Data/TokenizationAndEmbedding.jpg)


## Appendix A

### A.1 Dataset
{% highlight python linenos %}
# datasets: enumerator over data records
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)                         
        for i in range(0, len(token_ids) - max_length, stride):   
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):                                            
        return len(self.input_ids)
    def __getitem__(self, idx):                                   
        return self.input_ids[idx], self.target_ids[idx]

{% endhighlight %}


### A.2 Dataloader
{% highlight python linenos %}
# dataloader: manage the batch over dataset's records for training
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                   
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,                                      
        num_workers=0                                             
    )
    return dataloader
{% endhighlight %}

### A.3 position and integration

{% highlight python linenos %}

# position: position is important for maintaning the position info for words

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

{% endhighlight %}

a

## 6 参考资料 ##
- [Deep Learning](https://book.douban.com/subject/26883982/);