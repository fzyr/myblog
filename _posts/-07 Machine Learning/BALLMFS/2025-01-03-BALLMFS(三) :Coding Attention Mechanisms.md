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

## 1 self-attention ##

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/0_CodingAttentionMechanisms.png)

### 1.1 simplified self-attention

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/1_simplifiedSelfAttension.png)

{% highlight python linenos %}
import torch

#3.1 simplified self-attention

inputs = torch.tensor(
  [
    [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55] # step
  ]
)
print('inputs', inputs)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
  # attn_scores_2[i] = torch.dot(x_i, query)
  attn_scores_2[i] = torch.dot(query, x_i)

print('attn_scores_2', attn_scores_2)

res = 0
for idx, element in enumerate(inputs[0]):
  res += inputs[0][idx] * query[idx]

print('res', res)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
# attn_weights_2_naive = softmax_naive(attn_scores_2)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = inputs[1]

context_vect_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
  context_vect_2 += attn_weights_2[i] * x_i

print(context_vect_2)


attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)


attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

{% endhighlight %}


### 1.2 trainable self-attention with Q, K, V weights

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/2.1_trainableSelfAttentionWIthQKV.png)

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/2.2_trainableSelfAttentionWIthQKV.png)

{% highlight python linenos %}

#3.2 self-attention with trainable weight matrics

x_2 = inputs[1]                                                   
d_in = inputs.shape[1]                                            
d_out = 2     


torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)
print(key_2)
print(value_2)

print(torch.tensor([1,2,3])@torch.tensor([[4,40], [5,50], [6,60]]))
print("keys.shape:", torch.tensor([1,2,3]).shape)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)


keys_2 = keys[1]                                                  #A
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)

d_k = keys.shape[-1]
print('d_k', d_k)
attn_weights_2 = torch.softmax(attn_scores_2, dim=-1)
print(attn_weights_2)
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)


context_vec_2 = attn_weights_2 @ values
print(context_vec_2)



class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

{% endhighlight %}

### 1.3 casula/mask self-attention with no future leakage and dropoff

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/3.1_casualSelfAttention.png)

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/3.2_casualSelfAttention.png)

{% highlight python linenos %}
class CausalAttention(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
    super().__init__()
    self.d_out = d_out
    self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.dropout = torch.nn.Dropout(dropout)                        #A
    self.register_buffer(
        'mask',
        torch.triu(torch.ones(context_length, context_length),
        diagonal=1)
    ) #B
  def forward(self, x):
    b, num_tokens, d_in = x.shape                             #C
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
    attn_scores = queries @ keys.transpose(1, 2)              #C
    attn_scores.masked_fill_(                                 #D
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = attn_weights @ values
    return context_vec

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)   
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
{% endhighlight %}

### 1.4 multiple casula/mask self-attention 

{: .img_middle_lg}
![Transformer]({{site.url}}/assets/images/posts/-07_Machine Learning/BALLMFS/2025-01-03-BALLMFS(三) :Coding Attention Mechanisms/4.1_multiHeadSelfAttention.png)

{% highlight python linenos %}
#3.4 multi-head self attention
class MultiHeadAttentionWrapper(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    self.heads = torch.nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
  def forward(self, x):
    return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads                       
    self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = torch.nn.Linear(d_out, d_out)                  
    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer( 'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)                                      
    queries = self.W_query(x)                                 
    values = self.W_value(x)                                  
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
    values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

    keys = keys.transpose(1, 2)                               
    queries = queries.transpose(1, 2)                         
    values = values.transpose(1, 2)                           
    attn_scores = queries @ keys.transpose(2, 3)  
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    
    attn_scores.masked_fill_(mask_bool, -torch.inf)           
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = (attn_weights @ values).transpose(1, 2) 
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)                  
    return context_vec

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print('context_vecs', context_vecs)
print("context_vecs.shape:", context_vecs.shape)
{% endhighlight %}



## 6 参考资料 ##
- [Deep Learning](https://book.douban.com/subject/26883982/);