---
layout: post
title: "放回采样最终不同样本数量"
description: ""
category: "math"
tags: ["数学", "采样"]
published: true
---

* 目录
{:toc}
机器学习很多场景中会用到放回采样，比如bagging方法。采样后的数据集会有一些数据重复，一些数据缺失，从$$N​$$个样本中采样$$K​$$个样本，不同样本数量的期望为$$U(K)=N(1-\left(\frac{N-1}{N}\right)^K)​$$。怎么来的呢？这里给出简单的证明。

首先，显然有$$U(1)=1$$；其次，设从$$N$$个样本中采样$$k-1$$个样本，不同样本数量的期望为$$U(k-1)$$，则第$$k$$个样本是未曾抽到的样本的概率为$$1-\frac{U(k-1)}{N}$$，所以$$U(k)=1+\frac{N-1}{N}U(k-1)$$$$=1+\frac{N-1}{N}+\left(\frac{N-1}{N}\right)^2+\cdots+\left(\frac{N-1}{N}\right)^{k-1}$$，根据[等比数列](https://en.wikipedia.org/wiki/Geometric_progression)求和公式得$$U(K)=N(1-\left(\frac{N-1}{N}\right)^K)$$。

对于一种特殊情况，当$$K=N$$且$$N$$足够大时，则有最终不同样本数量是原始样本数量的期望为$$(1-\frac{1}{e})$$，大约是$$\frac{2}{3}$$。

可以通过一段Python程序来验证结论的正确性：

```python
import random, math

S = set()
N = 10000000
[S.add(random.randint(1,N)) for i in range(N)]
print(len(S), int(N*(1-1/math.e)))
```

我得到的输出结果为

```bash
(6321214, 6321205)
```

当然，你的运行结果可能和上面有所差别。