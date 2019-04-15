---
layout: post
title: "归一化与标准化"
description: ""
category: "machine_learning"
tags: ["归一化", "标准化"]
published: true
redirect_from: 
  - /2016/01/normalization-and-standardization
---

在机器学习和数据挖掘中，经常会听到两个名词：归一化（Normalization）与标准化（Standardization）。它们具体是什么？带来什么益处？具体怎么用？本文来具体讨论这些问题。

## 一、是什么

### 1. 归一化

常用的方法是通过对原始数据进行线性变换把数据映射到[0,1]之间，变换函数为：

$$x'=\frac{x-\min}{\max-\min}$$

其中$$\min$$是样本中最小值，$$\max$$是样本中最大值，注意在数据流场景下最大值与最小值是变化的。另外，最大值与最小值非常容易受异常点影响，所以这种方法鲁棒性较差，只适合传统精确小数据场景。

### 2. 标准化

常用的方法是z-score标准化，经过处理后的数据均值为0，标准差为1，处理方法是：

$$x'=\frac{x-\mu}{\sigma}$$

其中$$\mu$$是样本的均值，$$\sigma$$是样本的标准差，它们可以通过现有样本进行估计。在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景。

## 二、带来什么

归一化的依据非常简单，不同变量往往量纲不同，归一化可以消除量纲对最终结果的影响，使不同变量具有可比性。比如两个人体重差10KG，身高差0.02M，在衡量两个人的差别时体重的差距会把身高的差距完全掩盖，归一化之后就不会有这样的问题。

标准化的原理比较复杂，它表示的是原始值与均值之间差多少个标准差，是一个相对值，所以也有去除量纲的功效。同时，它还带来两个附加的好处：均值为0，标准差为1。

均值为0有什么好处呢？它可以使数据以0为中心左右分布（这不是废话嘛），而数据以0为中心左右分布会带来很多便利。比如在去中心化的数据上做[SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)分解等价于在原始数据上做[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)；机器学习中很多函数如[Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)、[Tanh](https://en.wikipedia.org/wiki/Hyperbolic_function#Tanh)、[Softmax](https://en.wikipedia.org/wiki/Softmax_function)等都以0为中心左右分布（不一定对称）。

标准差为1有什么好处呢？这个更复杂一些。对于$$x_i$$与$$x_{i'}$$两点间距离，往往表示为

$$D(x_i,x_{i'})=\sum\limits_{j=1}^pw_j\cdot d_j(x_{ij},x_{i'j});\sum\limits_{j=1}^pw_j=1$$

其中$$d_j(x_{ij},x_{i'j})$$是属性$$j$$两个点之间的距离，$$w_j$$是该属性间距离在总距离中的权重，注意设$$w_j=1,\forall j$$并不能实现每个属性对最后的结果贡献度相同。对于给定的数据集，所有点对间距离的平均值是个定值，即

$$\bar{D}=\frac{1}{N^2}\sum\limits_{i=1}^N\sum\limits_{i'=1}^ND(x_i,x_{i'})=\sum\limits_{j=1}^pw_j\cdot \bar{d}_j$$

是个常数，其中

$$\bar{d}_j=\frac{1}{N^2}\sum\limits_{i=1}^N\sum\limits_{i'=1}^Nd_j(x_{ij}, x_{x'j})$$

可见第$$j$$个变量对最终整体平均距离的影响是$$w_j\cdot \bar{d}_j$$，所以设$$w_j\sim 1/\bar{d}_j$$可以使所有属性对全数据集平均距离的贡献相同。现在设$$d_j$$为欧式距离（或称为二范数）的平方，它是最常用的距离衡量方法之一，则有

$$\bar{d_j}=\frac{1}{N^2}\sum\limits_{i=1}^N\sum\limits_{i'=1}^N(x_{ij}-x_{i'j})^2=2\cdot var_j$$

其中$$var_j$$是$$Var(X_j)$$的样本估计，也就是说每个变量的重要程度正比于这个变量在这个数据集上的方差。如果我们让每一维变量的标准差都为1（即方差都为1），每维变量在计算距离的时候重要程度相同。

## 三、怎么用

在涉及到计算点与点之间的距离时，使用归一化或标准化都会对最后的结果有所提升，甚至会有质的区别。那在归一化与标准化之间应该如何选择呢？根据上一节我们看到，如果把所有维度的变量一视同仁，在最后计算距离中发挥相同的作用应该选择标准化，如果想保留原始数据中由标准差所反映的潜在权重关系应该选择归一化。另外，标准化更适合现代嘈杂大数据场景。