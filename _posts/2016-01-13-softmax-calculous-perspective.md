---
layout: post
title: "从数学分析的角度来看Softmax"
description: ""
category: "machine_learning"
tags: ["数值计算","机器学习"]
published: true
---

Softmax是机器学习中最常用的输出函数之一，网上有很多资料介绍它是什么以及它的用法，但却没有资料来介绍它背后的原理。本文首先简单地介绍一下Softmax，然后着重从数学分析的角度来分析一下它背后的原理。

分类问题是监督学习中最重要的问题之一，它试图根据输入$$\bf{x}$$来预测对应标签$$y$$的概率。Softmax便是计算标签概率的重要工具之一：

$${\bf p}=\rm{softmax}({\bf a})\Leftrightarrow p_i=\frac{\exp({a_i})}{\sum_j\exp({a_j})}$$

其中$$a_i$$是模型对于第$$i$$个分类的输出。接下来简单地证明一下：通过对数最大似然以及梯度下降方法可以使$$p_i$$逼近第$$i$$个分类的真实概率。对数最大似然中的损失函数为$$L_{NLL}({\bf p},y)=-\log p_y$$，对它关于$${\bf a}$$求导得：

$$\frac{\partial}{\partial a_k}L_{NLL}({\bf p},y)=\frac{\partial}{\partial a_k}(-\log p_y)=\frac{\partial}{\partial a_k}(-a_y+\log\sum_j e^{a_j})$$

$$=-{\bf 1}_{y=k}+\frac{e^{a_k}}{\sum_j{e^{a_j}}}=p_k-{\bf 1}_{y=k}$$

即$$\frac{\partial}{\partial {\bf a}}L_{NLL}({\bf p},y)=({\bf p}-{\bf e}_y)$$，其中$${\bf e}_y=[0,\cdots,0,1,0,\cdots,0]$$是一个向量，除了位置$$y$$为1之外全是0。相同$${\bf x}$$的样本对应相同的$${\bf a}$$，我们可以看到，随着越来越多样本参与梯度下降，$$p_i$$会逼近第$$i$$个分类的真实概率，即$${\bf p}=\mathbb{E}[{\bf e}_{y}\|{\bf x}]$$，因为$$\lim\limits_{N\to\infty}\frac{1}{N}\sum\limits_{i=1}^N({\bf p}-{\bf e}_y^{(i)})=0$$，其中$$\lim\limits_{N\to\infty}\frac{1}{N}\sum\limits_{i=1}^N{\bf e}_y^{(i)}$$是真实概率。

从收敛速度方面，对数最大似然与梯度下降在Softmax身上简直是绝配。对于一个输入为$${\bf x}$$的样本，假设它的真实分类是$$i$$，对于模型的第$$j(j\neq i)$$个输出有$$\frac{\partial}{\partial a_j}L_{NLL}({\bf p}, y)=p_j$$，如果$$p_j\approx 0$$（即模型认为不太可能是分类$$j$$，预测结果与实际相符），梯度接近0，会进行很小的修正，如果$$p_j\approx 1$$（即模型非常有信心地预测是分类$$j$$，预测结果与实际相反），梯度接近1，会进行很大的修正。另外，对于模型的第$$i$$个输出有$$\frac{\partial}{\partial a_i}L_{NLL}({\bf p}, y)=1-p_i$$，如果$$p_i\approx 0$$（即模型认为不太可能是分类$$i$$，预测结果与实际相反），梯度接近1，会进行很大的修正，如果$$p_i\approx 1$$（即模型非常有信心地预测是分类$$i$$，预测结果与实际相符），梯度接近0，会进行很小的修正。综上，在Softmax上使用对数最大似然作为损失函数，梯度下降情况非常理想——预测错误时修正大，预测正确时修正小。

当然也有人在Softmax上尝试其他损失函数，比如最有名的最小二乘。结果是两者并不搭，因为在最小二乘下模型如果预测完全错误时修正也会非常小。设$${\bf y}={\bf e}_i$$（注意这里的$${\bf y}$$是黑体），对最小二乘$$L_2({\bf p}({\bf a}),{\bf y})=\|\|{\bf p}({\bf a})-{\bf y}\|\|^2$$关于$$a_i$$（假设$$i$$是正确类别）求导得

$$\frac{\partial}{\partial a_i}L_2({\bf p}({\bf a}),{\bf y})=\frac{\partial{L_2({\bf p}({\bf a}), {\bf y})}}{\partial {\bf p}({\bf a})}\frac{\partial {\bf p}({\bf a})}{\partial a_i}$$

$$=\sum_{j\neq i}2(p_j-{\bf y}_j)p_j(0-p_i)+2(p_i-{\bf y}_i)p_i(1-p_i)$$

如果对于正确类别$$i$$模型的预测是$$p_i\approx 0$$（与实际强烈不符），显然有$$\frac{\partial}{\partial a_i}L_2({\bf p}({\bf a}),{\bf y})\approx 0$$，也就是说梯度下降对模型几乎不修正，可见Softmax搭配最小二乘的梯度下降情况并不好。

PS：Softmax还有一个重要性质是平移不变性，即$${\rm softmax}({\bf a})={\rm softmax}({\bf a}+b)$$，因为$$\frac{\exp({a_j+b})}{\sum_k\exp({a_k+b})}=\frac{\exp({a_j})}{\sum_k\exp({a_k})}$$。由于平移不变性的存在，模型只需要学到$${\bf a}$$中元素的相对大小，而不需要学到绝对大小。另外，我们还可以根据$${\rm softmax}({\bf a})={\rm softmax}({\bf a}-\max_ia_i)$$有效地减少计算误差。

综上所述，首先，Softmax的确可以表示概率，且随着样本的增多通过对数最大似然与梯度下降可以无限逼近真实概率值；其次，Softmax与对数最大似然这一组合在梯度下降中有很好的修正速度；最后，因为平移不变性，我们只需要关心模型不同类别输出间的相对大小，不需要关心绝对大小。