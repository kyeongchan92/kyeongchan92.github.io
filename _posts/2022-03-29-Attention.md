---
title:  "Attention 논문(2014) 정리"
excerpt: "Attention이 뭔지 정확히 이해하기"
layout: single
categories:
  - Attention
tags:
    - Attention
---

# 원 논문
https://arxiv.org/abs/1409.0473

Bahdanau, D., Cho, K., & Bengio, Y. (2014). **Neural machine translation by jointly learning to align and translate.** arXiv preprint arXiv:1409.0473.

# Abstract
**뉴럴 기계번역**은 기계번역에 있어서 최근에 제안되었다. 기존의 통계적 방법의 기계번역과는 다르게, **뉴럴 기계번역**은 단일 신경망을 만들고 결합적으로(jointly) 튜닝시키도록 해서 번역 성능을 최대화하려고 한다. 뉴럴 기계번역에서 최근에 제안된 모델들은 인코더-디코더 구조이며, 소스(source) 문장을 고정길이의 벡터로 인코딩하고 디코더가 그 벡터로부터 번역을 생성한다. 본 논문에서는, 고정길이의 벡터가 보틀넥이라고 추측하여, 이를 늘리는 것을 제안한다. 이를 위해, 하드한 구조를 쓰는 것이 아니라, 타겟 단어를 예측하는데 연관된 소스 문장의 파트들을 모델이 자동으로 찾게 하는 soft한 방식으로 만든다. 이 새로운 접근법으로 우리는 SOTA의 영어-프랑스어 번역 태스크에서 좋은 번역 성능을 달성했다.

# 1 INTRODUCTION
뉴럴 기계번역은 기계번역에 있어서 새로운 접근법이며, Kalchbrenner와 Blunsom(2013), Sutskever et al. (2014) and Cho et al. (2014b)에 의해 최근에 제안되었다. 기존의 구문 기반의 번역시스템은 작고 많은 sub-component로 구성돼있고 각각 튜닝되었었다. 그러나 뉴럴 기계번역은 이와 다르게 단일의 큰 신경망을 이용한다.

대부분의 뉴럴 기계번역 모델들은 인코더-디코더 구조를 따른다.

# 2 BACKGROUND : 뉴럴 기계번역
## RNN ENCODER-DECODER
RNN 인코더-디코더(Cho et al, 2014a)라고 하는, 기본이 되는 프레임워크를 짧게 설명하겠다. 인코더-디코더 구조에서 인코더는 인풋 문장을 읽어서 벡터 $c$로 만든다. 인풋 문장은 $\mathbf{x}=(x_1, \cdots, x_{T_x})$라고 쓸 수 있다.
![image](https://user-images.githubusercontent.com/25517592/160550376-09ca6576-7cdb-4354-8e0b-c814f2477a9f.png)

가장 보편적인 접근법은 인코더로서 RNN을 사용하는것이다.
$$h_t=f(x_t, h_{t-1})$$
$$c=q(\left\{ h_1, \cdots, h_{T_x} \right\})$$
여기서 $h \in \mathbb{R}^n$은 $t$시점에서의 $n$차원의 hidden state이다.그리고 $c$는 이 hidden state들의 시퀀스들로 이루어진다. $f$와 $q$는 비선형 함수이다. Sutskever ea al.(2014)에서는 $f$로 LSTM을, $q$로는 마지막 hidden state를 뽑는 함수를 사용했다.

![image](https://user-images.githubusercontent.com/25517592/160554668-d49e51d6-32f2-4857-8eb7-bf64ebd4fa01.png)

디코더는 context 벡터 $c$와 이전에 예측되었던 모든 단어들 $\left\{ y_1, \cdots, y_{t'-1} \right\}$이 주어졌을 때 다음 단어 $y_{t'}$를 예측하기 위해 학습된다. 

# LEARNING TO ALIGN AND TRANSLATE

여기서는 우리의 새로운 뉴럴 기계번역 구조를 소개한다. 새로운 아키텍쳐는 bidirectional RNN의 인코더와 디코더로 구성된다.

## 3.1 DECODER: GENERAL DESCRIPTION



