---
title: LDA(Latent Dirichlet Allocation, 잠재 디리클레 할당) 논문 정리
description:
categories:
tags:
---

이 논문은 2003년에 나와서 현재 기준 약 5만 인용수를 자랑한다. LDA는 대표적인 토픽모델링 알고리즘이다. 
논문은 다음과 같다 : [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf){:target="_blank"}

---

# Notation and terminology

word는 $$w$$, document는 $$\mathbf{w}$$, corpus는 $$D$$라고 나타낸다. 총 $$V$$개의 word가 있다고 하자.

document는 $$w$$의 시퀀스이다. document 1개의 단어 수가 $$N$$개라고 하자. $$\mathbf{w} = (w_1, w_2, \cdots , w_N)$$이다. 
corpus는 document의 집합이다. corpus가 $$M$$개의 document로 구성돼있다고 하자. $$D = \left\{ \mathbf{w}_1, \mathbf{w}_2, \cdots, \mathbf{w}_M \right\}$$이다.


# Latent Dirichlet allocation

LDA는 코퍼스에 대한 생성 확률 모델(generative probabilistic model)입니다. 기본 아이디어는 도큐먼트가 잠재 토픽들(latent topics)의 무작위 혼합으로 구성되어 있고, 각 토픽은 word의 분포에 의해 특성을 갖는다는 것입니다.
LDA는 다음을 가정합니다.

1. Choose $$N$$ $$\sim$$ Poisson($$\xi$$).
2. Choose $$\Theta$$ $$\sim$$ Dir($$\alpha$$).
3. For each of the $$N$$ words $$w_n$$:
   1. Choose a topic $$z_n$$ $$\sim$$ Multinomial($$\Theta$$).
   2. Choose a word $$w_n$$ from $$p(w_n \mid z_n, \beta)$$, a multinomial probability conditioned on the topic $$z_n$$.


첫째, 디리클레 분포의 차원이자 토픽 개수(토픽 변수 $$z$$의 차원) $$k$$는 알려진 값이거나 고정된 값이다.
둘째, 단어 확률은 $$k \times V$$ 행렬 $$\beta$$에 의해 파라미터화된다. $$\beta_{ij}=p(w^j=1 \mid z^i=1)$$ 이다.
마지막으로, 포아송 가정은 이후엔 중요하지 않으며 더 현실적인 문서 길이가 사용될 것이다. 추가적으로, $$N$$은 $$\Theta$$나 $$\mathbf{z}$$같은 다른 데이터 생성 변수와는 독립적이다.
이는 보조적인 변수이고 뒤에서는 이 변수의 무작위성을 무시할 예정이다.

$$k$$차원의 디리클래 랜덤변수 $$\Theta$$는 $$(k-1)$$-simplex 중 하나의 값을 갖는다.

$$p(\theta)$$

$$\alpha$$를 $$k$$-vector라고 해보자. 원소 $$\alpha_i > 0$$로 구성되어 있다.
그리고 $$\Gamma$$는 Gamma 함수라고 하며 다음과 같다.

$$\Gamma(n)=(n-1)!$$






















































