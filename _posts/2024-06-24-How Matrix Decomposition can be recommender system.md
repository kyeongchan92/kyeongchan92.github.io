---
title: Matrix Decomposition은 어떻게 추천시스템이 될 수 있을까?
description:
categories:
tags:
---

127p Example 4.14 (영화 평점과 유저)
![0](/assets/images/how matrix decomposition can be recommender system/rating_matrix.png)

세 명의 유저 Ali, Beatrix, Chandra가 있고, 4개의 영화 Star Wars, Blade Runner, Amelie, Delicatessen이 있다고 해보자.
평점은 0(worst)에서 5(best) 사이이고, 데이터는 $$\mathbf{A} \in \mathbb{R}^{4 \times 3}$$으로 나타내어진다.
각 row는 영화를, column은 유저를 나타낸다.
column 벡터는 $$x_{\text{Ali}}$$, $$x_{\text{Beatrix}}$$, $$x_{\text{Chandra}}$$로 나타낼 수 있다.

$$\mathbf{A}$$를 SVD를 사용해 분해(Factoring)하면 유저와 영화의 관계를 알 수 있다! Left-singular 벡터를 $$u_i$$, right-singular 벡터를 $$v_j$$라고 하자.


이어서..

참고 : Mathematics for Machine Learning