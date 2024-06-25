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
각 row는 영화를, column은 유저를 나타낸다.****
column 벡터는 $$x_{\text{Ali}}$$, $$x_{\text{Beatrix}}$$, $$x_{\text{Chandra}}$$로 나타낼 수 있다.

$$\mathbf{A}$$를 SVD를 사용해 분해(Factoring)하면 유저와 영화의 관계를 알 수 있다! Left-singular 벡터를 $$u_i$$, right-singular 벡터를 $$v_j$$라고 하자.

가정 하나를 만든다. '유저의 선호도는 $$v_j$$의 선형 조합으로 표현될 수 있음.' 또, '영화의 호감도는 $$u_i$$'의 선형 조합으로 표현될 수 있다.'

그러므로, SVD의 정의역에 존재하는 벡터는 유저로, 공역에 존재하는 벡터는 영화로 해석될 수 있다.

left-singular vector $$u_1$$은 science fiction에 큰 절대값을 갖고, 큰 first singular value를 갖는다. 그림에서 빨간색으로 칠해진 곳이다.

이어서..

참고 : Mathematics for Machine Learning