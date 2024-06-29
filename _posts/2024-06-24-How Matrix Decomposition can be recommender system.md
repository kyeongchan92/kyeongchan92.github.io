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
그러므로, 이는 특정 영화들(science fiction)에 대해 유저들을 한 타입으로 그룹화한다.

유사하게, first right-singular $$v_1$$은 science fiction 영화들에게 높은 별점을 준 Ali와 Beatrix에게 큰 절댓값을 주게 된다. 이는 $$v1$$이 science fiction을 좋아하는 사람들을 나타내는 것이라 볼 수 있다!

유사하게, $$u2$$는 프랑스 예술 영화 테마를 잘 포착하고 있는 것 같으며, $$v2$$는 Chandra가 그러한 영화를 이상적으로 사랑하는 사람임을 나타낸다.  
공상 과학 영화 애호가는 순수주의자이며, 오직 공상 과학 영화만을 좋아한다. 그래서 공상 과학 영화를 좋아하는 사람 $$v1$$은 공상 과학 테마가 아닌 모든 것에 대해 0점을 준다. 이 논리는 특이값 행렬 $$\Sigma$$의 대각 하부구조에 암시되어 있습니다

이어서..

참고 : Mathematics for Machine Learning