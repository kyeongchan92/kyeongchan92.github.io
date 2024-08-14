---
title: 추천시스템과 Matrix Decompositions — 3. 고유값과 고유벡터, Cholesky Decomposition
description:
categories:
tags:
---

**Graphical Intuition in Two Dimensions**

determinants, eigenvectors, 그리고 eigenvalues에 대해 직관적인 이해로 들어가보자. Figure 4.4는 행렬 $$\mathbf{A}_1$$, …, $$\mathbf{A}_5$$와 이들에 의한 점들의 변형을 보여주고 있다.

- $$\mathbf{A}_1 = \begin{bmatrix} \frac{1}{2} & 1 \\
0 & 2
\end{bmatrix}$$. 두 고유벡터들의 방향이 2차원 canonical basis 벡터들과 나란한 상황이다. 수직축 방향으로 2만큼 늘어나고(고유값 $$\lambda_1=2$$), 수평축 방향으로 $$\frac{1}{2}$$만큼 압축된다. 넓이는 보존된다.
- $$\mathbf{A}_2 = \begin{bmatrix} 1 & \frac{1}{2} \\
0 & 1
\end{bmatrix}$$은 전단 매핑(sheering mapping)인데, 즉, y축의 양의 방향에 있다면 오른쪽으로, 음의 방향에 있다면 왼쪽으로 전단한다. 이 매핑도 넓이를 보존한다. 고유값은 두 값이 동일한 $$\lambda_1=1=\lambda_2$$이며, 고유 벡터들은 collinear이다. 즉, 그림처럼 수평 축 방향으로만 늘어나거나 줄어든다.
- $$\mathbf{A}_3 = \begin{bmatrix} \cos{\frac{\pi}{6}} & -\sin{\frac{\pi}{6}} \\
\sin{\frac{\pi}{6}} & \cos{\frac{\pi}{6}}
\end{bmatrix}=\frac{1}{2}\begin{bmatrix} \sqrt{3} & --1 \\
1 & \sqrt{3}
\end{bmatrix}$$은 점들을 $$\frac{\pi}{6}$$, 즉 30도만큼 반시계 방향으로 회전시키다. 그리고 허수의 고유값을 갖는다.
- $$\mathbf{A}_4 = \begin{bmatrix} 1 & -1 \\
-1 & 1
\end{bmatrix}$$은 표준 기저에서의 2차원 도메인을 1차원으로 줄이는 매핑이다. 한 고유값이 0이기 때문에, 이에 해당하는 파란색 고유벡터 방향의 점들은 넓이가 0이 된다. 반면 이와 수직인 빨간색 고유벡터 방향으로는 고유값인 $$\lambda_2=2$$만큼 늘어난다.
- $$\mathbf{A}_5 = \begin{bmatrix} 1 & \frac{1}{2} \\
\frac{1}{2} & 1
\end{bmatrix}$$는 전단도 하고 늘리기도 하는 매핑이다. 이 행렬의 determinant는 $$|\det(\mathbf{A}_5)|=\frac{3}{4}$$이기 때문에, 넓이를 75%로 만든다. 빨간 고유벡터 방향의 넓이는 $$\lambda_2=1.5$$에 의해 늘어나고, 파란 고유벡터 방향의 넓이는 $$\lambda_1=0.5$$에 의해 줄어든다.

![0](/assets/images/matrix decomposition 3/transform.png)

---

> Theorem 4.12. 서로 다른 고유값 $$\lambda_1, ..., \lambda_n$$을 갖는 행렬 $$\mathbf{A}\in \mathbb{R}^{n\times n}$$의 고유벡터 $$x_1, ..., x_n$$는 선형 독립이다.
> 

위 정리는 n개의 서로 다른 고유값을 갖는 행렬의 고유 벡터들은  $$\mathbb{R}^n$$의 기저를 형성한다는 것이 된다.

> Definition 4.14. 만약 정방행렬 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$이 $$n$$보다 적은 선형 독립의 고유 벡터를 갖는다면 $$defective$$이다.
> 

$$non-defective$$ 행렬 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$이 필수적으로 $$n$$개의 서로 다른 고유값을 필요로 하는 것은 아니다. 하지만, 고유 벡터들이 $$\mathbb{R}^n$$의 기저를 형성해야한다. 

---

# 4.3 Cholesky Decomposition

머신러닝에서 우리가 자주 마주하는 특별한 유형의 행렬을 분해하는데는 다양한 방법들이 있다. 양의 실수에서 $$9$$가 $$3\cdot 3$$으로 분해되는걸 생각해보자. 행렬에 대해서 비슷하게 하려면 좀 조심해야한다. symmetric, positive definite matrices에 대해서는, Cholesky 분해가 유용하다!

> Theorem 4.18 (Cholesky Decomposition). Symmetric, positive definite 행렬 $$\mathbf{A}$$는 $$\mathbf{A}=\mathbf{L}\mathbf{L}^{\top}$$로 분해될 수 있다. $$\mathbf{L}$$은 양의 대각 요소를 가진 lower triangular 행렬이다.
> 
> 
> $$
> 
> \begin{bmatrix}a_{11} & \cdots & a_{1n} \\\vdots & \ddots & \vdots \\a_{n1} & \cdots & a_{nn}\end{bmatrix}=\begin{bmatrix}l_{11} & \cdots & 0 \\\vdots & \ddots & \vdots \\l_{n1} & \cdots & l_{nn}\end{bmatrix}\begin{bmatrix}l_{11} & \cdots & l_{1n} \\\vdots & \ddots & \vdots \\0 & \cdots & l_{nn}\end{bmatrix}
> $$
> 
> 이 때의 $$\mathbf{L}$$을 $$\mathbf{A}$$의 Cholesky factor라고 부르며, $$\mathbf{L}$$은 유일하다.
> 

Symmetric positive definite 행렬은 3단원에 나오는데, 잠시 살펴보자.

> Definition 3.4 (Symmetric, Positive Definite Matrix)
> 
> $$
> \forall x \in V \setminus \left\{ \mathbf{0}\right\} : x^{\top}\mathbf{A} x > 0
> $$
> 
> 위 식을 만족하는 대칭 행렬 $$\mathbf{A} \in \mathbb{R}^{n \times n}$$을 symmetric, positive definite,  또는 그냥 positive definite이라고 부른다.
> 
> 예를 들어 $$\mathbf{A} = \begin{bmatrix} 9 & 6 \\ 6 & 5  \end{bmatrix}$$의 경우, $$\begin{bmatrix} x_1 & x_2 \end{bmatrix}
> \begin{bmatrix} 9 & 6 \\ 6 & 5  \end{bmatrix}
> \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = (3x_1 + 2x_2)^2 + x_2^2 > 0$$이기 때문에 symmetric, positive definite이다. 
> 

이제 symmetric positive definite 행렬에 대해 Cholesky 분해하는 예제를 보자.

> Example 4.10 (Cholesky Factorization)
Symmetric, positive definite 행렬 $$\mathbf{A} \in \mathbb{R}^{3 \times 3}$$이 있다고 하자. Cholesky 분해 $$\mathbf{A}=\mathbf{L}\mathbf{L}^\top$$을 해보자.
> 
> 
> $$
> \mathbf{A} = \begin{bmatrix}a_{11} & a_{21} & a_{31} \\ a_{21} & a_{22} & a_{32} \\a_{31} & a_{32} & a_{33}\end{bmatrix}=\mathbf{L}\mathbf{L}^{\top}=\begin{bmatrix}l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\l_{31} & l_{32} & l_{33}\end{bmatrix} \begin{bmatrix}l_{11} & l_{21} & l_{31} \\ 0 & l_{22} & l_{32} \\0 & 0 & l_{33}\end{bmatrix}
> $$
> 
> 우변을 곱한 결과는
> 
> $$
> \mathbf{A} = \begin{bmatrix}l_{11}^2 & l_{21}l_{11} & l_{31}l_{11} \\ l_{21}l_{11} & l_{21}^2 + l_{22}^2 & l_{31}l_{21} + l_{32}l_{22} \\l_{31}l_{11} & l_{31}l_{21} + l_{32}l_{22} & l_{31}^2 + l_{32}^2 + l_{33}^2\end{bmatrix}
> $$
> 
> 그럼 다음과 같은 관계가 도출된다.
> 
> $$
> l_{11} = \sqrt{a_{11}},\;\; l_{22}=\sqrt{a_{22}-l_{21}^2}, \;\; l_{33}=\sqrt{a_{33} - (l_{31}^2) + l_{32}^2)}
> $$
> 
> 이런 방식으로 어떠한 symmetric, positive definite $$3 \times 3$$ 행렬에 대하여도 Cholesky 분해를 할 수 있다.

