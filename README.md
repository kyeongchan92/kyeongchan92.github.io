# 실행
```shell
bundle exec jekyll serve
```


# 마크다운 꿀팁

### 형광펜 효과
```
<span style="background-color:#fff5b1">OOM 우선순위</span>
```

### 글자색 바꾸기
```
<span style="color:red"> red </span>
```

### 바로가기 문자
↗

### 링크 새창띄우기
[Runtime options with Memory, CPUs, and GPUs](https://docs.docker.com/config/containers/resource_constraints/){:target="_blank"}

### 표
|제목|내용|설명|
|:---|---:|:---:|
|왼쪽정렬|오른쪽정렬|중앙정렬|

### bold 수식
```\boldsymbol{b}``` -> $$\boldsymbol{b}$$

# 형식 바꾸기

scaffolding/base.scss 에서 바꾸면 된다. main.scss에서 바꿔봤는데 계속 다시 원래대로 돌아간다 스스로....

# Topics 탭 파일
navigator > index.md

# 이미지 캡션 달기
![0](/assets/images/ngcf/figure2.png)*Figure2. NGCF 아키텍쳐. 화살표는 정보가 흐른다는 것을 의미한다. 그림 가장 밑에 $$u_1$$과 $$i_4$$가 있다. 각각 여러개의 임베딩 전파 레이어를 거치고 마지막엔 레이어의 아웃풋들이 concat되어, 최종 예측 스코어 계산에 쓰인다.*