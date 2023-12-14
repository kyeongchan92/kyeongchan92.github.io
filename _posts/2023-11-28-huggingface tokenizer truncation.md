---
title: Huggingface 토크나이저 Padding and truncation
description:
categories:
tags:
---

huggingface에서 tokenizer를 ```.from_pretrained``` 메서드로 가져다 쓰는데, 갑자기 padding과 truncation, max_length 인자에 대해 헷갈렸다. 그래서 한 번 정리해보고자 한다.

이 글은 다음을 번역하였습니다 : 
[Huggingface](https://huggingface.co/) > [Docs](https://huggingface.co/docs) > [Transformers](https://huggingface.co/docs/transformers/index) > [Padding and truncation](https://huggingface.co/docs/transformers/pad_truncation)

---

배치 인풋의 길이는 모두 다르기 때문에, 고정된 길이로 변환시킬 수 없는 노릇이다. Padding과 truncation 전략으로 이 문제를 해결한다.

### ```Padding```

스페셜 토큰인 padding token을 뒤쪽에 붙여서 짧은 시퀀스가 배치 내 가장 긴 시퀀스와 같은 길이를 갖게 한다. 또는 모델에 허용되는 가장 긴 길이로 만든다. 다음 값들을 갖는다.
- ```True``` or ```'longest'``` : 배치에서 가장 긴 길이만큼 패딩한다. 시퀀스가 하나라면 당연히 패딩하지 않는다.
- ```'max_length'``` : ```max_length``` 인자로 넘겨받은 길이까지 패딩한다. max_length 인자를 넘겨주지 않는 경우, 혹은 ```max_length```=```None인``` 경우엔 모델이 허용하는 최대 길이만큼 패딩한다. 시퀀스가 하나여도 패딩이 적용된다.
- ```False``` or ```'do_not_pad'``` : default값이며 패딩이 적용되지 않는다.

### ```Truncation```

긴 시퀀스를 잘라버림으로써 해결한다. 다음과 같은 인자 값을 갖는다.

- ```True``` or ```'longest_first'``` : ```max_length``` 인자로 받은 길이까지 자른다. 또는 ```max_length``` 인자를 넘겨주지 않거나 `None으로` 넘겨주는 경우, 모델이 허용하는 최대 길이까지 자른다. 당연히 토큰 단위로 자른다.
- ```'only_second'``` : ```True``` or ```'longest_first'```와 비슷하지만, 두 문장이 pair로 입력되는 경우 두 번째 sentence만 자른다.
- `'only_first'` : 마찬가지로 두 문장이 pair로 입력되는 경우 첫 번째 sentence만 자른다.
- `False` of `'do_not_truncate'` : defualt 값이며, 자르지 않는다.

