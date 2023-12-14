---
title: Javascript 기초
description:
categories:
tags:
---

별점도 반영하고 클릭 로그도 수집하려면 javascript를 알아야한다..! 감사하게도 [Do it! 인터랙티브 웹 페이지 만들기](https://m.yes24.com/Goods/Detail/103401495) 책에
javascript 기본 내용이 있어서 읽고 정리하게 되었다. 아직 장고와 css, html, js 사이의 데이터 이동을 시키기엔 어렵다.. 하여 또 다른 javascript 책을 파야하지 않을까 싶다!

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="ie=edge">
    <title>Javascript</title>
    <link rel="stylesheet" href="css/style.css">
    <script>
        console.log("Hello world")
    </script>
</head>
<body>
</body>
</html>
```

![0](/assets/images/javascript%20basic/helloworld.png)

개발자도구를 열어보면 Hello World가 출력되어있다. 웹브라우저는 HTML 파일의 위부터 순서대로 읽는다. <body> 영역이 HTML 출력을 담당하는데, <body>보다 먼저 출력되면 아직 생성되지도 않은
HTML요소를 제어할 수 없으므로 문제가 생길 것이다.

실험해보자. 그래서 script 태그에서 #title을 잡아 출력하게 하고, <body>안에는 #title을 넣어보자.

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="ie=edge">
    <title>Javascript</title>
    <link rel="stylesheet" href="css/style.css">
    <script>
        const title = document.querySelector("#title");
        console.log(title);
    </script>
</head>
<body>
<h1 id="title">HELLO WORLD</h1>
</body>
</html>
```

![0](/assets/images/javascript%20basic/helloworld_title.png)

콘솔에서는 null이 출력된다. 자바스크립트가 실행되는 시점에서 아직 <body>부분을 읽지 못했기 때문에 #title을 잡지 못한 것이다. 호출위치를 <body>안으로 넣어보자.

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="ie=edge">
    <title>Javascript</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
<h1 id="title">HELLO WORLD</h1>
<script>
    const title = document.querySelector("#title");
    console.log(title);
</script>
</body>
</html>
```

![0](/assets/images/javascript%20basic/helloworld_in_body.png)

제대로 #title이 출력된다!!

이제부터는 <head>태그 안에 ```<script defer src="custom.js"></script>``` 처럼 외부에서 js파일을 호출하여 사용하도록 하자.
defer는 body를 해석하면서 동시에 외부 js파일을 가져오도록 한다. 그 다음 body 영역이 모두 출력된 후 외부 스크립트 파일을 실행한다.

## js로 HTML 요소 선택하기

### document.querySelector()

요소를 선택할 때 사용한다.

```html

<section id="wrap">
    <article class="box1">box1</article>
    <article class="box1">box2</article>
    <article class="box1">box3</article>
</section>
```

```javascript
document.querySelector("#wrap");
```

body에서 id가 wrap인 요소를 찾으라는 의미다.

```javascript
const frame = document.querySelector("#wrap");
console.log(frame);
```

frame이라는 변수에 #wrap 요소가 저장된다.

![0](/assets/images/javascript%20basic/selector_wrap.png)

#wrap 요소가 제대로 출력되었다.

이번엔 #wrap 안에 있는 .box1을 찾아보자.

```javascript
const box1 = document.querySelector("#wrap .box1");
console.log(box1);
```

![0](/assets/images/javascript%20basic/selector_box1.png)

article 속의 .box1이 잘 출력되었다.

### document.querySelectorAll()

요소를 모두 선택할 때 사용한다. html은 위 예시를 그대로 쓰자.

```javascript
const item = document.querySelector("#wrap article");
console.log(item);
```

이렇게하면 가장 첫 번째 요소만 선택한다! 즉 box1만 출력된다. 대신 이렇게 쓰자.

```javascript
const items = document.querySelectorAll("#wrap article");
console.log(items);
```

![0](/assets/images/javascript%20basic/selector_selectall.png)

변수엔 NodeList가 담겼다. for of 문을 사용해서 하나씩 꺼내보자.

```javascript
const items = document.querySelectorAll("#wrap article");
for (let item of items) {
    console.log(item);
}
```

변수에는 const, let이 있다. const는 변경되지 않는 것, let은 변경되는 것이다. item은 계속 변경될 것이므로 let 변수에 담자.

![0](/assets/images/javascript%20basic/selector_for_of.png)

하나씩 꺼내어 잘 출력했다. for문으로 사용하는 예시는 다음과 같다.

```javascript
const items = document.querySelectorAll("#wrap article");
for (let i = 0; i < items.length; i++) {
    console.log(items[i]);
}
```

### 부모, 자식, 형제 요소 선택

```html

<body>
<ul class="list">
    <li class="item1">box1</li>
    <li class="item2">box2</li>
    <li class="item3">box3</li>
    <li class="item4">box4</li>
</ul>
</body>
```

```javascript
const list = document.querySelector(".list");
const items = list.children;

console.log(items);
console.log(items[0]);
console.log(items[1]);
console.log(items[2]);
console.log(items[3]);
```

![0](/assets/images/javascript%20basic/parents_brother.png)

.list를 선택하면 <ul>태그를 선택한 것이다. 콘솔을 보면 자식 4개가 묶여서 출력되었다. 그리고 .children으로 자식요소를 선택해 출력하면 각각의 <li>가 선택된 것을 알 수 있다.

### 부모 요소 선택하기

```javascript
const item2 = document.querySelector(".item2");
console.log(item2.parentElement);
```

.parentElement를 사용하면 부모인 ul이 출력된다.

### 제일 가까운 상위 부모 요소 선택

```html

<body>
<main>
    <section>
        <article>
            <ul>
                <li>list</li>
            </ul>
        </article>
    </section>
</main>
</body>
```

여기서 li로부터 main을 선택하고싶다면? 다음과같이 .closest를 쓰면 된다.

```javascript
const li = document.querySelector(".list");
console.log(li.closest("main"));
```

### 형제 요소 선택

이전 형제 : item3.previousElementSibling

다음 형제 : item3.nextElementSibling

## 자바스크립트로 스타일 제어

```html

<body>
<article id="box"></article>
</body>
```

```css
#box {
    width: 200px;
    height: 200px;
    background-color: aqua;
    border: 5px solid #000;
    transform: rotate(0deg);
}
```

![0](/assets/images/javascript%20basic/css_control.png)

그림과 같이 가로세로 200px의 사각형을 하나 만들었다. 그리고 다음처럼 js파일을 연결하자.

```javascript
const box = document.querySelector("#box");

box.style.width = "10%";
box.style.height = "300px";
box.style.backgroundColor = "hotpink";
box.style.border = "none";
box.style.transform = "rotate(10deg)"
```

![0](/assets/images/javascript%20basic/css_control_box.png)

article 태그를 box 변수로 잡고, .style을 붙여 속성명을 입력하면 된다. background-color 처럼 하이픈으로 연결된 속성은 camel case로 변경해야한다. js에 예약어가 존재하기
때문이다.


> document.querySelector로 선택한 요소는 DOM 객체이다. DOM에는 HTML + 스타일 정보(style속성)도 있어서 style 속성값을 변경할 수 있다.

## 이벤트 연결하기

마우스 동작과 관련된 이벤트를 HTML과 연결하고 제어해보자.

### 클릭 이벤트 연결하기

```html

<body>
<a href="https://www.naver.com"></a>
</body>
```

```css
a {
    font-size: 100px;
    color: #555;
}
```

```javascript
const link = document.querySelector("a")

link.addEventListener("click", () => {
    console.log("링크를 클릭했습니다.");
})
```

변수 link에 <a>태그를 선택했다. .addEventListner()문은 (이벤트명, 전달될값=>실행할구문) 처럼 인자를 전달해 쓴다. 여기서는 첫 번째 인자(이벤트명) "click"을 사용했다.
두 번째 인자는 리스너인데, =>같은 화살표를 쓰며 이벤트가 발생할 때 응답해서 실행할 동작을 의미한다. 근데 위처럼 작성하면 클릭하자마자 네이버로 넘어가므로 링크이동을 막고 콘솔창에 출력해보도록 하자.

```javascript
const link = document.querySelector("a")

link.addEventListener("click", (e) => {
    e.preventDefault();
    console.log("링크를 클릭했습니다.");
})
```

![0](/assets/images/javascript%20basic/click.png)

preventDefault는 이벤트의 기본 기능을 수행하지 말라는 것이다.

여기서 e는 이벤트 객체이다.

### 호버 이벤트 연결하기

```html

<body>
<div id="box"></div>
</body>
```

```css
#box {
    width: 200px;
    height: 200px;
    background: aqua;
    margin: 100px auto;
}
```

```javascript
const box = document.querySelector("#box");

box.addEventListener("mouseenter", () => {
    box.style.backgroundColor = "hotpink";
});

box.addEventListener("mouseleave", () => {
    box.style.backgroundColor = "aqua";
});
```

![0](/assets/images/javascript%20basic/hover_enter.png)

![0](/assets/images/javascript%20basic/hover_leave.png)

이벤트 mouseenter와 mouseleave를 사용했다. div 태그의 background-color 속성이 hotpink로 바뀌는 것을 볼 수 있다.

### 반복되는 요소에 이벤트 한꺼번에 연결하기

```html

<body>
<ul class="list">
    <li><a href="" #>item1</a></li>
    <li><a href="" #>item2</a></li>
    <li><a href="" #>item3</a></li>
    <li><a href="" #>item4</a></li>
</ul>
</body>
```

```javascript
const list = document.querySelectorAll(".list li");

for (let el of list) {
    el.addEventListener("click", e => {
        e.preventDefault();
        console.log(e.currentTarget.innerText);
    })
}
```

querySelectorAll로 <li> 태그를 리스트로 묶어 list라는 변수에 담자. for of 문을 사용하여 변수 el에 저장되고 있는 반복 요소를 클릭할 때마다 해당요소를 e.currentTarget으로
선택하고 .innerText 구문을 연결해준다.
innerText 구문은 선택한 요소의 텍스트를 불러온다. 이제 각 item을 클릭하면 console에 각각 item1, item2, item3, item4가 출력된다.

![0](/assets/images/javascript%20basic/click_for_of.png)

### 클릭 이벤트가 발생할 때 숫자를 증가, 감소시키기

```html

<body>
<a href="#" class="btnPlus">plus</a>
<a href="#" class="btnMinus">minus</a>
</body>
```

```javascript
const btnPlus = document.querySelector(".btnPlus");
const btnMinus = document.querySelector(".btnMinus");
let num = 0; // 제어할 숫잣값을 0으로 초기화

//btnPlust를 클릭할 때마다
btnPlus.addEventListener("click", e => {
    e.preventDefault();
    //num값을 1씩 증가
    num++;
    console.log(num);
})

//btnMinus를 클릭할 때마다
btnMinus.addEventListener("click", e => {
    e.preventDefault();
    //num값을 1씩 감소
    num--;
    console.log(num);
})
```

플러스, 마이너스 버튼을 각각 btnPlus와 btnMinus에 담아두자. 변수 num을 0으로 초기화했다.
btnPlus에 클릭 이벤트가 생기면 num++ 해준다. 마찬가지로 btnMinus에는 num--를 해준다.

![0](/assets/images/javascript%20basic/plus_minus.png)

### 문자 안에 변수 삽입하기

```javascript
const myName = "홍길동";
console.log(`내 이름은 ${myName}입니다.`);
```

> 콘솔문에서 백틱(`)으로 감싸줘야한다.

그리고 문자 안에서 ${변수}로 사용해주면 변수값을 사용할 수 있다.

### 클릭하면 좌우로 회전하는 박스 만들기

```html

<body>
<a href="#" class="btnLeft">왼쪽으로 회전</a>
<a href="#" class="btnRight">오른쪽으로 회전</a>
<div id="box"></div>
</body>
```

```css
#box {
    width: 300px;
    height: 300px;
    margin: 50px;
    background: aqua;
    transition: 0.5s;
}
```

가로세로 300px의 박스를 만들었습니다. transition 시간은 0.5초로 설정했습니다.

```javascript
const btnLeft = document.querySelector(".btnLeft");
const btnRight = document.querySelector(".btnRight");
const box = document.querySelector("#box");
const deg = 45;
let num = 0;

btnLeft.addEventListener("click", e => {
    e.preventDefault();
    num--;
    box.style.transform = `rotate(${num * deg}deg)`;
});

btnRight.addEventListener("click", e => {
    e.preventDefault();
    num++;
    box.style.transform = `rotate(${num * deg}deg)`;
});
```

btnLeft를 클릭할때마다 num은 0에서 1, 2, ... 처럼 1씩 증가한다. 반대로 btnRight를 클릭할때마다 1씩 감소한다.
btnLeft를 클릭할때마다 각도는 0, 45, 90, 135, ... 처럼 45도씩 증가한다. 반대로 btnRight를 클릭할때마다 0, -45, -90, -135, ...처럼 45도씩 감소한다.

주의할 점은 rotate(${num * deg})를 백틱(`)으로 감싸야한다는 것이다.

![0](/assets/images/javascript%20basic/rotate_0.png)

![0](/assets/images/javascript%20basic/rotate_45.png)

클릭할때마다 45도만큼 회전한다.

## 자바스크립트로 클래스 제어하기

자바스크립트로 HTML 스타일을 변경하는건 추천되지 않는다. CSS 파일이 우선순위에서 무시되기 때문이다. 그러니 CSS에서 스타일을 설정해주고, 자바스크립트는 클래스 이름만 추가/제거하도록 해보자.

```html

<body>
<section id="wrap">
    <article></article>
</section>
</body>
```

```css
#wrap {
    width: 500px;
    height: 500px;
    border: 1px solid #000;
    padding: 100px;
    box-sizing: border-box;
    margin: 100px auto;
}

#wrap article {
    width: 100%;
    height: 100%;
    background: aqua;
    transition: 1s;
}
```

![0](/assets/images/javascript%20basic/control_class_body.png)

일단 body부터 살펴보자. 왠지모르게 가장 위부터가 아니지만, 이는 section에게 margin(바깥여백)을 준만큼 띄워졌기 때문이다. section 태그를 살펴보자.

![0](/assets/images/javascript%20basic/control_class_section.png)

가로세로 500px의 검정(#000)테두리 박스이다. 근데 margin을 상하 100px을 줬기 때문에 body 영역을 상하로 넘어섰고, 좌우는 auto를 줬기 때문에 중앙에 배치되었다.
또한 padding을 100px 줬기 때문에 내부는 300px 300px이 됐을 것이다.
또한 ```box-sizing: border-box``` 옵션을 줬으므로 500 x 500을 해치지 않고(?), 변형없이(?) 500 X 500의 검정 박스가 생겼다. 이제 article을 보면..

![0](/assets/images/javascript%20basic/control_class_article.png)

가로세로 부모의 100%를 쓴다고 해도 300px X 300px 이 되었다.

```javascript
const wrap = document.querySelector("#wrap");
const box = wrap.querySelector("article");

wrap.addEventListener("click", () => {
    box.style.backgroundColor = "hotpink";
});
```

wrap을 변수에 저장하고, 다시 wrap을 이용해 querySelector를 사용해 article 자식을 선택한다.
wrap을 클릭하면 article의 background-color를 hotpink로 바꾼다.

![0](/assets/images/javascript%20basic/control_class_hotpink.png)

클릭하면 색깔이 1초동안 hotpink로 바뀐다. article 태그의 style 속성에 "background-color:hotpink"가 추가되었다.
그런데 이렇게 태그에 인라인 형태로 삽입된 스타일 구문은 우선순위가 매우 높아서 기존 CSS를 무시하게 된다. 그러니 클래스명을 추가해서 제어해보자.

```css
#wrap.on article {
    background: hotpink;
}
```

```javascript
wrap.addEventListener("click", () => {
    wrap.classList.add("on");
});
```

클릭하면 ```on``` 클래스를 추가하고, 클래스에 ```on```이 추가되면 article의 배경색이 hotpink로 바뀌도록 합니다.

![0](/assets/images/javascript%20basic/control_class_addon.png)

클릭하면 #wrap 요소에 "on"이라는 요소가 추가됐다. 스타일을 강제로 바꾸지 않으면서 부모 요소에 "on"만 추가되어 변경되도록 했다.
다시 클릭했을때 색이 돌아오도록 하려면 클릭상태를 저장했다가 "on"을 제거하면 된다.

```javascript
wrap.addEventListener("click", () => {
    let isOn = wrap.classList.contains("on");
    console.log(isOn);

    if (isOn) {
        wrap.classList.remove("on");
    } else {
        wrap.classList.add("on");
    }
});
```

또는 삼항연산자를 이용해서 다음과 같이 쓸 수 있다.

```javascript
(isOn) ? wrap.classList.remove("on") : wrap.classList.add("on");
```

![0](/assets/images/javascript%20basic/control_class_removeon.png)

클릭했다가 다시 클릭하면 inOn은 False로 바뀌며 aqua로 돌아온다.


**이처럼, HTML이 이미 출력되었다 하더라도 이벤트가 발생할때마다 언제든지 그 요소를 동적으로 변경할 수 있다!**

위 기능은 ```classList.toggle()``` 기능을 쓰면 효율적으로 구현할 수 있다.

```javascript
wrap.addEventListener("click", () => {
    wrap.classList.toggle("on")
})
```

선택한 요소에 클래스가 있으면 제거해주고 없으면 추가해주라는 의미이다.

---

함수를 활용하여 코드 패키징하기와 속성값 활용하기는 다음에 정리.






































































































