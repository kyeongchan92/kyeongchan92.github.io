---
title: gunicorn, WSGI, CGI란 무엇인가, 그리고 gunicorn 사용 명령어
description:
categories:
tags:
---

**웹 서버**

웹 서버란 웹 브라우저(크롬 등)의 요청을 처리하는 서버이다. 웹 서버에 요청이 들어온다 치자. 예를 들어 ```www.pseudorec.com```, ```www.pseudorec.com/movie/sasrec/``` 등을 주소창에 치는 것이다. 


웹 서버는 동적 요청이 들어오면 알맞는 파이썬 프로그램을 호출해야 할 것이다. (대표적인 웹 서버에는 아파치(`Apache`), 엔진엑스(`Nginx`) 등이 있다.) 하지만 대부분의 웹서버는 파이썬 프로그램을 호출할 수 있는 기능이 없고, 그래서 파이썬 프로그램을 호출하는 WSGI가 필요하다.
웹 서버에 동적 요청이 발생하면 웹 서버가 WSGI 서버를 호출하고, WSGI 서버는 파이썬 프로그램을 호출하여 동적 페이지 요청을 대신 처리하는 것이다. 동적 요청이란 DB의 데이터를 가져온다든지 하는, 요청을 처리한 후 응답하여 응답이 계속 변하는 요청을 말한다.  

**gunicorn과 WSGI**

gunicorn은 wsgi(발음 위스키)의 일종이다. wsgi는 cgi에서 파생된 개념이다.

cgi는 common gateway interface의 약자로, 공통 경로 인터페이스이다. 개발하는 언어가 제각각이니, 중간의 동시통역사처럼 '이 문을 지나면 이러한 형태가 된다'고 정해놓은 규약이다.
사용자의 http 요청은 웹서버로 들어온다. 이 때 cgi를 통해 일관된 형태로 해석되어(번역되어) 웹서버에 전달, 웹서버 내부로 들어오는 것이다.

wsgi는 web server gateway interface의 약자로, 웹을 위해 만들어진 인터페이스다. 근데 python이란 단어가 들어가있지 않지만, python 전용으로 쓰이는 단어라고 한다. 즉, '웹 서버'에서의 요청을 파이썬 애플리케이션에 던지는 역할이다.
WSGI 서버는 웹서버가 동적 페이지 요청을 처리하기 위해 호출하는 서버이다.

그러니까 gunicorn은 HttpRequest를 python이 이해할 수 있게 동시통역해주는 녀석이다. Gunicorn과 uwsgi를 가장 많이 사용한다.
웹서버에 동적 페이지 요청이 발생하면 웹 서버는 WSGI 서버를 호출하고 WSGI 서버는 다시 WSGI 애플리케이션을 호출한다.
WSGI 애플리케이션에는 장고(Django), 플라스크(Flask) 등이 있다.

![0](/assets/images/wsgi_diagram.png)


## Gunicorn 사용 명령어
pip 명령어로 gunicorn을 설치한다. gunicorn은 서버에 설치되는 것이기 때문에 꼭 프로젝트 경로가 아니어도 된다.
```shell
pip install gunicorn
```

그 다음 프로젝트 경로로 이동 후 다음과 같은 명령어를 수행
```shell
gunicorn --bind 0:8000 config.wsgi:application
```
`--bind 0:8000` : 8000번 포트를 사용하겠다는 것이다.

`config.wsgi:application` : config/wsgi.py 파일의 application이란 어플리케이션을 실행하겠다는 것이다.
config/wsgi.py의 application이란 예를 들면 다음과 같이 작성된다.

```python
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

application = get_wsgi_application()
```

여기서의 application이 바로 장고의 어플리케이션이며, 이 파일은 장고 프로젝트 생성시 자동으로 만들어져 있을 것이다. 따로 수정할 필요가 없다.

## Worker와 Thread 수 설정

**Worker**

Worker는 독립적인 프로세스로, 클라이언트의 요청을 동시에 처리하는 역할을 합니다.
Gunicorn은 여러 개의 worker 프로세스를 생성하여 동시에 여러 요청을 처리할 수 있게 합니다.
**각 worker는 고유한 메모리 공간을 가지고 있어**, 하나의 worker가 오류로 인해 종료되더라도 다른 worker는 계속해서 서비스를 제공할 수 있습니다.
Worker의 수를 늘리면 동시에 처리할 수 있는 요청의 수가 늘어나지만, 이에 따라 메모리 사용량이 늘어나게 됩니다.

Worker 수 설정

--workers 또는 -w 옵션을 사용하여 Gunicorn이 생성할 worker 프로세스의 수를 지정할 수 있습니다. 예를 들어, -w 4는 4개의 worker를 생성하도록 지시합니다.
worker의 수는 일반적으로 코어 1개당 2-4개를 곱해 사용하면 된다.


**Thread** 

Thread는 프로세스 내에서 동작하는 가장 작은 실행 단위입니다. thread는 자원을 공유합니다.
Python GIL(Global Interpreter Lock) 때문에 기본적으로 Python은 하나의 스레드에서만 코드를 실행할 수 있습니다. 이는 멀티코어 환경에서 병렬 처리가 어렵게 만들 수 있습니다.
그러나 I/O 바운드 작업에서는 여러 스레드를 사용하여 동시에 여러 작업을 처리할 수 있습니다.
Gunicorn에서 worker와 thread의 설정은 다음과 같습니다:

**Thread 수 설정**

Gunicorn은 worker 내에서 스레드를 사용할 수 있습니다. --threads 또는 -t 옵션을 사용하여 각 worker가 생성하는 스레드의 수를 지정할 수 있습니다. 예를 들어, -t 2는 각 worker가 2개의 스레드를 사용하도록 지시합니다.

이외 다른 매개변수들

`-k STRING` 또는 `--worker-class STRING`, default STRING `'sync'`

Gunicorn의 워커(worker) 클래스를 지정.

1. sync :
   1. 기본값으로 사용되며, 각 요청을 동기적으로 처리합니다.
   2. 간단하고 안정적이며 대부분의 일반적인 상황에 적합합니다.
2. eventlet:
   2. `pip install gunicorn[eventlet]` 명령을 사용하여 설치할 수 있습니다.
   3. Eventlet 기반의 워커로, 비동기 이벤트 기반 프로그래밍을 지원합니다.
3. gevent:
   1. `pip install gunicorn[gevent]` 명령을 사용하여 설치할 수 있습니다.
   2. Gevent 라이브러리를 사용하여 이벤트 기반 및 비동기 프로그래밍을 지원합니다.
4. tornado:
   1. `pip install gunicorn[tornado]` 명령을 사용하여 설치할 수 있습니다.
   2. Tornado 라이브러리를 사용하여 비동기 및 웹 소켓 지원이 특징입니다.
5. gthread:
   1. `pip install gunicorn[gthread]` 명령을 사용하여 설치할 수 있습니다.
   2. Python 2에서 사용되며, futures 패키지를 통해 멀티스레딩을 지원합니다. (Python 3에서는 asyncio를 사용)


`-t INT` 또는 `--timeout INT` : Gunicorn 웹 서버의 작업자(worker)가 얼마 동안 응답을 하지 않으면 종료되고 다시 시작되어야 하는지를 설정.
기본값은 30초입니다.
이 옵션은 작업자(worker) 프로세스가 몇 초 동안 응답하지 않으면 재시작되어야 하는지를 나타냅니다.
0으로 설정하면 작업자 프로세스가 응답하지 않는 것을 무한정으로 허용하게 됩니다. 이는 모든 작업자에 대해 제한 시간을 비활성화하는 효과가 있습니다.
예를 들어, -t 30은 기본값과 동일하게 30초 동안 응답이 없으면 작업자를 재시작하도록 설정합니다.

주로, 작업자가 일정 시간 동안 응답하지 않으면 이를 감지하고 해당 작업자를 다시 시작함으로써 서버 안정성을 유지하는 데 사용됩니다. 특히 동기(sync) 작업자에 대해서는 더 높은 값으로 설정하는 것에 대해 주의가 필요하며, 일반적으로 기본값인 30초가 적절한 경우가 많습니다.


`--keep-alive INT` : Gunicorn 웹 서버가 Keep-Alive 연결에서 요청을 기다리는 시간을 지정. Keep-Alive는 하나의 TCP 연결을 통해 여러 HTTP 요청 및 응답을 처리할 수 있도록 하는 기술입니다.
기본값은 2초입니다.
이 옵션은 Keep-Alive 연결에서 요청을 기다리는 시간을 나타냅니다.
보통은 1-5초의 범위 내에서 설정되며, 특히 클라이언트와 직접 연결된 서버의 경우 사용됩니다. (예: 로드 밸런서가 없는 경우)
로드 밸런서 뒤에 Gunicorn이 배치된 경우, 이 값을 더 높게 설정하는 것이 합리적일 수 있습니다.
이 옵션을 조정함으로써, 서버가 Keep-Alive 연결을 유지하는 동안 클라이언트로부터의 추가 요청에 대한 응답을 더 빠르게 처리하거나, 로드 밸런서와 같은 중간 계층이 요청을 전달하는 데 걸리는 시간에 대한 대응을 조절할 수 있습니다.

만약 로드 밸런서가 있는 환경에서 작업 중이라면, 로드 밸런서와의 통신 및 Keep-Alive 설정에 대한 문서도 함께 확인하는 것이 좋습니다.

출처
---
[점프 투 장고 - 4-09 WSGI](https://wikidocs.net/75556)

[gunicorn은 대체 뭐하는 놈일까 (부제: CGI, WSGI는 대체 뭐냐)](https://this-programmer.tistory.com/345)