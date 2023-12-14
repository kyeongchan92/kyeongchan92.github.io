---
title: AWS ECS task, container 메모리 제한
description:
categories:
tags:
---


ECS를 사용하기 위해서는 도커 먼저 이해해야한다. 도커는 client-server 어플리케이션이다. 리눅스, 윈도우, 맥OS 어느곳에나 설치되어 도커 컨테이너를 실행할 수 있게 해준다.
컨테이너란 앱 또는 앱의 일부를 실행할 가벼운 환경이다. 하나의 컴퓨터에 Docker 소프트웨어가 설치되어 있다면, 여러 가지 다른 컨테이너를 동시에 여러개 실행할 수 있다.

도커 컨테이너를 사용하면 팀원들이 모두 일관된 개발 환경을 가질 수 있다. 소프트웨어, os, 하드웨어의 구성을 어느 컴퓨터에서나 동일하고 표준적인 블록으로 추상화할 수 있기 때문이다.

# Cluster

![0](/assets/images/ecs example.png)

cluster는 ECS 컨테이너 인스턴스의 그룹이다. ECS는 이러한 인스턴스에 대한 일정, 유지 및 확장 요청 처리 로직을 다룬다. 또한 CPU 및 메모리 요구 사항을 기반으로 각 태스크의 최적 위치를 찾아준다.

# task에 메모리 할당하기

태스크 레벨과 컨테이너 레벨 모두에서 메모리를 정의할 수 있다.

태스크 레벨에서 정의된 메모리는 그 태스크의 hard limit이라고 부른다.

컨테이너 레벨에서는 태스크에 메모리를 할당하기 위한 두 가지 파라미터가 있다. momoryReservation(soft limit)과 memory(hard limit)이다.

소프트 메모리 제한은 스케쥴러가 태스크를 인스턴스에서 실행하기 위해 메모리를 예약해두는 양이다. 소프트 제한은 짧은 시간동안 초과되기도 한다.

한편 하드 메모리 제한은 엄격히 메모리를 제한하는 limit이며 초과될 수 없다. 하드 제한이 넘어가면 컨테이너는 종료된다.

> Note! memory와 memoryReservation은 태스크 정의 >  컨테이너 정의에서 설정한다.
 
컨테이너 정의에서 momoryReservation(soft limit)나 memory(hard limit) 둘 중 하나에는 0이 아닌 Integer를 입력해야한다!
만약 둘 다 적을 것이라면 하드는 소프트보다 커야한다.
([AWS > aws-documentation > Amazon ECS > 개발자 안내서 > 파라미터 참조 및 리소스 템플릿 > 태스크 정의 파라미터](https://docs.aws.amazon.com/ko_kr/AmazonECS/latest/developerguide/task_definition_parameters.html))

---
참고

[A beginner’s guide to Amazon’s Elastic Container Service](https://www.freecodecamp.org/news/amazon-ecs-terms-and-architecture-807d8c4960fd/)

[How can I allocate memory to tasks in Amazon ECS?](https://repost.aws/knowledge-center/allocate-ecs-memory-tasks)

[Task definition parameters](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html)
