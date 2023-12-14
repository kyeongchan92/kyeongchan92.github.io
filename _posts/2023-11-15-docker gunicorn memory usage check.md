---
title: Docker 내 gunicorn 사용시 memory usage 확인
description:
categories:
tags:
---

도커 컨테이너에서 Gunicorn을 실행 중일 때, 메모리 사용량을 확인하는 방법

1. **도커 CLI를 통한 메모리 통계 확인:**
    - 다음 명령을 사용하여 도커 컨테이너의 메모리 사용량을 확인할 수 있습니다.
      ```bash
      docker stats [options] [container_name]
      ```
        - ```container_name```은 Gunicorn이 실행 중인 컨테이너의 이름 또는 ID로 대체됩니다.
        - 이 명령은 실시간으로 CPU, 메모리, 네트워크 등의 통계를 표시합니다.

| Column name           | Description|
|:----------------------|---|
| CONTAINER ID and Name | the ID and name of the container                                                             |
| CPU % and MEM %       | the percentage of the host's CPU and memory the container is using                           |
| MEM USAGE / LIMIT     | the total memory the container is using, and the total amount of memory it is allowed to use |
| NET I/O               | The amount of data the container has received and sent over its network interface            |
| BLOCK I/O             | The amount of data the container has written to and read from block devices on the host      |
| PIDs                  | The number of processes or threads the container has created                                 |


2. **도커 컨테이너 내부에서 Gunicorn 프로세스의 메모리 사용량 확인:**
    - 컨테이너 내부에서 직접 Gunicorn 프로세스의 메모리 사용량을 확인할 수 있습니다.
    - `ps`나 `top` 명령을 사용할 수 있습니다. 예를 들어:
      ```bash
      docker exec -it [container_name] ps aux
      ```
      또는
      ```bash
      docker exec -it [container_name] top
      ```
        - `container_name`은 Gunicorn이 실행 중인 컨테이너의 이름 또는 ID로 대체됩니다.
        - 이 명령은 컨테이너 내부에서 실행 중인 모든 프로세스의 목록과 자원 사용량을 표시합니다.


3. **도커 컨테이너의 로그 확인:**
    - Gunicorn은 로그에 메모리 사용에 관한 정보를 기록할 수 있습니다. Gunicorn 로그를 확인하여 메모리 사용량을 파악할 수 있습니다.
    - Gunicorn 로그는 일반적으로 STDOUT 또는 특정 로그 파일에 출력됩니다.


4. **도커 컨테이너의 메모리 리소스 사용 확인:**
    - 도커는 ```docker stats``` 명령 외에도 ```docker inspect```를 사용하여 컨테이너의 자세한 정보를 얻을 수 있습니다.
      ```bash
      {% raw %}
      docker inspect --format='{{.Config.Memory}}' [container_name]
      {% endraw %}
      ```
        - `container_name`은 Gunicorn이 실행 중인 컨테이너의 이름 또는 ID로 대체됩니다.
        - 이 명령은 컨테이너가 할당된 메모리 양을 표시합니다.


5. 컨테이너의 메모리 한도 확인
    - 컨테이너 메모리 구성을 확인하는 명령어
       ```bash
      {% raw %}
       docker inspect --format='{{.Config.Memory}}' <container_id_or_name>
       # 이게 안되면 Docker inspect 결과에서 "Memory" 필드가 존재하지 않는다는 의미이므로 아래와같이
       docker inspect --format='{{.HostConfig.Memory}}' <container_id_or_name>
      {% endraw %}
       ```

      모든 설정을 보고 싶다면 다음과 같이!
       ```bash
       docker inspect 962ec674f2c5
       ```

이러한 방법 중 하나를 선택하여 Gunicorn이 실행 중인 도커 컨테이너의 메모리 사용량을 확인할 수 있다.