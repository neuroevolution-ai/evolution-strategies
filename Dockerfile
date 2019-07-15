FROM ubuntu

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y redis-server

COPY redis_config/redis_master.conf /etc/redis/redis.conf

#RUN systemctl enable --now redis.service

CMD ["python", "--version"]