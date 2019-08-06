FROM ubuntu:xenial

RUN apt-get update
RUN apt-get dist-upgrade -y

# Install base requirements
RUN apt-get install -y python3 python3-pip git

# Roboschool Requirements
RUN apt-get install -y libgl1-mesa-dev libharfbuzz0b libpcre3-dev

# Install Python prerequisites
RUN pip3 install gym roboschool click tensorflow numpy

# Environment variables for click
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir evolution-strategies
ADD . evolution-strategies/

WORKDIR evolution-strategies/es_distributed/