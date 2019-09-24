FROM ubuntu:xenial

RUN apt-get update
RUN apt-get dist-upgrade -y

# Install base requirements
RUN apt-get install -y python3 python3-pip git redis-server tmux

# Roboschool Requirements
RUN apt-get install -y libgl1-mesa-dev libharfbuzz0b libpcre3-dev libqt5x11extras5

# Install Python prerequisites
RUN pip3 install gym roboschool click tensorflow numpy redis

RUN apt-get install -y nano

# Environment variables for click
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir /home/evolution-strategies

RUN groupadd -g 1010 appuser && useradd -r -u 1010 -g appuser appuser
USER appuser

ADD . /home/evolution-strategies/

WORKDIR /home/evolution-strategies/

ADD gym_forward_walker.py /usr/local/lib/python3.5/dist-packages/roboschool/gym_forward_walker.py

CMD ["/bin/bash"]
