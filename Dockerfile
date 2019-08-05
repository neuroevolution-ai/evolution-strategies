FROM ubuntu:xenial

RUN apt-get update
RUN apt-get dist-upgrade -y

# Install Python3 and Redis server
RUN apt-get install -y python3 python3-pip git

# Install Python prerequisites
RUN pip3 install gym roboschool click tensorflow numpy

RUN git clone https://github.com/spikingevolution/evolution-strategies.git
WORKDIR evolution-strategies/es_distributed/

# todo: Delete when merged into master branch
RUN git checkout develop

# Train the environment provided in the configuration file
CMD ["python3", "main.py", "--exp_file", "../configurations/humanoid.json"]