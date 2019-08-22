# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments and AMI but the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

## Instructions

### Build Dockerfile

`docker build -t evolution-strategies .`

### Run built image

`docker run -p 8888:8888 evolution-strategies`