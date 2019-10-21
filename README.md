# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments and AMI but the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

## Instructions

### Build Dockerfile

`docker build -t evolution-strategies .`

### Run built image

# Run with hashed password

The password is hashed and set in the Dockerfile.

`docker run -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`

# Run with token authentication (not up to date)

`docker run -d -p 8888:8888 evolution-strategies xvfb-run -s -screen 0 1400x900x24 start-notebook.sh`

Then run

`docker logs --tail 3 container_name`

which will show you the token which was created when starting the jupyter server.

# Run without any security measures (not up to date)

`docker run -d -p 8888:8888 evolution-strategies xvfb-run -s -screen 0 1400x900x24 start-notebook.sh --NotebookApp.token=''`
