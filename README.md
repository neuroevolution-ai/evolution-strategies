# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments, instead the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

## Instructions

### Build Dockerfile

`docker build -t evolution-strategies .`

### Run built image

# Run with hashed password

The password is hashed and set in the Dockerfile. Currentliy it is set to `es-jupyter`. If you want to change it, [here](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#preparing-a-hashed-password) is a guide to hash a password.

`docker run -d -p 8888:8888 -v $(pwd):/home/jovyan/work/evolution-strategies evolution-strategies`

# Run without any security

If you want to run the Jupyter Notebook without any security measures replace the argument `--NotebookApp.password=...` with
`--NotebookApp.token=''` in the Dockerfile.
