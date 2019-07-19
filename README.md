# Distributed evolution

This is a fork of the implementation of the algorithm described in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) (Tim Salimans, Jonathan Ho, Xi Chen, Ilya Sutskever).

The implementation does not use MuJoCo environments and AMI but the [Roboschool](https://github.com/openai/roboschool/) from OpenAI. 

The implementation here uses a master-worker architecture: at each iteration, the master broadcasts parameters to the workers, and the workers send returns back to the master. The humanoid scaling experiment in the paper was generated with an implementation similar to this one.

## Instructions

1. Install required Python packages
2. Start the redis server with the provided configuration
3. Change the JSON configuration file to your needs
4. Start the master, providing the socket for the redis server and the configuration JSON
5. Start the workers and let them train the environment
6. With viz.py you can test the trained weights, by providing the environment ID and the weight file