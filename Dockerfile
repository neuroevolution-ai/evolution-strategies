FROM jupyter/base-notebook:latest

# Switch to root user to install packages
USER root

# Update the system and install base and roboschool requirements
RUN apt-get update -y && apt-get install -y git xvfb ffmpeg libgl1-mesa-dev libharfbuzz0b libpcre3-dev libqt5x11extras5 build-essential

# Switch back to unprivileged user for python packages. User is defined in base docker image
USER $NB_USER

# NumPy has changed something in version 1.17+ which causes import errors in TensorFlow. Until this fix is merged
# use a slightly older version of NumPy, same with gast
RUN conda install --quiet --yes \
    'matplotlib' \
    'pandas' \
    'ipywidgets'

RUN conda clean --yes --all -f && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Roboschool is deprecated after version 1.0.48
# Install TensorFlow and NumPy with pip to prevent using the MKL version which in this implementation is slower
RUN pip install --quiet \
    tensorflow \
    numpy \
    gym \
    pybullet

# TODO remove this if no widgets are used or Jupyter notebook is sed
RUN conda install --yes -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR work/evolution-strategies/

ADD hashed_password.txt .
ADD launch.sh .

# Used to test if the robot specifiation file for the MuJoCo Ant is easier to train than the PyBullet Ant. I.e, this
# replaces the PyBulletAnt with the MuJoCo Ant (in terms of the robot specification, but not within the code itself,
# (Power coefficient, energy cost, etc.)).
# RUN mv /opt/conda/lib/python3.7/site-packages/pybullet_data/mjcf/ant.xml /opt/conda/lib/python3.7/site-packages/pybullet_data/mjcf/ant.xml.bak
# RUN ln -s /home/jovyan/work/evolution-strategies/tests/mujoco_envs_pybullet/mujoco_ant.xml /opt/conda/lib/python3.7/site-packages/pybullet_data/mjcf/ant.xml

CMD ["/bin/bash", "launch.sh"]
