FROM jupyter/base-notebook:latest

# Switch to root user to install packages
USER root

# Update the system and install base and roboschool requirements
RUN apt-get update -y && apt-get install -y git xvfb ffmpeg libgl1-mesa-dev libharfbuzz0b libpcre3-dev libqt5x11extras5 build-essential

# Switch back to unprivileged user for python packages. User is defined in base docker image
USER $NB_USER

# Install TensorFlow and NumPy with pip to prevent using the MKL version which in this implementation is slower
# TensorFlow must be 1.14.0 because after that the needed time for 'predict_on_batch' on the model increases
# gast==0.2.2 and numpy==1.16.6 fix warnings which result in the used TensorFlow and gast versions
RUN pip install --quiet \
    gast==0.2.2 \
    tensorflow==1.14.0 \
    numpy==1.16.6 \
    gym \
    pybullet \
    matplotlib \
    pandas \
    ipywidgets

RUN conda install --yes -c conda-forge nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR work/evolution-strategies/

ADD hashed_password.txt .
ADD launch.sh .

CMD ["/bin/bash", "launch.sh"]
