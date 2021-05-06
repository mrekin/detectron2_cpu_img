FROM debian:buster-slim
RUN apt-get update -y; 
RUN apt-get upgrade -y;

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip python3-opencv wget

RUN python3 --version;
RUN pip3 install --upgrade pip

# Detectron2 prerequisites
RUN pip install -U torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Detectron2 - CPU
RUN pip install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
RUN pip install -U cython \
	flask \
	flask-cors \
	requests \
	opencv-python \
	Image \
	piexif \
	textblob \
	pyyaml \
	numpy \
;

# Python service
RUN mkdir -p /home/detec_srv /home/detec_srv/html /home/detec_srv/log /home/detec_srv/static/js
COPY service.py /home/detec_srv
COPY html/ /home/detec_srv/html
COPY static/js/ /home/detec_srv/static/js

WORKDIR /home/detec_srv/
CMD ["/usr/bin/python3","service.py"]