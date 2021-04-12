FROM debian:buster-slim
RUN apt-get update -y; 
RUN apt-get upgrade -y;
#RUN apt-get install -y python3.8;
#RUN pip3 install pip;

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip python3-opencv wget


RUN python3 --version;
#RUN apt-get install -y python3-pip
#RUN pip3 install pip;
RUN pip3 install --upgrade pip
RUN pip install -U numpy

# Detectron2 prerequisites
RUN pip install -U torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Detectron2 - CPU
RUN pip install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
RUN pip install -U cython flask flask-cors==3.0.9 requests opencv-python \
	Image \
	piexif \
;
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install -U cython requests opencv-python pyyaml==5.1
# Revert back pycocotools==2.0.2 - need to check why it is moved to 2.0
RUN pip install -U pycocotools==2.0.2

# Python flask service
RUN mkdir /home/detec_test
ADD service.py /home/detec_test
RUN mkdir /home/detec_test/html
COPY html/ /home/detec_test/html
RUN mkdir /home/detec_test/static
RUN mkdir /home/detec_test/static/js
COPY static/js/ /home/detec_test/static/js

# CMD python3 /home/detec_test/service.py
WORKDIR /home/detec_test/
CMD ["/usr/bin/python3","service.py"]