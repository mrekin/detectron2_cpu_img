# detectron2_cpu_img
This is 'development' docker image with installed:
* detectron2 for CPU
* python3.7 and libs
* torch 1.8.1
* simple web service, which get image as input and returns detectron2 instanseSegmentation result and exif info (Python/Flask)

# Note
I don't find 'ready to use' solution for my task (detect objects on image) so I had to build it

I don't know much about CV/image recognition/python/ docker, so there may be errors and duck code. __Any help is welcome__.

I don't have CUDA GPU, so I build CPU based service which is more slower.

# Usage
Build and run docker image:

`docker-compose up -d` or `docker-compose up -d --build` (to re-build image)

Service (`<your_host>:<your_port>/api/v1.0/imgrecognize/`) will started with docker up. Some time will spend for segmentation model downloading at start up (onse per model since last image build)

Post image any way you preffer, some thing like:

`curl --request POST -F "file=@IMG.JPG" localhost:5000/api/v1.0/imgrecognize/`

-- get json as result

# Performance
My HW instanse is
* Xeon E3 1260L
* DDR3 16Gb
* HDD 3Tb WD Red

Software:
* VM (Proxmox) with 6 cores (host) and 1.5Gb Ram
** Ubuntu 20.04 inside VM
** Docker 20.10.5
```
 time  `curl --request POST -F "file=@IMG_3448.JPG" 192.168.1.111:5000/api/v1.0/imgrecognize/ > /dev/null`
 
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 3452k  100 52765  100 3401k   4940   318k  0:00:10  0:00:10 --:--:-- 15201

real    0m10.698s
user    0m0.009s
sys     0m0.023s
```
