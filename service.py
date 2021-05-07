import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import torch
assert torch.__version__.startswith("1.8")   # need to manually install torch 1.8 if Colab changes its default version

# import some common libraries
import numpy as np
import os, json, cv2, random, io , base64

import detectron2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.logger import setup_logger
setup_logger()

from flask import Flask, jsonify, abort,render_template,request,redirect,url_for

from werkzeug.utils import secure_filename
from PIL import Image , ImageColor
from PIL.ExifTags import TAGS
import piexif
import logging, logging.handlers, sys, time, requests
from textblob import TextBlob


predictor = None
segmodel = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
#segmodel = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

def _create_names_dict(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    dict = []
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    for i, (l, s) in enumerate(zip(labels, scores)):
#        dict.append([l,"{:.2f}".format(s * 100)])
        dict.append(
            {
                "name" : l, 
                "score" : "{:.2f}".format(s * 100)
            })

    return dict


def preparePredictor():
    log.info('Prepearing predictor...')
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    
    log.debug(' Using model ' + segmodel)
    log.info('  You can find more models in model-zoo: https://github.com/facebookresearch/detectron2/tree/master/configs/COCO-InstanceSegmentation')
    cfg.merge_from_file(model_zoo.get_config_file(segmodel))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(segmodel)
    cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)
    log.info('Predictor ready!')
    return predictor


def analizeImg (image):
    global predictor
    if predictor is None:
        predictor =  preparePredictor()

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    outputs = predictor(image)

    metadata = MetadataCatalog.get("coco_2017_val")
    names = metadata.get("thing_classes", None)
    segments =[]
    if 'panoptic_seg' in outputs:
        
        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
        segments_info = outputs['panoptic_seg'][1]
        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            if not segments_info[i]["isthing"]:
                name = meta.stuff_classes[c]
                segments.append(name)


    classes = outputs["instances"].pred_classes
    scores = outputs["instances"].scores

    label2 = _create_names_dict(classes,scores,names,None)
    log.debug('Labels: %s', label2)

    return label2, segments, outputs

def getSegmentetdImage(image:Image, outputs):
    """
    Args:
        image: 
        outputs: segments and objects

    Returns:
        str: base64 decoded image
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
   
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    res_im = Image.fromarray(v.get_image())

    rawBytes = io.BytesIO()
    res_im.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return str(img_base64)

def getLabesShortList(labels:{}):
    res = list(dict.fromkeys([ e['name'] for e in labels ]))
    return res

def getExif(image:Image):
    exif = image.getexif()

    labeled = {}
    for (key, val) in exif.items():
        if TAGS.get(key) is not None:
            labeled[TAGS.get(key)] = str(val)

    return labeled



def getExif2(image:Image):
    tags = {}
    if 'exif' not in image.info:
        return tags

    exif_dict = piexif.load(image.info["exif"])
    for ifd in ("0th", "Exif", "GPS", "1st"):
        for tag in exif_dict[ifd]:
            if piexif.TAGS[ifd][tag]["name"] is not None:
                if isinstance(exif_dict[ifd][tag], (bytes)):
                    try: 
                        val = exif_dict[ifd][tag].decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        val = exif_dict[ifd][tag].decode('unicode-escape')
                        pass
                else:
                    val = exif_dict[ifd][tag]
                tags[piexif.TAGS[ifd][tag]["name"]] = val
    if 'XPKeywords' in tags:
        keyword = tags['XPKeywords']
        tags['XPKeywords'] = decodeXP(keyword)

    if 'XPComment' in tags:
        keyword = tags['XPComment']
        tags['XPComment'] = decodeXP(keyword)
    return tags

def decodeXP(t):
    """
    Takes a exif XPKeywords tag and decodes it to a text string
    """
    b = bytes(t)
    return b[:-2].decode('utf-16-le')

def decode_to_deg(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])

    return d + (m / 60.0) + (s / 3600.0)


def reverse_geocoding(lat, lon, language='en'):
    """
    https://nominatim.org/release-docs/develop/api/Reverse/
    """
    log.info('Using reverse geocoding from https://nominatim.org/release-docs/develop/api/Reverse/')
    res ={}
    lat = decode_to_deg(lat)
    lon = decode_to_deg(lon)

    # zoom = building
    zoom = 18
    # Output types: https://nominatim.org/release-docs/develop/api/Output/
    ftype = 'geocodejson'

    url = 'https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format={format}&zoom={zoom}&addressdetails=1&accept-language={lang}'
    url = url.format(lat=lat,lon=lon,zoom=zoom,lang=language, format=ftype)
    response = requests.get(url)
    if response.status_code == 200:
        log.debug('Got 200')
        res['code'] = 200
    else:
        log.debug('Not Found.')
        res['code'] = response.status_code
        
    jres = json.loads(response.text)
    res['data'] = jres

    return res


def rotate(img, args, exif):
    log.debug('Rotation: %s', args['rotation'])

    delta = 0
    angles = {
        '1' : 0,
        '2' : 0,
        '3' : 180,
        '4' : 180,
        '5' : 90,
        '6' : 90,
        '7' : 270,
        '8' : 270
        }
    if 'Orientation' in exif:
        delta = angles[str(exif['Orientation'])]
    log.debug('Autorotation: %s', delta )
    angle = -(int(args['rotation'])+delta)
    log.debug('Total angle: %s: ', angle )
    img = img.rotate(angle, expand=1) 
    return img


def resize(img, size):
    maxwh = max(img.width, img.height)
    if maxwh <= size:
        log.debug('Keeping original imaze size')
    else:
        max_ratio = size / maxwh
        size = (int(img.width*max_ratio), int(img.height*max_ratio))
        log.debug('New size: %s', size)
        img = img.resize(size)
    return img

def getColors(img, num):
    res = []

    q = img.quantize(colors=num,method=0)
    p = q.getpalette()[:num*3]
    return res



#############################################################################
###                WebService methods                                     ###
#############################################################################
app = Flask(__name__, template_folder='html')
app.config['JSON_SORT_KEYS'] = False

def prepareArgs(args):
    reqArgs = reqArgsDef.copy()

    for arg in reqArgs:
        
        if arg in args:

            if type(reqArgs[arg]) is int and args[arg] is not '0' and args[arg] is not '':
                reqArgs[arg] = args[arg]
            if type(reqArgs[arg]) is bool and args[arg] is not False:
                reqArgs[arg] = True
            if type(reqArgs[arg]) is str  and args[arg] is not '':
                reqArgs[arg] = args[arg]

    log.debug('Request prepeared args: %s', reqArgs)
    return reqArgs

reqArgsDef = {
    'autorotation': False,
    'rotation': 0,
    'exif' : False,
    'resimg': False,
    'resize' : 1000,
    'geodata' : False,
    'lang' : 'en',
    'translate' : False,
    'colors' : False
}


def getHelp():
    h ={}
    h['github url'] = 'https://github.com/mrekin/detectron2_cpu_img'
    h['default args'] = reqArgsDef
    h['model'] = segmodel
    h['usage'] = 'curl --request POST -F "file=@IMG.JPG" localhost:5000/api/v1.0/imgrecognize/?exif=False&autorotation&rotation=90'
    return h


######################################################################
###########  URLS
######################################################################

@app.route('/')
def index():
    return render_template('index.html')



# 
@app.route('/api/v1.0/imgrecognize/', methods=['POST','GET'])
def upload_file():
    start_time = time.time()

    resp ={}
    respInfo ={}
    respInfo['model'] = segmodel

    log.debug('Request: %s',request)

    if 'help' in request.args:
        resp['help'] = getHelp()
        return jsonify(resp)

    if request.method == 'POST':
        reqArgs = prepareArgs(request.args)
        respInfo['args'] = reqArgs
        log.info(request.files.keys)
        for fn in request.files:
            file = request.files[fn]
            if file:
                filename = secure_filename(file.filename)
                resp[filename] = {}
                img = Image.open(file)

                exif = {}
                if reqArgs['exif'] or reqArgs['autorotation'] or reqArgs['geodata']:
                    log.info('Extracting exif data..')
                    exif = getExif2(img)
                       
                if reqArgs['rotation'] is not 0 or reqArgs['autorotation'] is True:
                    log.info('Rotating image..')
                    img = rotate(img, reqArgs, exif)

                if reqArgs['resize'] is not 0:
                    log.info('Resizing image..')
                    img = resize(img, reqArgs['resize'])

                if reqArgs['colors']:
                    log.info('Getting colors..')
                    img = getColors(img, 3)

                if reqArgs['geodata'] is True and 'GPSLatitude' in exif:
                    log.info('Getting geodata..')
                    resp[filename]['geodata'] = reverse_geocoding(exif['GPSLatitude'],exif['GPSLongitude'], reqArgs['lang'])
                
                log.info('Analizing image..')
                resp[filename]['objects'] , resp[filename]['segments'],  out = analizeImg(img)
                resp[filename]['objectsShortList'] = getLabesShortList(resp[filename]['objects'])

                if reqArgs['translate'] is True and reqArgs['lang'] is not 'en' and resp[filename]['objectsShortList'] + resp[filename]['segments']:
                    log.info('Translating LabelsShortList and Segments..')
                    blob = TextBlob(','.join(map(str, resp[filename]['objectsShortList'] + resp[filename]['segments'])))
                    res = blob.translate(to=reqArgs['lang'])
                    resp[filename]['objectsAndSegments_'+reqArgs['lang']] = res.split(', ')

                if reqArgs['exif']:
                    resp[filename]['exif'] = exif

                if reqArgs['resimg']:
                    resp[filename]['img_res'] = getSegmentetdImage(img, out)

                

    totalTime = time.time() - start_time
    respInfo['exectime'] = totalTime
    log.debug('Exec time: %s sec', totalTime)
    log.debug('Done')
    
    result = {}
    result['data'] = resp
    result['_info'] = respInfo

    return jsonify(result)


    


if __name__ == '__main__':

    if 'SEGMENTATION_MODEL' in os.environ:
        segmodel = os.environ['SEGMENTATION_MODEL']
    if 'FLASK_DEBUG' in os.environ:
        fl_debug = os.environ['FLASK_DEBUG']
    else:
        fl_debug = False
    if 'FLASK_HOST' in os.environ:
        fl_host = os.environ['FLASK_HOST']
    else:
        fl_host = '0.0.0.0'
    if 'FLASK_PORT' in os.environ:
        fl_port = os.environ['FLASK_PORT']
    else:
        fl_port = 5000
    if 'LOG_LVL' in os.environ:
        srv_loglvl = os.environ['LOG_LVL']
    else:
        srv_loglvl = 'DEBUG'
    if 'SRV_LANG' in os.environ:
        reqArgsDef['lang'] = os.environ['SRV_LANG']
    
    
    rfh = logging.handlers.RotatingFileHandler(
        filename='log/service.log', 
        mode='a',
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8',
        delay=0
    )

    logging.basicConfig( 
        format="[%(asctime)s] %(levelname)-8s [%(thread)d.%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[
                rfh
                ]
        )
    log = logging.getLogger("SERVICE")
    log.setLevel(srv_loglvl)
    log.addHandler(logging.StreamHandler(sys.stdout))

    log.info('Starting service:\n  HOST: %s, PORT: %s, MODEL: %s',fl_host,fl_port,segmodel)
    predictor =  preparePredictor()
    app.debug=fl_debug
    app.run(host='0.0.0.0',port=5000)
