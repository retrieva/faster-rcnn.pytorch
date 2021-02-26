import cv2
import json
import numpy as np
import torch
from torch.autograd import Variable
import urllib.request

from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file
from model.utils.blob import im_list_to_blob


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _load_category(category_file):
    category = {}
    with open(category_file) as f:
        for i in f.readlines():
            splited = i.split("\t")
            entry = {}
            entry["keyword"] = splited[1].strip()
            entry["type"] = splited[2].strip()
            entry["num"] = int(splited[3].strip())
            category[splited[0].strip()] = entry
    return category


class LandmarkDetector:

    def __init__(self, model_path, cfg_file, class_file, apikey, category_file, cuda=True):
        # initialization of faster-rcnn.pytorch
        cfg_from_file(cfg_file)
        cfg.USE_GPU_NMS = cuda

        np.random.seed(cfg.RNG_SEED)

        self.classes = ['__background__']
        with open(class_file) as f:
            for line in f.readlines():
                self.classes.append(line.strip())
        self.classes = np.asarray(self.classes)

        self.rcnnmodel = resnet(self.classes, 101, pretrained=False)
        self.rcnnmodel.create_architecture()
        if cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=(
                lambda storage, loc: storage))
        self.rcnnmodel.load_state_dict(checkpoint['model'])

        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if cuda > 0:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        with torch.no_grad():
            self.im_data = Variable(self.im_data)
            self.im_info = Variable(self.im_info)
            self.num_boxes = Variable(self.num_boxes)
            self.gt_boxes = Variable(self.gt_boxes)

        if cuda > 0:
            cfg.CUDA = True

        if cuda > 0:
            self.rcnnmodel.cuda()

        self.rcnnmodel.eval()
        self.thresh = 0.5

        # initialization of GCP
        self.baseurl = self._initializeGCP(apikey)
        self.category = _load_category(category_file)

    def _initializeGCP(self, apikey):
        baseurl = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        fields = "fields=" + "review"
        radius = "&radius=" + "200"
        rankby = "&rankby=prominence"
        language = "&language=ja"
        key = "&key=" + apikey
        return baseurl + fields + radius + rankby + language + key

    def detectLandmark(self, im_in, latitude, longitude):
        # im_in = np.array(imread(im_file))

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2],
                                im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self.gt_boxes.resize_(1, 1, 5).zero_()
            self.num_boxes.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.rcnnmodel(self.im_data, self.im_info,
                                        self.gt_boxes, self.num_boxes)

        scores = cls_prob.data
        scores = scores.squeeze()

        labels = []
        for j in range(1, len(self.classes)):
            inds = torch.nonzero(scores[:, j] > self.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                labels.append(self.classes[j])
        landmarks = []
        for label in labels:
            if label in self.category:
                landmark = self._searchLandmark(latitude,
                                                longitude,
                                                self.category[label])
                if landmark:
                    landmarks.extend(landmark)
        return landmarks

    def _searchLandmark(self, lat, lon, search_info):
        location = "&location={},{}".format(lat, lon)
        keyword = "&keyword=" + urllib.parse.quote(search_info["keyword"])
        request_type = "&type=" + search_info["type"]
        url = self.baseurl + location + keyword + request_type
        uh = urllib.request.urlopen(url)
        data = uh.read().decode()
        js = json.loads(data)
        if len(js["results"]) == 0:
            return []
        else:
            return [js["results"][i]["name"] for i in range(search_info["num"])]
