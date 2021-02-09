import argparse
import numpy as np
from scipy.misc import imread

import detect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Landmark Detector test')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--model', dest='model_path',
                        help='path to load models',
                        default="res101.pth")
    parser.add_argument('--class_file', dest='class_file',
                        help='path to category file',
                        default="object_vocab.txt")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--category', dest='category',
                        help='Category file of Landmark',
                        default='category.txt')
    parser.add_argument('apikey',
                         help='apikey of Google Place API')
    parser.add_argument('lat', type=float,
                        help="latitude of where picture is taken")
    parser.add_argument('lon',  type=float,
                        help="longitude of where picture is taken")
    parser.add_argument('images', nargs='+')
    args = parser.parse_args()

    ld = detect.LandmarkDetector(args.model_path, args.cfg_file,
                                 args.class_file, args.apikey,
                                 args.category, args.cuda)
    for img in args.images:
        im_in = np.array(imread(img))
        print(img, ld.detectLandmark(im_in, args.lat, args.lon))
