# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
from CRAFT import craft_utils
from CRAFT import imgproc
from CRAFT import file_utils
from collections import OrderedDict
from MORAN.models.moran import MORAN
from torchvision import transforms

# Load the CRAFT model and utility functions
from CRAFT.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Text Extraction and Classification Pipeline')
parser.add_argument('--craft_model', default='./CRAFT/weights/craft_mlt_25k.pth', type=str, help='CRAFT pretrained model')
parser.add_argument('--moran_model', default='./MORAN/checkpoints/best_model.pth', type=str, help='MORAN pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./inputs/', type=str, help='folder path to input images')
parser.add_argument('--result_folder', default='./results/', type=str, help='folder path to save results')

args = parser.parse_args()

# Define transforms for preprocessing for MORAN
transform = transforms.Compose([
    transforms.Resize((32, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_moran_model(checkpoint_path, device):
    # Initialize the MORAN model
    model = MORAN(nc=1, nclass=2, nh=256, targetH=32, targetW=100)
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model

def predict_text_class(image, model, device):
    # Load and preprocess the image
    image = Image.fromarray(image).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image, None, None, None, test=True)  # Test mode
        _, predicted = torch.max(outputs, 1)
    
    # Map class indices to labels
    labels = {0: 'Handwritten', 1: 'Printed'}
    predicted_label = labels[predicted.item()]
    return predicted_label

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def annotate_image(image, boxes, predictions):
    for box, prediction in zip(boxes, predictions):
        x_min = int(min(box[:, 0]))
        y_min = int(min(box[:, 1]))
        x_max = int(max(box[:, 0]))
        y_max = int(max(box[:, 1]))

        if prediction == 'Handwritten':
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image

if __name__ == '__main__':
    # Create result folder if it doesn't exist
    if not os.path.isdir(args.result_folder):
        os.mkdir(args.result_folder)

    # Load CRAFT model
    net = CRAFT()
    print('Loading weights from checkpoint (' + args.craft_model + ')')
    if args.cuda:
        print("Using Cuda")
        net.load_state_dict(copyStateDict(torch.load(args.craft_model)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        print("Using CPU")
        net.load_state_dict(copyStateDict(torch.load(args.craft_model, map_location='cpu')))

    net.eval()

    # Load MORAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moran_model = load_moran_model(args.moran_model, device)

    t = time.time()

    # Load images
    image_list, _, _ = file_utils.get_files(args.test_folder)

    for k, image_path in enumerate(image_list):
        print("Processing image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)

        # Extract text regions using CRAFT
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

        # Classify each text region using MORAN
        predictions = []
        for box in bboxes:
            x_min = int(min(box[:, 0]))
            y_min = int(min(box[:, 1]))
            x_max = int(max(box[:, 0]))
            y_max = int(max(box[:, 1]))

            cropped_img = image[y_min:y_max, x_min:x_max]
            if cropped_img.size == 0:
                continue

            prediction = predict_text_class(cropped_img, moran_model, device)
            predictions.append(prediction)

        # Annotate image with predictions
        annotated_image = annotate_image(image, bboxes, predictions)

        # Save annotated image
        result_image_path = os.path.join(args.result_folder, os.path.basename(image_path))
        cv2.imwrite(result_image_path, annotated_image)

    print("Elapsed time: {}s".format(time.time() - t))
