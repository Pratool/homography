#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
from rasterization import Rasterizer
from transformation import multiply
from transformation import TransformGenerator
import argparse
import DirectLinearTransform
import json
import math
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def main():
    args = _getParsedArgs(sys.argv[1:])

    if args.p:
        _pointPicker(args.input_image)
        sys.exit(0)

    transformImage(
        args.input_image,
        args.output_image,
        _readCorrespondences(args.correspondences)
    )

def transformImage(input_image_path, output_image_path,
                      corresponding_points):

    im = mpimg.imread(input_image_path)

    transform_matrix = DirectLinearTransform.computeTransform(
        corresponding_points
    )
    image_rasterization = Rasterizer(
        im,
        transformation_matrix = transform_matrix
    )
    matplotlib.image.imsave(
        output_image_path,
        image_rasterization.rasterize()
    )

def _pointPicker(input_image_path):
    """ Utility function to select coordinates on image """
    image = mpimg.imread(input_image_path)
    onclick = lambda ev: print(ev.xdata, ev.ydata)
    fig = plt.figure()
    axes = plt.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def _readCorrespondences(correspondenceFilePath):
    with open(correspondenceFilePath, 'r') as correspondenceFileHandler:
        return json.load(correspondenceFileHandler)

def _getParsedArgs(args):
    parser = argparse.ArgumentParser(
        description = "CLI input to homography application"
    )

    parser.add_argument(
        "-p",
        action = "store_true",
        help = "use point picker utility")
    parser.add_argument(
        "--input-image",
        default = "./media/t2.png",
        help = "input image to be transformed")
    parser.add_argument(
        "--output-image",
        default = "output.png",
        help = "output image path for saving new image")
    parser.add_argument(
        "--correspondences",
        default = "3.json",
        help = "corresponding set of points to derive transform")

    return parser.parse_args(args)

if __name__ == "__main__":
    main()
