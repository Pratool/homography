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
    transformImage(
        args.input_image,
        args.output_image,
        _readCorrespondences(args.correspondences)
    )

def transformImage(input_image_path, output_image_path,
                      corresponding_points):
    transform_matrix = DirectLinearTransform.computeTransform(
        corresponding_points
    )

    im = mpimg.imread(input_image_path)
    image_rasterization = Rasterizer(
        im,
        transformation_matrix = transform_matrix
    )
    matplotlib.image.imsave(
        output_image_path,
        image_rasterization.rasterize()
    )

def _readCorrespondences(correspondenceFilePath):
    with open(correspondenceFilePath, 'r') as correspondenceFileHandler:
        return json.load(correspondenceFileHandler)

def _getParsedArgs(args):
    parser = argparse.ArgumentParser(
        description = "CLI input to homography application"
    )

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
