# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:34:37 2019

@author: 群青雨
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import subprocess
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import dlib
import argparse
from collections import OrderedDict


input_path = 'C:/Users/Lizhaoyan/Desktop/NLP/lip reading/'
result_path = input_path + 'result/'

#import Neural Network model
model_path = input_path + 'lip_reading_model.h5'
model = keras.models.load_model(model_path)

def find_minimum_rect(x_axis_, y_axis_):
    """
    find the minimum bounding rectangle of points.
    
    :param x_axis_: the points' x-axis coordinates.
    :param y_axis_: the points' y-axis coordinates.
    :return: left upper point and right lower point, which are (min_x, min_y), (max_x, max_y).
    """
    min_x = min(x_axis_)
    min_y = min(y_axis_)
    max_x = max(x_axis_)
    max_y = max(y_axis_)
    return min_x, min_y, max_x, max_y


def predict_an_image(detector_, predictor_, image__path_, results_dir_):
    """
    predict an image's mouth's landmarks using dlib, and find the minimum bounding rectangle,
    write the results into results_dir_.
    
    :param detector_: dlib's detector.
    :param predictor_: dlib's predictor.
    :param image__path_: a single image's file path.
    :param results_dir_: directory where the results will be wrote to.
    :return: a python OrderedDict object containing both the points and rectangle.
    """
    img = np.array(Image.open(image__path_))
    dets = detector_(img, 1)
    if len(dets) > 0:
        shape_ = predictor_(img, dets[0])
        points_dict_ = OrderedDict()
        x_axis = []
        y_axis = []
        for b in range(49, 68):  # get the mouth's landmarks only.
            x_axis.append(shape_.part(b).x)
            y_axis.append(shape_.part(b).y)
            points_dict_['point' + str(b)] = [shape_.part(b).x, shape_.part(b).y]
        
        rect = find_minimum_rect(x_axis, y_axis)  # get the minimum bounding rectangle.
        points_dict_['minimum_rect'] = rect

#        resultpath = results_dir_ + '/' + os.path.basename(image__path_) + '.json'
#        json.dump(points_dict_, open(resultpath, 'w'))
    else:
        print('Fail to detect face of image: {}, return -1...'.format(image__path_))
        return -1
    return points_dict_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed, can be a directory or a single image.")
    parser.add_argument("--model", default='./shape_predictor_68_face_landmarks.dat', help="model to be executed")
    parser.add_argument("--results", default='./results', help="where the results will be saved, must be a directory.")
    
    args = parser.parse_args()
    
#    try:
#        model_file = args.model
#        image_file_path = args.image
#        results_dir = args.results
#        if None in [model_file, image_file_path, results_dir] or os.path.isdir(results_dir):
#            raise ValueError
#    except ValueError:
#        print("You missed args! They are as following:\n\t{}".format(parser.format_help()))
#        exit(0)
model_file = input_path + 'shape_predictor_68_face_landmarks.dat'

results_dir = input_path
    
#if not os.path.exists(results_dir):
#    os.makedirs(results_dir)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_file)

dim = (100,50)


#Extract every frames from moive files
mov_path = input_path + 'mov/'
os.chdir(mov_path)
for root, dirs1, files1 in os.walk(mov_path):
   i = 0
   for file1 in files1:
        path = mov_path  + files1[i][0:len(files1[i])-4] + '/'
        os.makedirs(path)
        name = path +files1[i][0:len(files1[i])-4]+'_%04d.jpg'
        subprocess.call(['ffmpeg', '-i', file1, name])
#detect the mouth area
        for root1, dirs, files in os.walk(path):
            for file in files:
                image_file_path = path+file
                img = cv2.imread(image_file_path)
        
                if os.path.isfile(image_file_path):
                    result = predict_an_image(detector, predictor, image_file_path, results_dir)
        
                elif os.path.isdir(image_file_path):
                    for path, subdir, filelnames in os.walk(image_file_path):
                        for filename in filelnames:
                            if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
                                image_path = os.path.join(path, filename)
                                result = predict_an_image(detector, predictor, image_file_path, results_dir)
        
                else:
                    print('Wrong image path! Can be a directory or single image.')
        
#use the detected area to crop the mouth from the picture
                x,y,w,h = result.get('minimum_rect')
                img = img[y:h, x:w]
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(image_file_path, img)
#import the mouth pictures as array
        pic = []
        for root, dirs, files0 in os.walk(path):
            for file0 in files0:
                image_file_path = path + file0
                image = Image.open(image_file_path)
                image = image.convert('LA')
                pic.append(asarray(image)[:,:,0])
        pic = np.asarray(pic)
#use pre_trained model to make prdiction
        predictions = model.predict(pic)
#name and save the result        
        result_name = 'NeuralNetwork_Jeffersmapping_' + files1[i][0:len(files1[i])-4] + '.txt'
        np.savetxt(result_path + result_name, predictions)
        i += 1
        


##this is the map from result to viseme
# map_num = {
#         "A" : "0",
#         "B" : "1",
#         "C" : "2",
#         "D" : "3",
#         "E" : "4",
#         "F" : "5",
#         "G" : "6",
#         "H" : "7",
#         "I" : "8",
#         "J" : "9",
#         "K" : "10",
#         "S" : "11",
#         }