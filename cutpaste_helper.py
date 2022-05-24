#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:18:41 2020

@author: philip
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
from sklearn.decomposition import PCA
import time
import seaborn as sns
import cv2


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.backend import reshape
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from scipy.ndimage import zoom

from scipy.signal import find_peaks

sns.set()

def rescale_array(arr):
    '''
    Takes a 2D array and normalizes it such that the min value becomes
    0 and the max value becomes 1.
    '''
    return (arr-np.min(arr))/np.max(arr-np.min(arr))

def display_single_array(array, dpi=None, shape=None):
    
    if dpi:
        fig = plt.figure(dpi=200)
    elif shape:
        fig = plt.figure(figsize=shape)
    else:
        fig = plt.figure()
        
    num_rows = array.shape[0]
    plt.imshow(array.reshape((num_rows,-1)), cmap='gray', vmin=0.0, vmax=1.0)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return

def display_random_files(file_list, file_dir, dim=(64,90), rescale=True, indices=[]):
    '''
    Takes in a list of files, randomly selects subset of the files,
    and display the files.
    '''
    n = 5
    if not(indices):
        random_indices = np.random.randint(0, len(file_list), n)
    else:
        random_indices = indices
    plt.figure(figsize=(20,3))
    for i, j in enumerate(random_indices):
        ax = plt.subplot(1, n, i+1)
        path = os.path.join(file_dir, file_list[j])
        arr = np.fromfile(path, dtype=np.float32)
        arr = np.reshape(arr, dim)
        if rescale:
            arr = rescale_array(arr)
        plt.imshow(arr, cmap='gray', vmin=0.0, vmax=1.0)
        plt.title('{}'.format(file_list[j]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
    return list(random_indices)
    
 
def display_random_arrays(tensor, dim=(64,90), norm=True, indices=[], label=None):
    '''
    Takes in a tensor (samples, num_rows, num_cols, None), randomly selects a 
    subset of the arrays, and display the arrays.
    '''
    n = 5
    if not(indices):
        random_indices = np.random.randint(0, len(tensor), n)
    else:
        random_indices = indices
    if label:
        print(label)
    plt.figure(figsize=(20,3))
    for i, j in enumerate(random_indices):
        ax = plt.subplot(1, n, i+1)
        arr = tensor[j]
        arr = np.reshape(arr, dim)
        if norm:
            ax.imshow(arr, cmap='gray')
        else:
            ax.imshow(arr, cmap='gray', vmin=0.0, vmax=1.0)
        
        ax.set_title('{}'.format(j))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
    return list(random_indices)


def split_train_test_arrays(full_arrays, train_pct=0.7, seed=10):
    '''
    Takes in a dictionary of full-size TEM arrays and randomly assigns
    70% of the images to the training set. Note, this train test split it 
    done prior to generating random crops from each full-size array.
    '''

    train_n = int(train_pct * len(full_arrays))
    random.seed(seed)
    train_full_arrays = dict(random.sample(full_arrays.items(), train_n))
    test_full_arrays = {key:value for key, value in full_arrays.items() if 
                        not(key in train_full_arrays)}
    
    return train_full_arrays, test_full_arrays


def single_image_pca(full_arr, n_samples=500, dims=(64,90), n_components=500, name='dir', plot=False):
    '''
    Create samples from single large TEM image and train a PCA on these 
    samples.
    '''

    array_dict = {name: full_arr}
    samples_tensor = get_nondefect_samples(array_dict, row_spacing=dims[0], col_spacing=dims[1], num_reps=n_samples)
    samples_array = samples_tensor.reshape((n_samples, -1))

    arr = samples_array
    tensor = samples_tensor
    # Create PCA object with components number
    pca = PCA(n_components = n_components)
    pca = pca.fit(arr)
    encoded = pca.transform(arr)
    recon_array = pca.inverse_transform(encoded)
    recon_tensor = recon_array.reshape((-1, dims[0], dims[1], 1))
    resid_tensor = tensor - recon_tensor

    if plot:

        # Plot increasing variance explained as num components increases
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Effect of Number of Principal Components')
        plt.xlabel('number of compsonents')
        plt.ylabel('cumulative explained variance')
        plt.show()

        print('Original')
        plt.imshow(samples_tensor[0,:,:,0], vmax=1.0, vmin=0.0, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        print('PCA Reconstruction')
        plt.imshow(recon_tensor[0,:,:,0], vmax=1.0, vmin=0.0, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

        print('Residual')
        plt.imshow(resid_tensor[0,:,:,0], vmax=1.0, vmin=0.0, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return pca, resid_tensor

def get_pca_recon(pca_model, input_arr, dims=(64, 90)):

    input_vector = input_arr.reshape((1,-1))
    encoding = pca_model.transform(input_vector)
    decoding = pca_model.inverse_transform(encoding)
    recon = decoding.reshape((dims[0],dims[1],1))

    return recon

def get_nondefect_samples(full_arrays, segment_height=64, segment_width=90, num_reps=10):
    '''
    Takes in a "full-size" image stored as an array and creates random 
    crops of the image that are each 2*row_spacing by 2*col_spacing in size.
    The random crops ensure that the training set is robust to the starting 
    point of the upper left corner.
    '''
    
    samples = []
    
    sorted_keys = sorted(full_arrays.keys())
    for key in sorted_keys:
        array = full_arrays[key]
        array = array.reshape((array.shape[0], array.shape[1], 1))
        # Choose random point to be the upper left corner of the crop
        row_starts = np.random.randint(0, array.shape[0] - segment_height, size=num_reps)
        col_starts = np.random.randint(0, array.shape[1] - segment_width, size=num_reps)
        new_samples = [array[i:i+segment_height, j:j+segment_width :] for (i,j) in zip(row_starts, col_starts)]
        samples += new_samples
        
    return np.array(samples)


def get_defect_samples(nondefect_tensor, noise_pct=0.5):
    '''
    Takes in a tensor where each array is a TEM image with no defect. 
    For each array in the list, a defect is added to the array.
    '''
    
    # Fix the size of the defects being injected
    defect_radii = np.random.randint(5, 7, len(nondefect_tensor))

    # Randomly choose the corner where to start the defect
    defect_center_xs = np.random.randint(10, nondefect_tensor[0].shape[0]-10, len(nondefect_tensor))
    defect_center_ys = np.random.randint(10, nondefect_tensor[0].shape[1]-10, len(nondefect_tensor))
    defect_centers = list(zip(defect_center_xs, defect_center_ys))

    # Inject defects and store arrays
    defect_tensor = []
    for i, x in enumerate(nondefect_tensor):
      
        # insert round defect
        center_x, center_y = defect_centers[i]
        r = defect_radii[i] # radius of the circle
        min_row, min_col = center_x-r, center_y-r
        indices = [(i, j) for i in range(min_row, min_row + 2*r + 1) 
                    for j in range(min_col, min_col + 2*r + 1)] # create rectangular grid
        filter_indices = [(i,j) for (i,j) in indices 
                          if (((i-center_x)**2 + (j-center_y)**2)<=r**2)] # filter grid with circle
        defect_x = x.copy()
        max_pixel_value = np.max(x)
        min_pixel_value = np.min(x)
        pixel_range = max_pixel_value - min_pixel_value
        for (i,j) in filter_indices:
            distance_from_center = np.sqrt((i-center_x)**2 + (j-center_y)**2)/r
            pct_brightness = 1 - 0.2*distance_from_center 
            new_pixel_value = min_pixel_value + pct_brightness * noise_pct * pixel_range
            defect_x[i,j,0] = np.random.normal(new_pixel_value, 0.05) # add noise to each matrix value
        defect_x = np.minimum(defect_x, max_pixel_value)
        defect_x = np.maximum(defect_x, min_pixel_value)
        defect_tensor.append(defect_x)
        
    return np.array(defect_tensor)

def get_defect_samples_2(full_size_array, defect_type, row_spacing, col_spacing,
                        segment_height, segment_width, trim_width):
    
    defect_arrat, start_row, end_row, start_col, end_col, true_row, true_coladd_custom_defect(array, defect_type, 
                          row_spacing, col_spacing,
                          segment_height, segment_width,
                          trim_width)
    
    return

def add_defect_to_image(arr, trim=10, noise_pct=0.5, radius=None):
    '''
    Takes in an array where each array is a TEM image with no defect. 
    For each array in the list, a defect is added to the array.
    '''
    
    # Fix the size of the defects being injected
    defect_radii = np.random.randint(5, 7)
    if radius:
        defect_radii = radius
    
    # Randomly choose the corner where to start the defect
    center_x = np.random.randint(trim, arr.shape[0]-trim)
    center_y = np.random.randint(trim, arr.shape[1]-trim)
    
    x = arr.copy()
    r = defect_radii # radius of the circle
    defect_x = x.copy()
    max_pixel_value = np.max(x)
    min_pixel_value = np.min(x)
    pixel_range = max_pixel_value - min_pixel_value
    
    if defect_radii > 1:    
        # For radius>1 we create a round defect
        min_row, min_col = center_x-r, center_y-r
        indices = [(i, j) for i in range(min_row, min_row + 2*r + 1) 
                    for j in range(min_col, min_col + 2*r + 1)] # create rectangular grid
        filter_indices = [(i,j) for (i,j) in indices 
                          if (((i-center_x)**2 + (j-center_y)**2)<=r**2)] # filter grid with circle

        for (i,j) in filter_indices:
            distance_from_center = np.sqrt((i-center_x)**2 + (j-center_y)**2)/r
            pct_brightness = 1 - 0.2*distance_from_center 
            new_pixel_value = min_pixel_value + pct_brightness * noise_pct * pixel_range
            defect_x[i,j,0] = np.random.normal(new_pixel_value, 0.05) # add noise to each matrix value
    else:
        pct_brightness = 1
        new_pixel_value = min_pixel_value + pct_brightness * noise_pct * pixel_range
        print(defect_x[center_x, center_y, 0])
        defect_x[center_x, center_y, 0] = np.random.normal(new_pixel_value, 0.05)
        print(defect_x[center_x, center_y, 0])
    
    defect_x = np.minimum(defect_x, max_pixel_value)
    defect_x = np.maximum(defect_x, min_pixel_value)

    return defect_x, (center_x, center_y), defect_radii

def import_raw_images_to_full_arrays(image_dir):

    # Directories where simulated images from AFRL are stored
    proj_dir = '//Users//philip//Google Drive//Work//2019-2022 AFIT PhD//repositories//tem_imaging//project_2'
    file_dir = os.path.join(proj_dir, 'raw_images_2020_10')
    files = os.listdir(file_dir)
    files = [i for i in files if ('GaAs' in i)]

    # Store simulated images as arrays in dictionary
    arrays = {}
    for f in files:
        path = os.path.join(file_dir, f)
        arr = np.fromfile(path, dtype=np.float32)
        arr = np.reshape(arr, (64, 45))
        arr = rescale_array(arr) #rescale images between 0 and 1
        arrays[f] = arr
        
    # Store full-size simulated images as arrays in dictionary
    full_arrays = {}
    for key, arr in arrays.items():
        full_arr = np.tile(arr, (5,10))
        full_arrays[key] = full_arr

    return full_arrays

def import_raw_images(image_path, dims):
    '''
    Inputs:
        image_path - a filepath that contains a folder of .bin files
        dims - dimension of each of the .bin images in image_path

    Outputs:
        arrays - a dictionary {file_name: arr}, each arr is of shape dims
    '''


    # Directories where binary files
    files = os.listdir(image_path)
    files = [i for i in files if ('.bin' in i)]

    # Store simulated images as arrays in dictionary
    arrays = {}
    for f in files:
        path = os.path.join(image_path, f)
        arr = np.fromfile(path, dtype=np.float32)
        arr = np.reshape(arr, (dims[0], dims[1], 1))
        arr = rescale_array(arr) #rescale images between 0 and 1
        key = f.split('.bin')[0]
        arrays[key] = np.rot90(arr)
        
    return arrays

def add_swap_defect(arr):
    '''
    Adds a defect consisting of swapping Ga and As. Note the defect is always at a fixed
    location in the input array
    '''

    start_col = 39
    end_col = 63

    start_row = 24
    end_row = 41

    # define part of the image to flip
    area_to_flip = arr[start_row:end_row, start_col:end_col]

    # create a copy of the original image 
    defect_sample = arr.copy()

    # flip the pixel values for a GaAs bond so that atoms are "backwards"
    defect_sample[start_row:end_row, start_col:end_col] = np.flip(area_to_flip, 1)

    return defect_sample


def add_enlarged_defect(arr, zoom_factor=1.5):
    '''
    Find one atom and enlarge it to mimic a defect where a foreign atom replaces
    a Ga atom
    '''
    start_col = 39
    end_col = 51 #Half the region used for swap

    start_row = 24
    end_row = 41

    region = arr[start_row:end_row, start_col:end_col]
    zoomed_region = cv2_clipped_zoom(region, zoom_factor)
    defect_sample = np.copy(arr)
    defect_sample[start_row:end_row, start_col:end_col] = zoomed_region

    return defect_sample


def get_custom_defect_samples(full_arrays, defect_type, row_spacing=32, col_spacing=45, num_reps=10):
    '''
    Takes in a "full-size" image stored as an array and creates random 
    crops of the image that are each 2*row_spacing by 2*col_spacing in size.
    The random crops ensure that the training set is robust to the starting 
    point of the upper left corner.

    defect_type is either 'swap' or 'replaced
    '''
    
    nondefect_samples = []
    defect_samples = []
    for name, array in full_arrays.items():
        mult = 2 #each crop is 2 times the width and heigh of a single lattice structure
        
        base_nondefect_array = array[0:row_spacing*mult, 0:col_spacing*mult]

        if defect_type == 'swap':
            base_defect_array = add_swap_defect(base_nondefect_array)
        elif defect_type == 'enlarged':
            base_defect_array = add_enlarged_defect(base_nondefect_array, zoom_factor=1.5)

        # Create 3x3 tiled array of nondefect and defect base image
        nondefect_full_arr = np.tile(base_nondefect_array, (3,3))
        defect_full_arr = np.tile(base_defect_array, (3,3))

        # Define region from which upper left corner can be randomly chosen
        start_region_upper_left = (48, 68)
        start_region_lower_right = (48 + 32, 68 + 56)

        # Choose random point to be the upper left corner of the crop
        row_starts = np.random.randint(start_region_upper_left[0],
                                       start_region_lower_right[0], 
                                       size=num_reps)
        col_starts = np.random.randint(start_region_upper_left[1],
                                       start_region_lower_right[1], 
                                       size=num_reps)

        nondefect_samples += [nondefect_full_arr[i:i+mult*row_spacing, j:j+mult*col_spacing, None] 
                              for (i,j) in zip(row_starts, col_starts)]
        defect_samples += [defect_full_arr[i:i+mult*row_spacing, j:j+mult*col_spacing, None] 
                           for (i,j) in zip(row_starts, col_starts)]

    return np.array(nondefect_samples), np.array(defect_samples)


def convert_tensor_to_df(tensor):
    '''
    Takes an list of arrays and converts them to a dataframe where each row
    represents an array and each column is one pixel value.
    '''
    df = pd.DataFrame.from_records(tensor.reshape(len(tensor), -1))
    return df


def display_pca_samples(train_nondefect_df, train_defect_df,
                        test_nondefect_df, test_defect_df, 
                        recon_train_nondefect_arr, recon_train_defect_arr,
                        recon_test_nondefect_arr, recon_test_defect_arr):
    
    # Plot examples from each of the 4 datasets
    plt.figure(figsize = (20,8))
    ax = plt.subplot(2, 4, 1)
    ax.set_title('train_nondefect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(train_nondefect_df.iloc[0].values.reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    ax = plt.subplot(2, 4, 5)
    ax.set_title('recon_train_nondefect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(recon_train_nondefect_arr[0].reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    
    ax = plt.subplot(2, 4, 2)
    ax.set_title('train_defect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(train_defect_df.iloc[0].values.reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    ax = plt.subplot(2, 4, 6)
    ax.set_title('recon_train_defect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(recon_train_defect_arr[0].reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    
    ax = plt.subplot(2, 4, 3)
    ax.set_title('val_nondefect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(test_nondefect_df.iloc[0].values.reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    ax = plt.subplot(2, 4, 7)
    ax.set_title('recon_val_nondefect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(recon_test_nondefect_arr[0].reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    
    ax = plt.subplot(2, 4, 4)
    ax.set_title('val_defect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(test_defect_df.iloc[0].values.reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    ax = plt.subplot(2, 4, 8)
    ax.set_title('recon_val_defect')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(recon_test_defect_arr[0].reshape(64,90),interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
    
    # doc_images = '/Users/philip/Google Drive/Work/2019-2022 AFIT PhD/repositories/tem_imaging/doc_images'
    # plt.savefig(os.path.join(doc_images,'pca_train_val_data.png'))
    plt.show()
    
    return

def plot_pca_results(train_nondefect_mse, train_defect_mse, 
                     test_nondefect_mse, test_defect_mse, image_save_path=None):
    '''
    Takes the MSE for the reconstruction of each training and test image
    and aggregates the results into figures and metrics.
    '''
    
    # Plot reconstruction errors of training images with and without defects
    plt.figure(figsize=(4,4), dpi=100)
    plt.scatter(train_nondefect_mse, train_defect_mse, alpha=0.2, label='train')
    plt.scatter(test_nondefect_mse, test_defect_mse, alpha=0.5, marker='1', label='test')
    plt.plot(range(int(np.max(test_defect_mse))), range(int(np.max(test_defect_mse))))
    plt.xlabel('nondefect_reconstruction_mse')
    plt.ylabel('defect_reconstruction_mse')
    plt.legend()
    
    if not(image_save_path):
        image_save_path = '/Users/philip/Google Drive/Work/2019-2022 AFIT PhD/repositories/tem_imaging'
    plt.savefig(os.path.join(image_save_path,'pca_detection_scatter.png'))
    plt.show()
    
    return 
    

def run_pca_analysis(train_nondefect_df, train_defect_df,
                     test_nondefect_df, test_defect_df, proj_dir,
                     n_components=200, plot_results = True,
                     return_accuracy=False):
    
    '''
    Fits a PCA on the training data. The fitted PCA transform is applied to 
    both training and test images with and without defects to measure
    reconstruction error.
    '''
    
    # Create PCA object with components number
    pca = PCA(n_components = n_components)
    
    # Fit transform on training set
    pca = pca.fit(train_nondefect_df)

    # Plot increasing variance explained as num components increases
    if plot_results:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Effect of Number of Principal Components')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    
    # Apply PCA transform to the training set
    pca_train_nondefect_arr = pca.transform(train_nondefect_df)
    pca_train_defect_arr = pca.transform(train_defect_df)
    pca_test_nondefect_arr = pca.transform(test_nondefect_df)
    pca_test_defect_arr = pca.transform(test_defect_df)
    
    # Inverse transform back to regular dataset 
    recon_train_nondefect_arr = pca.inverse_transform(pca_train_nondefect_arr)
    recon_train_defect_arr = pca.inverse_transform(pca_train_defect_arr)
    recon_test_nondefect_arr = pca.inverse_transform(pca_test_nondefect_arr)
    recon_test_defect_arr = pca.inverse_transform(pca_test_defect_arr)
    
    if plot_results:
        # Check dimensions of applying the transform and inverse transform
        print("Nondefect train shape",train_nondefect_df.shape)
        print("PCA nondefect train shape",pca_train_nondefect_arr.shape)
        print("Inverse PCA nondefect train shape",recon_train_nondefect_arr.shape)
        print("")
        
        # Inspect shape of each data set
        print('Train nondefect', np.shape(train_nondefect_df))
        print('Train defect', np.shape(train_defect_df))
        print('Val nondefect', np.shape(test_nondefect_df))
        print('Val defect', np.shape(test_defect_df))
        print("")
    
    # Compute reconstruction MSE
    train_nondefect_mse = ((train_nondefect_df - recon_train_nondefect_arr)**2).sum(axis=1)
    train_defect_mse = ((train_defect_df - recon_train_defect_arr)**2).sum(axis=1)
    test_nondefect_mse = ((test_nondefect_df - recon_test_nondefect_arr)**2).sum(axis=1)
    test_defect_mse = ((test_defect_df - recon_test_defect_arr)**2).sum(axis=1)
    
    if plot_results:
        print('Avg MSE for nondefect training images: {}'.format(np.mean(train_nondefect_mse)))
        print('Avg MSE for defect training images: {}'.format(np.mean(train_defect_mse)))
        print('Avg MSE for nondefect test images: {}'.format(np.mean(test_nondefect_mse)))
        print('Avg MSE for defect test images: {}'.format(np.mean(test_defect_mse)))
        
        display_pca_samples(train_nondefect_df, train_defect_df,
                            test_nondefect_df, test_defect_df, 
                            recon_train_nondefect_arr, recon_train_defect_arr,
                            recon_test_nondefect_arr, recon_test_defect_arr)
        
        image_save_path = os.path.join(proj_dir, 'doc_images')
        plot_pca_results(train_nondefect_mse, train_defect_mse, 
                         test_nondefect_mse, test_defect_mse, image_save_path)

    train_diff_mse = train_nondefect_mse - train_defect_mse
    train_num_correct = len(train_diff_mse[train_diff_mse < 0])
    train_num_samples = len(train_diff_mse)
    train_acc = train_num_correct/train_num_samples

    if plot_results:
        print('Train percent defect_mse > nondefect_mse: {} ({}/{})'.format(train_acc,
                                                                           train_num_correct,
                                                                           train_num_samples))
    
    test_diff_mse = test_nondefect_mse - test_defect_mse
    test_num_correct = len(test_diff_mse[test_diff_mse < 0])
    test_num_samples = len(test_diff_mse)
    test_acc = test_num_correct/test_num_samples

    if plot_results:
        print('Test percent defect_mse > nondefect_mse: {} ({}/{})'.format(test_acc, 
                                                                          test_num_correct, 
                                                                          test_num_samples))
        
    return (train_nondefect_mse, train_defect_mse,
            test_nondefect_mse, test_nondefect_mse), train_acc, test_acc

def add_gaussian_noise_to_tensor(tensor, wt=0.08):
    n, row, col, _ = tensor.shape
    noise_tensor = np.zeros(tensor.shape)

    for i in range(n):
        arr = tensor[i,:,:,:]
        noise_arr = add_gaussian_noise(arr, wt)
        noise_tensor[i,:,:,:]= noise_arr
    
    return noise_tensor

def add_gaussian_noise(arr, wt=0.08):
    if (np.max(arr) > 1.0) or (np.min(arr) < 0.0):
        print('Received unscaled array')
    row, col, _ = arr.shape
    # Gaussian distribution parameters
    mean = 0
    var = 1
    sigma = var ** 0.5
    # wt = wt + np.random.rand()/20
    gaussian = wt * np.random.normal(mean, sigma, size = (row, col, 1)).astype(np.float32)
    # gaussian_img = cv2.addWeighted(arr, (1-wt), gaussian, wt, 0)
    gaussian_img = arr + gaussian
    gaussian_img = rescale_array(gaussian_img)
    
    return gaussian_img
    
def noise_image_generator(tensor, batch_size=64, wt=0.08):
    '''
    Takes in a 4D data tensor (n, x_dim, y_dim, 1) and returns a random
    pair of tensors, batch_x and batch_y
    '''
    while True:
        # Select random set of indices for batch
        n = tensor.shape[0]
        batch_indices = np.random.choice(n, batch_size, replace=False)
        batch_tensor = tensor[batch_indices]
        noise_batch_tensor = add_gaussian_noise_to_tensor(batch_tensor, wt)

        yield(noise_batch_tensor, batch_tensor)


def vgg16_scaler(tensor):
    '''
    Takes a one channel array with outputs a 3 channel array with values
    scaled between 0 to 255
    '''

    scaled_tensor = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], 3))

    for i in range(tensor.shape[0]):
        a = tensor[i, :, :, 0]
        scaled_a = rescale_array(a) * 255.0
        scaled_tensor[i, :, :, 0] = scaled_a
        scaled_tensor[i, :, :, 1] = scaled_a
        scaled_tensor[i, :, :, 2] = scaled_a
       
    scaled_tensor = keras.applications.vgg16.preprocess_input(scaled_tensor, data_format='channels_first')
    
    return scaled_tensor

def pca_residual_image_generator(tensor, pca, batch_size=64, wt=0.05, no_residual=False, vgg16=False):
    '''
    Takes in a 4D data tensor (n, x_dim, y_dim, 1) and returns a random
    pair of tensors, batch_x and batch_y
    '''
    batch_size = int(batch_size/2.0)
    wt = np.random.normal(wt, 0.005)
    
    aug_datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0,
        zoom_range=0.0,
        fill_mode='nearest')

    while True:
        # Select random set of indices for batch
        n = int(tensor.shape[0])
        batch_indices = np.random.choice(n, batch_size, replace=False)

        # Nondefect tensors
        batch_shape = tensor.shape
        batch_tensor = tensor[batch_indices]
        if wt > 0.0:
            noise_batch_tensor = add_gaussian_noise_to_tensor(batch_tensor, wt)
        else:
            noise_batch_tensor = batch_tensor
                                        
#         aug_gen = aug_datagen.flow(noise_batch_tensor,
#                                    y=None,
#                                    batch_size=batch_size,
#                                    shuffle=False)
#         noise_batch_tensor = next(aug_gen)
        
        if not(no_residual):
            nondefect_pca_encoded = pca.transform(noise_batch_tensor.reshape((batch_size, -1)))
            nondefect_pca_recon = pca.inverse_transform(nondefect_pca_encoded).reshape((batch_size, batch_shape[1], batch_shape[2], batch_shape[3]))
            nondefect_resid = noise_batch_tensor - nondefect_pca_recon

        # Create defect tensors
        defect_batch_tensor = get_defect_samples(batch_tensor)
        if wt > 0.0:
            defect_noise_batch_tensor = add_gaussian_noise_to_tensor(defect_batch_tensor, wt)
        else:
            defect_noise_batch_tensor = defect_batch_tensor
#         aug_gen = aug_datagen.flow(defect_noise_batch_tensor,
#                                    y=None,
#                                    batch_size=batch_size,
#                                    shuffle=False)
#         defect_noise_batch_tensor = next(aug_gen)  
        if not(no_residual):
            defect_pca_encoded = pca.transform(defect_noise_batch_tensor.reshape((batch_size, -1)))
            defect_pca_recon = pca.inverse_transform(defect_pca_encoded).reshape((batch_size, batch_shape[1], batch_shape[2], batch_shape[3]))
            defect_resid = defect_noise_batch_tensor - defect_pca_recon
        
        if not(no_residual):
            input_tensor = np.concatenate((nondefect_resid, defect_resid), axis=0)
            label_tensor = np.concatenate((np.zeros(batch_size), np.ones(batch_size)))
        else:
            input_tensor = np.concatenate((noise_batch_tensor, defect_noise_batch_tensor), axis=0)
            label_tensor = np.concatenate((np.zeros(batch_size), np.ones(batch_size)))
        
        if vgg16:
            input_tensor = vgg16_scaler(input_tensor)
            
        
        yield(input_tensor, label_tensor)
        
        
def get_simple_autoencoder():
        
    input_img = Input(shape=(64, 90, 1))  # adapt this if using `channels_first` image data format
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 3), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 3), padding='same')(x)
    
    # at this point the representation is (8, 8, 16)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 3))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 3))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='Nadam', loss='binary_crossentropy')
    
    autoencoder.summary()
    
    return autoencoder

def evaluate_autoencoder_performance(model, history, nondefect_train_tensor,
    defect_train_tensor, nondefect_test_tensor, defect_test_tensor, proj_dir,
    model_name):
    '''
    Take trained CNN autoencoder and 
    '''

    # plot training and validation loss
    num_epochs = len(history['loss'])

    plt.plot(range(num_epochs), history['loss'], label='train loss', alpha=0.5)
    plt.plot(range(0,num_epochs, 5), history['val_loss'], c='r', label='val loss', alpha=0.5)
    plt.legend()
    plt.show()
    
    recon_nondefect_train_tensor = model.predict(nondefect_train_tensor)
    recon_defect_train_tensor = model.predict(defect_train_tensor)
    recon_nondefect_test_tensor = model.predict(nondefect_test_tensor)
    recon_defect_test_tensor = model.predict(defect_test_tensor)


    print('Training image samples (orig, recon, orig with defect, recon with defect).')
    indices = display_random_arrays(nondefect_train_tensor)
    _ = display_random_arrays(recon_nondefect_train_tensor, indices=indices)
    _ = display_random_arrays(defect_train_tensor, indices=indices)
    _ = display_random_arrays(recon_defect_train_tensor, indices=indices)


    print('Test image samples (orig, recon, orig with defect, recon with defect).')
    indices = display_random_arrays(nondefect_test_tensor)
    _ = display_random_arrays(recon_nondefect_test_tensor, indices=indices)
    _ = display_random_arrays(defect_test_tensor, indices=indices)
    _ = display_random_arrays(recon_defect_test_tensor, indices=indices)
    
    
    pixel_sq_error = (nondefect_train_tensor - recon_nondefect_train_tensor)**2
    nondefect_train_mse = np.sum(pixel_sq_error, axis=(1,2,3))

    pixel_sq_error = (defect_train_tensor - recon_defect_train_tensor)**2
    defect_train_mse = np.sum(pixel_sq_error, axis=(1,2,3))

    pixel_sq_error = (nondefect_test_tensor - recon_nondefect_test_tensor)**2
    nondefect_test_mse = np.sum(pixel_sq_error, axis=(1,2,3))

    pixel_sq_error = (defect_test_tensor - recon_defect_test_tensor)**2
    defect_test_mse = np.sum(pixel_sq_error, axis=(1,2,3))

    #plot squared errors of images with defects and those without defects
    fig = plt.figure(figsize=(4,4), dpi=150)
    plt.scatter(nondefect_train_mse, defect_train_mse, alpha=0.2, label='train')
    plt.scatter(nondefect_test_mse, defect_test_mse, alpha=0.5, marker='1', label='test')
    plt.plot(range(int(np.max(nondefect_train_mse))), range(int(np.max(nondefect_train_mse))))
    plt.xlabel('nondefect_reconstruction_mse')
    plt.ylabel('defect_reconstruction_mse')
    plt.legend()

    plt.savefig(os.path.join(proj_dir, 'doc_images','mse_scatter_{}.png'.format(model_name)))
    plt.show

    # vector of differences between reeconstruction error with and without defect
    train_error_diff = defect_train_mse - nondefect_train_mse
    test_error_diff = defect_test_mse - nondefect_test_mse

    # count number of images where defect error is greater than nondefect error
    num_correct = (train_error_diff>0).sum()
    num_samples = len(train_error_diff)
    print('Training set defect detection accuracy is {} ({}/{}).'.format(round(num_correct/num_samples,3), 
                                                                         num_correct,
                                                                         num_samples))

    # count number of images where defect error is greater than nondefect error
    num_correct = (test_error_diff>0).sum()
    num_samples = len(test_error_diff)
    print('Val set defect detection accuracy is {} ({}/{}).'.format(round(num_correct/num_samples,3), 
                                                                    num_correct,
                                                                    num_samples))
    return (nondefect_train_mse, defect_train_mse, nondefect_test_mse, defect_test_mse)



def get_cnn_classifier_model(segment_height=64, segment_width=90):

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(segment_height, segment_width, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
#     model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    print(model.summary())

    return model
    
def get_pca_accuracy(n_components, ):

    # Create PCA object with components number
    pca = PCA(n_components = n_components)

    # Fit transform on training set
    pca = pca.fit(train_nondefect_df)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('number of components')
#     plt.ylabel('cumulative explained variance')
#     plt.show()

    # Apply PCA transform to the training set
    pca_train_nondefect_arr = pca.transform(train_nondefect_df)
    pca_train_defect_arr = pca.transform(train_defect_df)
    pca_val_nondefect_arr = pca.transform(val_nondefect_df)
    pca_val_defect_arr = pca.transform(val_defect_df)

    # Inverse transform back to regular dataset 
    recon_train_nondefect_arr = pca.inverse_transform(pca_train_nondefect_arr)
    recon_train_defect_arr = pca.inverse_transform(pca_train_defect_arr)
    recon_val_nondefect_arr = pca.inverse_transform(pca_val_nondefect_arr)
    recon_val_defect_arr = pca.inverse_transform(pca_val_defect_arr)

#     print("Nondefect train shape",train_nondefect_df.shape)
#     print("PCA nondefect train shape",pca_train_nondefect_arr.shape)
#     print("Inverse PCA nondefect train shape",recon_train_nondefect_arr.shape)

    # Compute reconstruction MSE
    train_nondefect_mse = ((train_nondefect_df - recon_train_nondefect_arr)**2).sum(axis=1)
    train_defect_mse = ((train_defect_df - recon_train_defect_arr)**2).sum(axis=1)
    val_nondefect_mse = ((val_nondefect_df - recon_val_nondefect_arr)**2).sum(axis=1)
    val_defect_mse = ((val_defect_df - recon_val_defect_arr)**2).sum(axis=1)

#     # Plot reconstruction errors of training images with and without defects
#     plt.figure(figsize=(6,6), dpi=100)
#     plt.scatter(val_nondefect_mse, val_defect_mse, alpha=0.2, label='val')
#     plt.scatter(train_nondefect_mse, train_defect_mse, alpha=0.2, marker='1', label='train')
#     plt.plot(range(int(np.max(val_defect_mse))), range(int(np.max(val_defect_mse))))
#     plt.xlabel('val_nondefect_reconstruction_mse')
#     plt.ylabel('val_defect_reconstruction_mse')
#     plt.legend()
#     plt.show()

    train_diff_mse = train_nondefect_mse - train_defect_mse
    train_num_correct = len(train_diff_mse[train_diff_mse < 0])
    train_num_samples = len(train_diff_mse)
    train_accuracy = train_num_correct/train_num_samples

    val_diff_mse = val_nondefect_mse - val_defect_mse
    val_num_correct = len(val_diff_mse[val_diff_mse < 0])
    val_num_samples = len(val_diff_mse)
    val_accuracy = val_num_correct/val_num_samples
    
    return train_accuracy, val_accuracy


def get_highlighted_array(arr, start_row, end_row, start_col, end_col):
    """ Highlight specified rectangular region of image by `factor` with an
        optional colored  boarder drawn around its edges and return the result.
    """
    array = np.copy(arr)

    array[start_row:end_row, start_col] = 0.8
    array[start_row:end_row, end_col] = 0.8
    array[start_row, start_col:end_col] = 0.8
    array[end_row, start_col:end_col+1] = 0.8

    return array

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    # result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def get_cnn_classifier_performance(nondefect_tain_tensor, defect_train_tensor,
                                   nondefect_test_tensor, defect_test_tensor,
                                   model):
    img_files = [f for f in listdir(directory) if f.endswith(".png")]
    pred_result = {} # store model prediction results
    for f in img_files:
        img = load_img(os.path.join(test_dir, f), color_mode='grayscale')
        img_array = img_to_array(img)
        img_tensor = tf.expand_dims(img_array, axis=0)

        pred_result[f] = {'pred_class': model.predict_classes(img_tensor).item(),
                          'pred_prob': model.predict(img_tensor).item(),
                          'was_correct': model.predict_classes(img_tensor).item() == truth_value,
                          'noise_level': noise_pct}

    performance_df = pd.DataFrame.from_dict(pred_result, orient='index')
    postfix = 'nondefect' if 'nondefect' in directory else 'defect'
    print('Accuracy ({}): '.format(postfix) + str(round(performance_df.was_correct.sum()/len(performance_df), 2)))

    return performance_df

def select_spacing_row_col(array):
    '''
    Select which row and column to use to determine the spacing. Ideally,
    we choose a row and col that has minimal noise so we can determine
    the spacing between the Ga-As bonds.
    '''
    #TODO: choose row with largest mean difference between darkest and brighest pixels
    row_index = np.argmax(np.mean(array, axis=1))
    col_index = np.argmax(np.mean(array, axis=0))
    
    return row_index, col_index

def find_peak_indices(values, show_plot=None):
    '''
    Takes a 1D array (vector) and returns the indices of the peaks and
    the values of the peaks.
    '''
    peaks, _ = find_peaks(values)
    peak_values = values[peaks]
    threshold = np.percentile(peak_values, 80)

    peaks = [peak for peak, value in zip(peaks, peak_values) if value>=threshold]
    peak_values = values[peaks]
    if show_plot:
        fig, ax = plt.subplots(figsize = (10,4))
        ax.plot(range(len(values)), values)
        ax.set_title(show_plot)
        ax.scatter(peaks, peak_values, c='r', marker='o')
        plt.show()

    return peaks, peak_values


def find_peak_distances(peak_indices):
    '''
    Takes vector of peak indices and returns a single distance.
    Currently returns the median distance.
    '''
    peak_distances = [j-i for i,j in zip(peak_indices, peak_indices[1:])]
    distance = np.median(peak_distances)
    
    return distance


def find_peak_spacing(array, row_values=None, col_values=None):
    '''
    Finds the pixel spacing between signficant peaks based on an array of 
    pixel values. The assumption is that the peaks in the image are where
    the atoms are located.
    
    Parameters:
    array - 2d numpy array that contains pixel values
    
    Returns:
    (row_spacing, col_spacing) - estimated distance between pixels 
    '''
    row, col = select_spacing_row_col(array)
    if row_values is None:
        row_values = array[row, :]
    if col_values is None:
        col_values = array[:, col]    
#     row_peak_indices, row_peak_values = find_peak_indices(row_values, show_plot='row_values')
#     col_peak_indices, col_peak_values = find_peak_indices(col_values, show_plot='col_values')
    row_peak_indices, row_peak_values = find_peak_indices(row_values, show_plot='row_values (col_spacing)')
    col_peak_indices, col_peak_values = find_peak_indices(col_values, show_plot='col_values (row_spacing)')
    row_spacing = find_peak_distances(col_peak_indices)
    col_spacing = find_peak_distances(row_peak_indices)
    
    return row_spacing, col_spacing


def get_atom_spacing(array):
    frame_width = 15
    
    num_cols = np.shape(array)[1]
    num_rows = np.shape(array)[0]

    # Create frames that have width 'frame_width' for each column
    frames = [array[:, i:i+frame_width] for i in range(num_cols-frame_width)]
    # Compare first frame to every other frame
    row_deltas = []
    first_frame = frames[2]
    for frame in frames[1:]:
        delta_frame = np.abs(first_frame - frame)
        total_delta = np.sum(delta_frame) # total_delta=0 if frame repeats
        row_deltas.append(total_delta)

    # Create frames that have width 'frame_width' for each column
    frames = [array[i:i+frame_width, :] for i in range(num_rows-frame_width)]
    # Compare first frame to every other frame
    col_deltas = []
    first_frame = frames[2]
    for frame in frames[1:]:
        delta_frame = np.abs(first_frame - frame)
        total_delta = np.sum(delta_frame) # total_delta=0 if frame repeats
        col_deltas.append(total_delta)

    row_deltas = -np.array(row_deltas)
    col_deltas = -np.array(col_deltas)
    row_spacing, col_spacing = find_peak_spacing(array, row_values=row_deltas, col_values=col_deltas)
#     print('Row spacing: ', row_spacing)
#     print('Column spacing: ', col_spacing)
    
    return row_spacing, col_spacing

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def get_trimmed_array(arr, trim_width = 1):
    '''
    Return an array where the a border is trimmed and removed from the array. 
    '''
    
    trimmed_array = arr[trim_width:-trim_width, trim_width:-trim_width]
    return trimmed_array

def find_n_largest_values(arr, n, trim=None):
    '''
    Returns the indices of the n largest values in an array
    '''
    
    N = n
    if trim:
        h_trim = trim[0]
        w_trim = trim[1]
        arr = arr[h_trim:-h_trim, w_trim:-w_trim]
    idx = np.argsort(arr.ravel())[-N:][::-1] #single slicing: `[:N-2:-1]`
    topN_val = arr.ravel()[idx]
    row_col = np.c_[np.unravel_index(idx, arr.shape)]

    return topN_val, row_col


def sliding_window_average(arr, model_type, 
                           window_height, window_width,
                           row_step, col_step,
                           pca_model = None, cnn_model = None,
                           display_progress = True,
                           lookup_dict = {},
                           upper_left = (), # defect region corners
                           lower_right = ()):
    '''
    Slides a window across a full-size TEM image and makes a computation for each window. The 
    output of the computation for each window is an output window of the same dimensions as 
    the input window. For each pixel in the original full-size TEM image, we return the 
    average output window value for all the sliding windows that included that pixel.
    
    If the lookup_dict is passed in as an argument, it is assumed that the computation for a 
    subset of the windows as been precalculated and can be lookup up rather than computed. This
    is useful in the case when you plan on sliding windows across many full-size images that 
    are mostly the same (i.e., only the location of the defect changes
    
    model_type is one of the following:
        'pca' - returns the pca reconstruction for each window
        'pca_mse' - returns a window with same MSE value across entire window
        'cnn' - returns the output of a cnn when window is passed into cnn
        'resid_cnn' - take resid between window and pca recon of window and
            pass it into a cnn (this option requires a lookup_dict for computational speed)
            
    pca_model must always be passed in for all model_type except cnn
    cnn_model is required for cnn and resid_cnn model type)
    '''

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    post_sums = np.zeros((arr.shape[0], arr.shape[1], 1))
    post_counts = np.zeros((arr.shape[0], arr.shape[1], 1))
    
    start = time.time()
    for row in range(0, n_rows - window_height + row_step, row_step):
        if display_progress:
            if row%100==0:
                minutes = round((time.time() - start)/60,2)
                print('Completed row {} of {} ({} minutes).'.format(row, n_rows, minutes))
        for col in range(0, n_cols - window_width + col_step, col_step):
            start_row = np.min((row, n_rows - window_height))
            start_col = np.min((col, n_cols - window_width))
            end_row = start_row + window_height
            end_col = start_col + window_width
            
            # Select a window
            window = arr[start_row:end_row, start_col:end_col]
            
            # Apply function to the window
            if lookup_dict: # check if a mse dictionary was passed through
                ul_row = upper_left[0] #upper left of defect region
                ul_col = upper_left[1]
                lr_row = lower_right[0]
                lr_col = lower_right[1]
                
                crit_box_start_row = ul_row - window_height
                crit_box_end_row = lr_row
                crit_box_start_col = ul_col - window_width
                crit_box_end_col = lr_col
                
                row_in_crit_box = True if crit_box_start_row < start_row < crit_box_end_row  else False
                col_in_crit_box = True if crit_box_start_col < start_col < crit_box_end_col else False
                
                if row_in_crit_box and col_in_crit_box:
                    post_window = get_pca_recon(pca_model, window, dims=(window_height, window_width))
                    if model_type == 'pca_mse':
                        mse = np.sum(((window - post_window)**2))
                        mse_or_prob = mse
                    elif model_type == 'cnn_resid':
                        resid_window = window - post_window
                        prob = cnn_model.predict(resid_window.reshape(1, window_height, window_width, 1)).item(0)
                        mse_or_prob = prob
                    elif model_type == 'vgg_resid':
                        resid_window = window - post_window
                        resid_window = resid_window.reshape((1, window.shape[0], window.shape[1], window.shape[2]))
                        vgg_scaled_window = vgg16_scaler(resid_window)
                        prob = cnn_model.predict(vgg_scaled_window)
                    else:
                        print('INVALID MODEL TYPE FOR SLIDING WINDOW')
                        return
                else:
#                     print('Using mse dict {}'.format((start_row, start_col)))
                    mse_or_prob = lookup_dict[(start_row, start_col)]
    
                # replicate the MSE value in an array of same size as window
                post_window = np.full((window_height, window_width, 1), mse_or_prob)
        
            elif (model_type == 'pca') or (model_type == 'pca_mse'):                    
                post_window = get_pca_recon(pca_model, window, dims=(window_height, window_width))
                
                if model_type == 'pca_mse':
                    mse = np.sum(((window - post_window)**2))
                    post_window = np.full((window_height, window_width, 1), mse)
            elif model_type == 'cnn':
                prob_defect = cnn_model.predict(window.reshape(1, window_height, window_width, 1))[0,0]
                post_window = np.full((window_height, window_width, 1), prob_defect)
            elif model_type == 'test':
                post_window = window + 1
            else:
                print('Invalid model type')
                return 
            
            post_sums[start_row:end_row, start_col:end_col] += post_window.reshape((window_height, window_width, 1))
            post_counts[start_row:end_row, start_col:end_col] += np.ones((window_height, window_width, 1))
    return_arr = post_sums/post_counts

    return return_arr, post_sums, post_counts

def get_sliding_window_dict(arr, pca, window_height, window_width, row_step, col_step,
                            model_type='pca', cnn_model=None):
    '''
    Take a full-size TEM array and apply a sliding window. For each window
    '''
    n_rows = arr.shape[0]
    n_cols = arr.shape[1]
    
    result_dict = {}
    index_list = []
    start = time.time()
    for row in range(0, n_rows - window_height + row_step, row_step):
        if row%100==0:
            print('Creating lookup dictionary...row {} of {} completed.'.format(row, n_rows))
        for col in range(0, n_cols - window_width + col_step, col_step):
            start_row = np.min((row, n_rows - window_height))
            start_col = np.min((col, n_cols - window_width))
            end_row = start_row + window_height
            end_col = start_col + window_width
                
            # Select a window
            window = arr[start_row:end_row, start_col:end_col]
            index_list.append((start_row, start_col))

            # Get PCA reconstruction of the window
            post_window = get_pca_recon(pca, window, dims=(window_height, window_width))
            
            if model_type == 'pca':
                # Compute MSE for window and store in dictionary
                mse = np.sum(((window - post_window)**2))
                result_dict[(start_row, start_col)] = mse                    
            elif model_type == 'cnn_resid':
                # Get residual between window and PCA recon of window
                resid_window = window - post_window
                prob = cnn_model.predict(resid_window.reshape((1, window.shape[0], window.shape[1], window.shape[2])))
                result_dict[(start_row, start_col)] = prob.item(0)
            elif model_type == 'vgg_resid':
                # Get residual between window and PCA recon of window
                resid_window = window - post_window
                resid_window = resid_window.reshape((1, window.shape[0], window.shape[1], window.shape[2]))
                vgg_scaled_window = vgg16_scaler(resid_window)
                prob = cnn_model.predict(vgg_scaled_window)
                result_dict[(start_row, start_col)] = prob.item(0)

    minutes = round((time.time() - start)/60,2)
    print('Time to finish: ', minutes)
    return result_dict


def find_n_largest_value_windows(arr, window_height, window_width, n, row_step, col_step, trim=0):
    '''
    Returns the indices of the n largest values in an array
    '''
    n_rows = arr.shape[0]
    n_cols = arr.shape[1]
    w = window_width
    h = window_height
    N = n
    
    window_means = []
    for row in range(0, n_rows - h + row_step, row_step):
        start_row = np.min((row, n_rows - window_height))
        end_row = start_row + window_height
        
        if (start_row < trim) or (end_row > (n_rows - trim)):
            continue
        
        for col in range(0, n_cols - w + col_step, col_step):
            start_col = np.min((col, n_cols - window_width))
            end_col = start_col + window_width
            
            if (start_col < trim) or (end_col > (n_cols - trim)):
                continue
            
            # Select a window
            window = arr[start_row:end_row, start_col:end_col]
            window_mean = np.mean(window)
            center_row = int(start_row + (window_height/2))
            center_col = int(start_col + (window_width/2))
            window_means.append((center_row, center_col, window_mean))
    
    sorted_windows = sorted(window_means, key=lambda tup: -tup[2])
    return sorted_windows[:n]

'''
**************************************************

**************************************************
'''

def compute_cnn_resid_accuracies(array_dict, pca, cnn,
                               row_spacing, col_spacing,
                               segment_height, segment_width,                           
                               trim_width,
                               row_step, col_step,
                               noise = 0.0,
                               n=10,
                               vgg16 = False,
                               output_dir = None):
    '''
    Randomly select a full-size image, add a round defect, then find the PCA residual
    and pass it into a CNN to classify the defect
    '''
    results = []
    
    count = 0
    total_iter = len(array_dict) * 4 * n
    col_shifts = [10*col_spacing + 3, 3*col_spacing + 1, 6*col_spacing + 2]
    start = time.time()
    for name, array in array_dict.items():
        # Add Gaussian noise to the image
        if noise > 0.0:
            array = add_gaussian_noise(array, wt=noise)
        
        # Store the pixel-by-pixel PCA based MSE for each full-size image for computational speed
        if vgg16:
            model_type = 'vgg_resid'
        else:
            model_type = 'cnn_resid'
        print('Creating CNN Probability dictionary for {}...'.format(name))
        prob_dict = get_sliding_window_dict(array, pca, segment_height, segment_width, row_step, col_step, 
                                            model_type=model_type, cnn_model=cnn)
        print('CNN probability dictionary stored.')
        
        # Create a full-size TEM image with each type of defect
        for defect_type in ['round', 'swap', 'enlarged', 'brightened']:
            for i in range(n):
                count += 1
                if count%10==0:
                    minutes = round((time.time() - start)/60,2)
                    print('Processing {} of {} iterations ({} minutes).'.format(count, total_iter, minutes))
                # add defect
                if defect_type == 'round':
                    defect_array, location, radius = add_defect_to_image(array, trim=np.max((segment_height, segment_width)))
                    
                    # store defect region for determining when to use dict vs recompute PCA
                    upper_left = (location[0] - radius, location[1] - radius)
                    lower_right = (location[0] + radius, location[1] + radius)
                elif defect_type == 'swap':
                    row_shift = random.randint(3,18)
                    col_shift = random.choice(col_shifts)

                    start_col = 23 + 3*col_spacing + col_shift - trim_width
                    end_col = 54 + 3*col_spacing + col_shift - trim_width
                    start_row = 9 + row_shift*row_spacing - trim_width
                    end_row = 30 + row_shift*row_spacing - trim_width
                    
                    # store defect region for determining when to use dict vs recompute PCA
                    upper_left = (start_row, start_col)
                    lower_right = (end_row, end_col)

                    area_to_swap = array[start_row:end_row, start_col:end_col]

                    # flip the pixel values for a GaAs bond so that atoms are "backwards"
                    defect_array = array.copy()
                    defect_array[start_row:end_row, start_col:end_col] = np.flip(area_to_swap, 1)
                    location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
                elif defect_type == 'enlarged':
                    row_shift = random.randint(3,18)
                    col_shift = random.choice(col_shifts)

                    start_col = 23 + 3*col_spacing + col_shift - trim_width
                    end_col = 39+ 3*col_spacing + col_shift - trim_width
                    start_row = 11 + row_shift*row_spacing - trim_width
                    end_row = 28 + row_shift*row_spacing - trim_width
                    
                    # store defect region for determining when to use dict vs recompute PCA
                    upper_left = (start_row, start_col)
                    lower_right = (end_row, end_col)
                    
                    area_to_zoom = array[start_row:end_row, start_col:end_col]
                    zoomed_region = cv2_clipped_zoom(area_to_zoom, 1.05)

                    defect_array = array.copy()
                    defect_array[start_row:end_row, start_col:end_col, 0] = zoomed_region

                    location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
                elif defect_type == 'brightened':
                    row_shift = random.randint(3,18)
                    col_shift = random.choice(col_shifts)

                    start_col = 29 + 3*col_spacing + col_shift - trim_width
                    end_col = 34 + 3*col_spacing + col_shift - trim_width
                    start_row = 17 + row_shift*row_spacing - trim_width
                    end_row = 22 + row_shift*row_spacing - trim_width
                    
                    # store defect region for determining when to use dict vs recompute PCA
                    upper_left = (start_row, start_col)
                    lower_right = (end_row, end_col)

                    area_to_brighten = array[start_row:end_row, start_col:end_col]
                    
                    location = (int(start_row + 2), int(start_col + 2))
                    
                    if area_to_brighten[2,2,0] < 0.5:
                        brighten_factor = 0.95 # if atom is dark, make it darker
                    else:
                        brighten_factor = 1.05

                    defect_array = array.copy()
                    defect_array[start_row:end_row, start_col:end_col] = area_to_brighten * brighten_factor

                # store true location of defect
                true_row = location[0]
                true_col = location[1]

                # generate pca heatmap
                segment_height = 84
                segment_width = 118
                heatmap_array, _, _ = sliding_window_average(defect_array, 
                                                             model_type = model_type,
                                                             window_height = segment_height,
                                                             window_width = segment_width,
                                                             row_step = row_step,
                                                             col_step = col_step,
                                                             pca_model=pca,
                                                             cnn_model=cnn,
                                                             display_progress = False,
                                                             lookup_dict = prob_dict,
                                                             upper_left = upper_left,
                                                             lower_right = lower_right)
                
                # get predicted row and col of the defect
                window_means = find_n_largest_value_windows(heatmap_array,
                                                            window_height = segment_height,
                                                            window_width = segment_width,
                                                            n=3,
                                                            row_step = row_step, 
                                                            col_step = col_step,
                                                            trim=np.max((segment_height, segment_width)))
                pred_row = window_means[0][0]
                pred_col = window_means[0][1]
                mean_window_mse = window_means[0][2]
                
                # check if predicted location is close to true location
                if (np.abs(pred_row-true_row) < (segment_height)) and (np.abs(pred_col-true_col) < (segment_width)):
                    is_correct = True
                else:
                    is_correct = False
                
                # display defect image and heatmap
                fig, axes = plt.subplots(1, 2, figsize=(10,5), dpi=200)
                ax = axes[0]
                ax.set_title('Defect Image for {}'.format(defect_type),fontsize=8, ha='center')
                ax.imshow(defect_array, cmap='gray')
                
                # diplay mse heatmap
                ax = axes[1]
                ax.set_title('Truth: ({},{}), Predicted=({},{}), {}'.format(true_row, true_col, pred_row, pred_col, is_correct),fontsize=8, ha='center')
                a = ax.imshow(heatmap_array, cmap='copper')
                plt.colorbar(a, ax=ax)
                
                if output_dir:
                    f_name = '{}_{}_iter{}'.format(name, defect_type, i)
                    plt.savefig(os.path.join(output_dir, f_name), bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

                results.append([name, noise, 'pca_cnn', is_correct, defect_type, true_row, true_col, pred_row, pred_col, round(mean_window_mse,2)])
            
    return results


# def add_random_defect_type(array, defect_type, 
#                            row_spacing, col_spacing,
#                            segment_height, segment_width,                           
#                            trim_width,n):
#     '''
#     1. Take a full-size TEM image as an input
#     2. Add a custom defect to the full-size TEM image
#     3. Create a random segment containing the defect
#     4. Return the '''
    
#     defect_type = 'swap'
    
#     for i in range(n):
#         count += 1
#         if count%10==0:
#             minutes = round((time.time() - start)/60,2)
#             print('Processing {} of {} iterations ({} minutes).'.format(count, total_iter, minutes))
#         # add defect
#         if defect_type == 'round':
#             defect_array, location, radius = add_defect_to_image(array, trim=np.max((segment_height, segment_width)))
#             upper_left = (location[0] - radius, location[1] - radius)
#             lower_right = (location[0] + radius, location[1] + radius)
#         elif defect_type == 'swap':
#             row_shift = random.randint(3,18)
#             col_shift = random.choice(col_shifts)

#             start_col = 23 + 3*col_spacing + col_shift - trim_width
#             end_col = 54 + 3*col_spacing + col_shift - trim_width
#             start_row = 9 + row_shift*row_spacing - trim_width
#             end_row = 30 + row_shift*row_spacing - trim_width

#             upper_left = (start_row, start_col)
#             lower_right = (end_row, end_col)

#             area_to_swap = array[start_row:end_row, start_col:end_col]

#             # flip the pixel values for a GaAs bond so that atoms are "backwards"
#             defect_array = array.copy()
#             defect_array[start_row:end_row, start_col:end_col] = np.flip(area_to_swap, 1)
#             location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
#         elif defect_type == 'enlarged':
#             row_shift = random.randint(3,18)
#             col_shift = random.choice(col_shifts)

#             start_col = 23 + 3*col_spacing + col_shift - trim_width
#             end_col = 39+ 3*col_spacing + col_shift - trim_width
#             start_row = 11 + row_shift*row_spacing - trim_width
#             end_row = 28 + row_shift*row_spacing - trim_width

#             upper_left = (start_row, start_col)
#             lower_right = (end_row, end_col)

#             area_to_zoom = array[start_row:end_row, start_col:end_col]
#             zoomed_region = cv2_clipped_zoom(area_to_zoom, 1.05)

#             defect_array = array.copy()
#             defect_array[start_row:end_row, start_col:end_col, 0] = zoomed_region

#             location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
#         elif defect_type == 'brightened':
#             row_shift = random.randint(3,18)
#             col_shift = random.choice(col_shifts)

#             start_col = 29 + 3*col_spacing + col_shift - trim_width
#             end_col = 34 + 3*col_spacing + col_shift - trim_width
#             start_row = 17 + row_shift*row_spacing - trim_width
#             end_row = 22 + row_shift*row_spacing - trim_width

#             upper_left = (start_row, start_col)
#             lower_right = (end_row, end_col)

#             area_to_brighten = array[start_row:end_row, start_col:end_col]

#             location = (int(start_row + 2), int(start_col + 2))

#             if area_to_brighten[2,2,0] < 0.5:
#                 brighten_factor = 0.95 # if atom is dark, make it darker
#             else:
#                 brighten_factor = 1.05

#             defect_array = array.copy()
#             defect_array[start_row:end_row, start_col:end_col] = area_to_brighten * brighten_factor


#             bottom_right_defect_location = (end_row, end_col)
    
#     return defect_array
    

def compute_pca_recon_accuracies(array_dict, pca,
                                 row_spacing, col_spacing,
                                 segment_height, segment_width,                           
                                 trim_width,
                                 row_step, col_step,
                                 noise = 0.0,
                                 n=10,
                                 output_dir = None,
                                 insert_defect=True):
    '''
    Randomly select a full-size image, add a round defect, then apply PCA reconstruction
    
    Inputs
        nondefect_full_arrays - dictionary {image_name: array}
    '''
    results = []
    
    count = 0
    total_iter = len(array_dict) * 4 * n
    col_shifts = [10*col_spacing + 3, 3*col_spacing + 1, 6*col_spacing + 2]
    start = time.time()
    for name, array in array_dict.items():
        # Add Gaussian noise to the image
        if noise > 0.0:
            array = add_gaussian_noise(array, wt=noise)

        # If an array with unknown defect location is passed in:
        if not(insert_defect):
            total_iter = len(array_dict)
            count += 1  
            if count%10==0:
                minutes = round((time.time() - start)/60,2)
                print('Processing {} of {} iterations ({} minutes).'.format(count, total_iter, minutes))

            defect_type = name
            defect_array = array
            display_progress = True if total_iter <= 5 else False
            heatmap_array, _, _ = sliding_window_average(defect_array, 
                                                         model_type='pca_mse', 
                                                         window_height = segment_height,
                                                         window_width = segment_width,
                                                         row_step = row_step,
                                                         col_step = col_step,
                                                         pca_model=pca,
                                                         display_progress = display_progress)
            # get predicted row and col of the defect
            window_means = find_n_largest_value_windows(heatmap_array,
                                                        window_height = segment_height,
                                                        window_width = segment_width,
                                                        n=3,
                                                        row_step = row_step, 
                                                        col_step = col_step,
                                                        trim=np.max((segment_height, segment_width)))
            true_row = 'NA'
            true_col = 'NA'
            pred_row = window_means[0][0]
            pred_col = window_means[0][1]
            mean_window_mse = window_means[0][2]
            is_correct = 'NA'
            
            # display defect image and heatmap
            fig, axes = plt.subplots(1, 2, figsize=(10,5), dpi=200)
            ax = axes[0]
            ax.set_title('Defect Image for {}'.format(defect_type),fontsize=8, ha='center')
            ax.imshow(defect_array, cmap='gray')

            # diplay mse heatmap
            ax = axes[1]
            ax.set_title('Truth: ({},{}), Predicted=({},{}), {}'.format(true_row, true_col, pred_row, pred_col, is_correct),fontsize=8, ha='center')
            a = ax.imshow(heatmap_array, cmap='copper')
            plt.colorbar(a, ax=ax)

            if output_dir:
                f_name = '{}_no_inserted defect'.format(name)
                plt.savefig(os.path.join(output_dir, f_name), bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            results.append([name, noise, 'pca', is_correct, defect_type, true_row, true_col, pred_row, pred_col, round(mean_window_mse,2)])
        else:   
            # Store the pixel-by-pixel PCA based MSE for each full-size image for computational speed
            print('Creating MSE dictionary for {}...'.format(name))
            mse_dict = get_sliding_window_dict(array, pca, segment_height, segment_width, row_step, col_step)
            print('MSE dictionary stored.')
            for defect_type in ['round', 'swap', 'enlarged', 'brightened']:  
                for i in range(n):
                    count += 1
                    if count%10==0:
                        minutes = round((time.time() - start)/60,2)
                        print('Processing {} of {} iterations ({} minutes).'.format(count, total_iter, minutes))
                    # add defect
                    if defect_type == 'round':
                        defect_array, location, radius = add_defect_to_image(array, trim=np.max((segment_height, segment_width)))
                        upper_left = (location[0] - radius, location[1] - radius)
                        lower_right = (location[0] + radius, location[1] + radius)
                    elif defect_type == 'swap':
                        row_shift = random.randint(3,18)
                        col_shift = random.choice(col_shifts)

                        start_col = 23 + 3*col_spacing + col_shift - trim_width
                        end_col = 54 + 3*col_spacing + col_shift - trim_width
                        start_row = 9 + row_shift*row_spacing - trim_width
                        end_row = 30 + row_shift*row_spacing - trim_width

                        upper_left = (start_row, start_col)
                        lower_right = (end_row, end_col)

                        area_to_swap = array[start_row:end_row, start_col:end_col]

                        # flip the pixel values for a GaAs bond so that atoms are "backwards"
                        defect_array = array.copy()
                        defect_array[start_row:end_row, start_col:end_col] = np.flip(area_to_swap, 1)
                        location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
                    elif defect_type == 'enlarged':
                        row_shift = random.randint(3,18)
                        col_shift = random.choice(col_shifts)

                        start_col = 23 + 3*col_spacing + col_shift - trim_width
                        end_col = 39+ 3*col_spacing + col_shift - trim_width
                        start_row = 11 + row_shift*row_spacing - trim_width
                        end_row = 28 + row_shift*row_spacing - trim_width

                        upper_left = (start_row, start_col)
                        lower_right = (end_row, end_col)

                        area_to_zoom = array[start_row:end_row, start_col:end_col]
                        zoomed_region = cv2_clipped_zoom(area_to_zoom, 1.05)

                        defect_array = array.copy()
                        defect_array[start_row:end_row, start_col:end_col, 0] = zoomed_region

                        location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
                    elif defect_type == 'brightened':
                        row_shift = random.randint(3,18)
                        col_shift = random.choice(col_shifts)

                        start_col = 29 + 3*col_spacing + col_shift - trim_width
                        end_col = 34 + 3*col_spacing + col_shift - trim_width
                        start_row = 17 + row_shift*row_spacing - trim_width
                        end_row = 22 + row_shift*row_spacing - trim_width

                        upper_left = (start_row, start_col)
                        lower_right = (end_row, end_col)

                        area_to_brighten = array[start_row:end_row, start_col:end_col]

                        location = (int(start_row + 2), int(start_col + 2))

                        if area_to_brighten[2,2,0] < 0.5:
                            brighten_factor = 0.95 # if atom is dark, make it darker
                        else:
                            brighten_factor = 1.05

                        defect_array = array.copy()
                        defect_array[start_row:end_row, start_col:end_col] = area_to_brighten * brighten_factor

                    # store true location of defect
                    true_row = location[0]
                    true_col = location[1]

                    # generate pca heatmap
                    heatmap_array, _, _ = sliding_window_average(defect_array, 
                                                                 model_type='pca_mse', 
                                                                 window_height = segment_height,
                                                                 window_width = segment_width,
                                                                 row_step = row_step,
                                                                 col_step = col_step,
                                                                 pca_model=pca,
                                                                 display_progress = False,
                                                                 lookup_dict = mse_dict,
                                                                 upper_left = upper_left,
                                                                 lower_right = lower_right)

                    # get predicted row and col of the defect
                    window_means = find_n_largest_value_windows(heatmap_array,
                                                                window_height = segment_height,
                                                                window_width = segment_width,
                                                                n=3,
                                                                row_step = row_step, 
                                                                col_step = col_step,
                                                                trim=np.max((segment_height, segment_width)))
                    pred_row = window_means[0][0]
                    pred_col = window_means[0][1]
                    mean_window_mse = window_means[0][2]

                    # check if predicted location is close to true location
                    if (np.abs(pred_row-true_row) < (segment_height)) and (np.abs(pred_col-true_col) < (segment_width)):
                        is_correct = True
                    else:
                        is_correct = False
                
                    # display defect image and heatmap
                    fig, axes = plt.subplots(1, 2, figsize=(10,5), dpi=200)
                    ax = axes[0]
                    ax.set_title('Defect Image for {}'.format(defect_type),fontsize=8, ha='center')
                    ax.imshow(defect_array, cmap='gray')

                    # diplay mse heatmap
                    ax = axes[1]
                    ax.set_title('Truth: ({},{}), Predicted=({},{}), {}'.format(true_row, true_col, pred_row, pred_col, is_correct),fontsize=8, ha='center')
                    a = ax.imshow(heatmap_array, cmap='copper')
                    plt.colorbar(a, ax=ax)

                    if output_dir:
                        f_name = '{}_{}_iter{}'.format(name, defect_type, i)
                        plt.savefig(os.path.join(output_dir, f_name), bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()

                    results.append([name, noise, 'pca', is_correct, defect_type, true_row, true_col, pred_row, pred_col, round(mean_window_mse,2)])
            
    return results

def add_custom_defect(array, defect_type, 
                      row_spacing, col_spacing,
                      segment_height, segment_width,
                      trim_width):
    '''
    Take a full-size TEM image and add a specific type of defect
    '''
    
    col_shifts = [10*col_spacing + 3, 3*col_spacing + 1, 6*col_spacing + 2]
    # add defect
    if defect_type == 'round':
        defect_array, location, radius = add_defect_to_image(array, trim=np.max((segment_height, segment_width)))
        
        upper_left = (location[0] - radius, location[1] - radius)
        lower_right = (location[0] + radius, location[1] + radius)
        
        start_row = upper_left[0]
        end_row = lower_right[0]
        start_col = upper_left[1]
        end_col = lower_right[1]
    elif defect_type == 'swap':
        row_shift = random.randint(3,18)
        col_shift = random.choice(col_shifts)

        start_col = 23 + 3*col_spacing + col_shift - trim_width
        end_col = 54 + 3*col_spacing + col_shift - trim_width
        start_row = 9 + row_shift*row_spacing - trim_width
        end_row = 30 + row_shift*row_spacing - trim_width

        upper_left = (start_row, start_col)
        lower_right = (end_row, end_col)

        area_to_swap = array[start_row:end_row, start_col:end_col]

        # flip the pixel values for a GaAs bond so that atoms are "backwards"
        defect_array = array.copy()
        defect_array[start_row:end_row, start_col:end_col] = np.flip(area_to_swap, 1)
        location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
    elif defect_type == 'enlarged':
        row_shift = random.randint(3,18)
        col_shift = random.choice(col_shifts)

        start_col = 23 + 3*col_spacing + col_shift - trim_width
        end_col = 39+ 3*col_spacing + col_shift - trim_width
        start_row = 11 + row_shift*row_spacing - trim_width
        end_row = 28 + row_shift*row_spacing - trim_width

        upper_left = (start_row, start_col)
        lower_right = (end_row, end_col)

        area_to_zoom = array[start_row:end_row, start_col:end_col]
        zoomed_region = cv2_clipped_zoom(area_to_zoom, 1.05)

        defect_array = array.copy()
        defect_array[start_row:end_row, start_col:end_col, 0] = zoomed_region

        location = (start_row + ((start_row-end_row)/2), start_col + ((end_col-start_col)/2))
    elif defect_type == 'brightened':
        row_shift = random.randint(3,18)
        col_shift = random.choice(col_shifts)

        start_col = 29 + 3*col_spacing + col_shift - trim_width
        end_col = 34 + 3*col_spacing + col_shift - trim_width
        start_row = 17 + row_shift*row_spacing - trim_width
        end_row = 22 + row_shift*row_spacing - trim_width

        upper_left = (start_row, start_col)
        lower_right = (end_row, end_col)

        area_to_brighten = array[start_row:end_row, start_col:end_col]

        location = (int(start_row + 2), int(start_col + 2))

        if area_to_brighten[2,2,0] < 0.5:
            brighten_factor = 0.95 # if atom is dark, make it darker
        else:
            brighten_factor = 1.05

        defect_array = array.copy()
        defect_array[start_row:end_row, start_col:end_col] = area_to_brighten * brighten_factor


        bottom_right_defect_location = (end_row, end_col)

    # store true location of defect
    true_row = location[0]
    true_col = location[1]
    
    
    return defect_array, start_row, end_row, start_col, end_col, true_row, true_col

#Just a test to see if the pull request works!