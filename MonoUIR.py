from __future__ import absolute_import, division, print_function
import collections
import sys
import argparse
import os
import datetime
from telnetlib import GA
import numpy as np
import scipy as sp
import math
import PIL.Image as pil
from PIL import Image
from rawpy import imread
import matplotlib
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, disk, square


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from tensorflow.python.keras.models import load_model
from DenseDepth.layers import BilinearUpSampling2D
from DenseDepth.utils import predict
from Evaluation import getScore

matplotlib.use('TkAgg')

def predictDepthMap(args, img):

    # Loading depth model
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    model = load_model(args.depthModel, custom_objects=custom_objects, compile=False)
    print('\nModel loaded ({0}).'.format(args.depthModel))

    # Predict depth map
    depth = predict(model, img)
    depth = np.squeeze(depth)
    depth = Image.fromarray(depth).resize((640, 480), pil.BICUBIC)
    depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth))).astype(np.float32)
    depth = (args.multi4depth * (1.0 - depth)) + args.addi4depth

    return depth

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r = []
    points_g = []
    points_b = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)

def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
         val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
         return val
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
         val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
         return val
    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                # p0=np.random.random(2) * bounds_upper,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)

    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
        BD = (slope * depths) + intercept
        return BD, np.array([slope, intercept])
    print("best loss:{0}\n".format(best_loss))
    return estimate(depths, *coefs), coefs

def construct_neighborhood_map(depths, epsilon=0.05):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                if 0 <= x < depths.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= x < depths.shape[0]:
                    x2, y2 = x - 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 0 <= y < depths.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= y < depths.shape[1]:
                    x2, y2 = x, y - 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0 #reset largest background to 0
    return nmap, n_neighborhoods - 1

def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            if 0 <= x < nmap.shape[0] - 1:
                x2, y2 = x + 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= x < nmap.shape[0]:
                x2, y2 = x - 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 0 <= y < nmap.shape[1] - 1:
                x2, y2 = x, y + 1
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= y < nmap.shape[1]:
                x2, y2 = x, y - 1
                if not mask[x2, y2]:
                    q.append((x2, y2))

def refine_neighborhood_map(nmap, min_size = 10, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
    refined_nmap = closing(refined_nmap, square(radius))
    return refined_nmap, num_labels - 1

'''
Estimate illumination map from local color space averaging
'''
def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    D = img - B
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))

'''
Estimate values for beta_D
'''
def estimate_wideband_attentuation(depths, illum, radius = 6, max_val = 10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations, []

'''
Calculate the values of beta_D for an image from the depths, illuminations, and constants
'''
def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

'''
Filter the data such that only one point is selected per "bin", defined using a radius.
The median value is selected per bin.
This prevents the regression from being overwhelmed due to the
large amount of junk data at certain points in the range.
'''
def filter_data(X, Y, radius_fraction=0.01):
    idxs = np.argsort(X)
    X_s = X[idxs]
    Y_s = Y[idxs]
    x_max, x_min = np.max(X), np.min(X)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    tempX = []
    tempY = []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
            idxs = np.argsort(tempY)
            med_idx = len(idxs) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
    return np.array(dX), np.array(dY)

def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
    def calculate_reconstructed_depths(depths, illum, a, b, c, d):
        eps = 1E-5
        res = -np.log(illum + eps) / (calculate_beta_D(depths, a, b, c, d) + eps)
        return res
    def loss(a, b, c, d):
        return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))
    dX, dY = filter_data(depths[locs], estimation[locs], radius_fraction)
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=calculate_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
            L = loss(*optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs], estimation[locs])
        BD = (slope * depths + intercept)
        return l * BD, np.array([slope, intercept])
    print(f'Found best loss {best_loss}', flush=True)
    BD = l * calculate_beta_D(depths, *coefs)              
    return BD, coefs

def recoverAndPostprocess(img, depths, B, beta_D, nmap):
    recovered = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    recovered = np.maximum(0.0, np.minimum(1.0, recovered))
    recovered[nmap == 0] = 0

    # Gray-world white balance
    R, G, B = recovered[:,:,0], recovered[:,:,1], recovered[:,:,2]
    RAver, GAver, BAver = np.mean(R[R!=0]), np.mean(G[G!=0]), np.mean(B[B!=0])
    K = (RAver+GAver+BAver)/3
    recovered[:,:,0], recovered[:,:,1], recovered[:,:,2] = K*R/RAver, K*G/GAver, K*B/BAver
    recovered = (recovered - np.min(recovered)) / (np.max(recovered) - np.min(recovered))
    recovered[nmap == 0] = img[nmap == 0]

    # Denoise
    sigma_est = estimate_sigma(recovered, multichannel=True, average_sigmas=True) / 10.0
    recovered = denoise_tv_chambolle(recovered, sigma_est, multichannel=True)

    recovered = np.round(recovered * 255.0).astype(np.uint8)
    return recovered

def pipeline(args, img, depth):
    print('Estimating backscatter...', flush=True)
    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depth, fraction=0.01, min_depth_percent=args.min_depth)
    
    print('Finding backscatter coefficients...', flush=True)
    Br, coefsR = find_backscatter_values(ptsR, depth, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depth, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depth, restarts=25)

    print('Constructing neighborhood map...', flush=True)
    nmap, _ = construct_neighborhood_map(depth, 0.1)
    
    print('Refining neighborhood map...', flush=True)
    nmap, n = refine_neighborhood_map(nmap, 50)

    print('Estimating illumination...', flush=True)
    illR = estimate_illumination(img[:, :, 0], Br, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    illG = estimate_illumination(img[:, :, 1], Bg, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    illB = estimate_illumination(img[:, :, 2], Bb, nmap, n, p=args.p, max_iters=100, tol=1E-5, f=args.f)
    
    print('Estimating wideband attenuation...', flush=True)
    beta_D_r, _ = estimate_wideband_attentuation(depth, illR)
    refined_beta_D_r, coefsR = refine_wideband_attentuation(depth, illR, beta_D_r, min_depth_fraction=args.min_depth, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_g, _ = estimate_wideband_attentuation(depth, illG)
    refined_beta_D_g, coefsG = refine_wideband_attentuation(depth, illG, beta_D_g, min_depth_fraction=args.min_depth, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_b, _ = estimate_wideband_attentuation(depth, illB)
    refined_beta_D_b, coefsB = refine_wideband_attentuation(depth, illB, beta_D_b, min_depth_fraction=args.min_depth, radius_fraction=args.spread_data_fraction, l=args.l)
    
    print('Reconstructing image...', flush=True)
    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
    recovered = recoverAndPostprocess(img, depth, B, beta_D, nmap)
    return recovered

def run(args):

    # Loading image
    if args.raw:
        img = Image.fromarray(imread(args.inputImage).postprocess())
    else:
        img = pil.open(args.inputImage).convert('RGB')
    img = img.resize((640,480), pil.BICUBIC)

    # Image quality assessing before restoration
    score = open(args.scoreLog, mode = 'a',encoding='utf-8')
    print("UICM-UISM-UIConM-UIQM-UCIQE", file=score)
    
    UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(img))
    print("{0}\t{1}".format(args.inputImage, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),file=score)
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)

    # Predicting depth map
    img = np.clip(np.asfarray(img) / 255, 0, 1)
    depth = predictDepthMap(args, img)

    try:
        ret = pipeline(args, np.array(img), depth)
    except ValueError:
        print("Fail to recover " + args.inputImage + "!")
    else:
        im = Image.fromarray(ret, mode='RGB')
        UICM, UISM, UIConM, UIQM, UCIQE = getScore(np.array(im))
        print("{0}\t{1}\t{2}\t{3}\t{4}".format(UICM, UISM, UIConM, UIQM, UCIQE), file=score)
        im.save(args.outputImage, format='png')
        print('Done.')

    score.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImage', default='Input/2.png')
    parser.add_argument('--outputImage', default='Output/2.png')
    parser.add_argument("--scoreLog", default='Logs/score.log')

    parser.add_argument('--depthModel', type=str, default="models/kitti.h5")
    parser.add_argument('--multi4depth', default=3.0)
    parser.add_argument('--addi4depth', default=1.0)

    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--raw', action='store_true', help='RAW image')
    args = parser.parse_args()
    run(args)
