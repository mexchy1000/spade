#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:54:17 2020

@author: user
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
import pandas as pd
import argparse
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
K.set_image_data_format='channels_last'
from keras.applications import vgg16
from keras import backend as K
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#Image Show
def spatial_featuremap(t_features, img, pd_coord_tissue, imscale, radius = 10, posonly=True):
    tsimg = np.zeros(img.shape[:2])    
    tsimg_row = np.array(round(pd_coord_tissue.loc[:,'imgrow']*imscale), dtype=int)
    tsimg_col = np.array(round(pd_coord_tissue.loc[:,'imgcol']*imscale), dtype=int)
    for rr, cc,t in zip(tsimg_row, tsimg_col,t_features):
        r, c = draw.circle(rr, cc, radius = 10)
        if posonly:
            if t>0:
                tsimg[r,c]= t
        else:
            tsimg[r,c]=t
    return tsimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patchsize', type=int, default=32)
    parser.add_argument('--position', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--meta', type=str)
    parser.add_argument('--outdir', type=str, default='./SPADE_output/')
    parser.add_argument('--numpcs', type=float, default= 2)
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        
    #Param
    sz_patch = args.patchsize
    
    br_coord = pd.read_csv(args.position,
                           header=None, names= ['barcodes','tissue','row','col','imgrow','imgcol'])
    br_meta = pd.read_csv(args.meta)
    if 'seurat_clusters' not in br_meta.columns:
        print("Warning: meta data including seruat_clusters show t-SNE map of image features with clustering info")
    else:
        print('Meta data is loaded')
    br_meta_coord = pd.merge(br_meta, br_coord, how = 'inner', right_on ='barcodes' , left_on='Unnamed: 0')
   
    brimg = plt.imread(args.image)
    print('Input image dimension:', brimg.shape)
    
    brscale = args.scale
    br_coord_tissue = br_meta_coord.loc[br_meta_coord.tissue==1,:]
    
    #Image Patch
    tsimg_row = np.array(round(br_coord_tissue.loc[:,'imgrow']*brscale), dtype=int)
    tsimg_col = np.array(round(br_coord_tissue.loc[:,'imgcol']*brscale), dtype=int)
        
    tspatches = []
    sz = int(sz_patch/2)
    for rr, cc in zip(tsimg_row, tsimg_col):
        tspatches.append(brimg[rr-sz:rr+sz, cc-sz:cc+sz])
    tspatches = np.asarray(tspatches)
    print('Image to Patches done', '....patchsize is ', sz_patch, ' .... number of patches ' , tspatches.shape[0])
    
    #pretrained model
    pretrained_model = vgg16.VGG16(weights='imagenet', include_top = False, pooling='avg', input_shape = (32,32,3))
    X_in = tspatches.copy()
    X_in = vgg16.preprocess_input(X_in)
    pretrained_model.trainable = False
    print('Architecture of CNN model')
    pretrained_model.summary()
    
    if 'seurat_clusters' in br_meta.columns:
        Y = np.asarray(br_meta['seurat_clusters'])
    
    #feature extraction
    ts_features = pretrained_model.predict(X_in)
    print('Image features extracted.')
    ts_tsne = TSNE(n_components=2, init='pca',perplexity=30,random_state=10).fit_transform(ts_features)
    print('t-SNE for image features ... done')
    
    plt.figure(figsize=(8, 7))
    if 'seurat_clusters' in br_meta.columns:
        plt.scatter(ts_tsne[:, 0], ts_tsne[:, 1], c=Y, cmap='Dark2' , s = 5, alpha=0.5)
    else:
        plt.scatter(ts_tsne[:, 0], ts_tsne[:, 1], s = 5, alpha=0.5)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title('t-SNE for image features (VGG16 Output)')
    plt.savefig(args.outdir+'/SPADE_image_features.png', dpi=300)
    print('t-SNE map is saved')
    
    #PCA
    numpcs = args.numpcs
    pca = PCA(n_components=numpcs)
    pca.fit(ts_features)
    ts_pca = pca.transform(ts_features)
    print('PCs of image features are extracted')
    #        
    #plt.figure(figsize=(8, 7))
    #plt.scatter(ts_pca[:, 0], ts_pca[:, 1], c=Y, cmap='Dark2' , s = 5, alpha=0.5)
    #plt.colorbar()
    #plt.xlabel("z[0]")
    #plt.ylabel("z[1]")
    
    for ii in range(numpcs):
        tsimg = spatial_featuremap(ts_pca[:,ii], brimg, br_coord_tissue, brscale, posonly=False)
        plt.figure(figsize=(10,10))
        plt.imshow(brimg)
        plt.imshow(tsimg, alpha=0.7, cmap='bwr', vmin = -1.0, vmax=1.0)
        plt.savefig(args.outdir+'/SPADE_pc'+str(ii+1)+'.png', dpi=300)
    
    pd_ts_pca = pd.DataFrame(ts_pca, index = br_meta_coord.barcodes)
    pd_ts_pca.to_csv(args.outdir+'/ts_features_pc.csv')
    print('PCs of image features are saved')


