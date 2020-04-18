# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:23:42 2020

@author: Chia Shin Ho
"""
import sys
import cv2
import numpy as np
import numpy.ma as ma


cluster_n_fg = 2
cluster_n_bg = 3

if __name__ == '__main__':
    
    # read src image and trimap into pixel 
    file_name = 'woman'
    img = cv2.imread('./img/'+ file_name + '.png', cv2.IMREAD_COLOR)
    img = img/255
    trimap = cv2.imread('./trimap/'+ file_name + '.png', cv2.IMREAD_GRAYSCALE)

    length = np.size(trimap,0)
    width = np.size(trimap,1)

    # prepare masks
    fg_mask = trimap == 255
    bg_mask = trimap == 0
    unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
    
    # fill in known fg, bg, and alpha
    alpha = np.zeros(trimap.shape)
    alpha[fg_mask] = 1
    alpha[unknown_mask] = np.nan

    #add more axis to fg_mask to fit the 3-D(RGB) image color pixel
    fg = img*np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2)
    bg = img*np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)

    
    #########################################
    # TODO: prepare data points and apply GMM
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
    unknown_dilated = cv2.dilate(np.float32(unknown_mask), kernel)

    fg_mask_reduced = np.logical_and(fg_mask, unknown_dilated)
    bg_mask_reduced = np.logical_and(bg_mask, unknown_dilated)

    fg_reduced = img*np.repeat(fg_mask_reduced[:, :, np.newaxis], 3, axis=2)
    bg_reduced = img*np.repeat(bg_mask_reduced[:, :, np.newaxis], 3, axis=2)

    fg_2d = fg_reduced.reshape((width*length, 3))
    bg_2d = bg_reduced.reshape((width*length, 3))
    
    fg_train = []
    bg_train = []
    for i in range (width*length):
        if fg_2d[i].any() == True:
            fg_train.append(fg_2d[i])

        if bg_2d[i].any() == True:
            bg_train.append(bg_2d[i])

    fg_train = np.array(fg_train)
    bg_train = np.array(bg_train)
            

    """    
    cv2.imshow("new pic====", np.float32(fg_reduced))
    cv2.imshow("new pic", np.float32(bg_reduced))

    cv2.waitKey(0)

    cv2.waitKey(0)
    ll"""

    
    
    #train foreground to GMM's group
    em_fg = cv2.ml.EM_create()
    em_fg.setClustersNumber(cluster_n_fg)
    em_fg.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
    em_fg.trainEM(fg_train)
    
    means_fg = em_fg.getMeans()
    covs_fg = em_fg.getCovs()
    print(means_fg,'\n', covs_fg)


    #train background to GMM's group
    em_bg = cv2.ml.EM_create()
    em_bg.setClustersNumber(cluster_n_bg)
    em_bg.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_DIAGONAL)
    em_bg.trainEM(bg_train)


    means_bg = em_bg.getMeans()
    covs_bg = em_bg.getCovs()
    print(means_bg,'\n', covs_bg)
    

    """cv2.imshow("unknown", np.float32(unknown_mask))
    cv2.imshow("dilated", dilated)
    cv2.waitKey(0)"""
    
    
    #########################################
    # get coordinates of unknown pixels
    unknown = np.argwhere(np.isnan(alpha))
    
    # for showing how many pixels were computed
    PASS = 0
    total = len(unknown)
    
    while len(unknown) > 0:
        y,x = unknown[0]
        
        ##########################################################################
        # TODO: try to set different initial of alpha and optimize F, B, and alpha

        likelihood_max = -1000000000

        #chose combination on different fg and bg clusters
        for i in range(cluster_n_fg):
            for j in range (cluster_n_bg):

                mean_fg = np.array([means_fg[i]]).transpose()
                cov_fg = covs_fg[i]
                cov_fg_inv = np.linalg.inv(cov_fg)
                mean_bg = np.array([means_bg[j]]).transpose()
                cov_bg = covs_bg[j]
                cov_bg_inv = np.linalg.inv(cov_bg)
                alpha_ = 0.5
                sigma_c = 0.01
                I = np.identity(3)
                C = np.array([img[y][x]]).transpose()

                for k in range (200):

                    #fix alpha
                    a1 = cov_fg_inv + I*pow(alpha_,2)/pow(sigma_c,2)
                    a2 = I*alpha_*(1-alpha_)/pow(sigma_c,2)
                    a3 = I*alpha_*(1-alpha_)/pow(sigma_c,2)
                    a4 = cov_bg_inv + I*pow((1-alpha_),2)/pow(sigma_c,2)
                    A = np.array([[a1[0][0], a1[0][1], a1[0][2], a2[0][0], a2[0][1], a2[0][2]],
                                  [a1[1][0], a1[1][1], a1[1][2], a2[1][0], a2[1][1], a2[1][2]],
                                  [a1[2][0], a1[2][1], a1[2][2], a2[2][0], a2[2][1], a2[2][2]],
                                  [a3[0][0], a3[0][1], a3[0][2], a4[0][0], a4[0][1], a4[0][2]],
                                  [a3[1][0], a3[1][1], a3[1][2], a4[1][0], a4[1][1], a4[1][2]],
                                  [a3[2][0], a3[2][1], a3[2][2], a4[2][0], a4[2][1], a4[2][2]]])

                    b1 = np.matmul(cov_fg_inv, mean_fg) + C*alpha_/pow(sigma_c,2)
                    b2 = np.matmul(cov_bg_inv, mean_bg) + C*(1-alpha_)/pow(sigma_c,2)
                    b = np.array([[b1[0][0]], [b1[1][0]], [b1[2][0]], [b2[0][0]], [b2[1][0]], [b2[2][0]]])

                    X = np.linalg.solve(A,b)

                    #fix F, B

                    F = X[:3]           #(3,1)
                    B = X[3:6]          #(3,1)

                
                    dis = np.dot(np.subtract(F, B).transpose(), np.subtract(F,B))
                    alpha_s = np.matmul(np.subtract(C, B).transpose(), np.subtract(F, B)) / dis[0][0]
                    alpha_ = alpha_s[0][0]
                    
                #check the likelihood of each pair of F and B
                d1 = (pow(C[0][0]-alpha_*F[0][0]-(1-alpha_)*B[0][0],2) + pow(C[1]-alpha_*X[1]-(1-alpha_)*X[4],2) + pow(C[2][0]-alpha_*F[2][0]-(1-alpha_)*B[2][0],2))/(2*pow(sigma_c,2))
                d2 = np.matmul(np.matmul(np.subtract(F,mean_fg).transpose(), cov_fg_inv), np.subtract(F, mean_fg)) / 2
                d3 = np.matmul(np.matmul(np.subtract(B,mean_bg).transpose(), cov_bg_inv), np.subtract(B, mean_bg)) / 2
                likelihood = -(d1) - d2[0][0] - d3[0][0]


                if(likelihood > likelihood_max):
                    likelihood_max = likelihood
                    index_max = [i,j]
                    alpha_op = alpha_

        alpha[y][x] =  alpha_op


        ##########################################################################
        
        unknown = np.delete(unknown,0,0)      
        PASS += 1
        sys.stdout.write("\rprogress:\t{}/{}".format(PASS,total))
        sys.stdout.flush()
    
    target_scene = cv2.imread('landscape.png',cv2.IMREAD_COLOR)

    target_scene = cv2.resize(target_scene, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_CUBIC)/255
    
    ####################################################
    # TODO: attach the result foreground to target scene
    

    new = np.zeros((length, width, 3))

    for i in range (length):
        for j in range(width):
            value_list = alpha[i][j]*img[i][j] + (1-alpha[i][j])*target_scene[i][j]
            new[i][j] = value_list

    cv2.imwrite('./result/'+file_name+'.png',new)
    cv2.imshow("new pic", np.float32(new))

    cv2.waitKey(0)
    ####################################################