import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_point_clicker import clicker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from polarLV.common.funBullsEye import *
from polarLV.common.funObj import SegCollectionObj
from scipy import spatial









#######################################################
# 
# 
# 
# 
# 
# DEPRECATED FILE
# 
# 
# 
# 
# 
####################################################### 
















#######################################################
# Usefull function
####################################################### 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_colorbar(fig, ax, ims, side='right') : 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size='5%', pad=0.05)
    return fig.colorbar(ims, cax=cax, orientation='vertical')


#######################################################
# Latent space function
####################################################### 

def GetSigmaPoints(emb, n_comp, d1, d2):
    d_mean = np.array([np.mean(emb[:,i]) for i in range(n_comp)], dtype=float)
    # d_mean = np.array([0 for i in range(n_comp)], dtype=float)
    d_std  = np.array([np.std(emb[:,i]) for i in range(n_comp)], dtype=float)
    points = np.array([d_mean for _ in range (10)], dtype=float)
    coeff_sigma = np.array([-2,-1,0,1,2], dtype=float)

    points[:len(coeff_sigma),d1] += coeff_sigma*d_std[d1]
    points[len(coeff_sigma):,d2] += coeff_sigma*d_std[d2]

    return points

def ReconstructImg(img_inv, idx_notnan, shape_img) :
    tmp_img = np.full(shape_img[0]*shape_img[1], fill_value=np.nan)
    tmp_img[idx_notnan] = img_inv
    img = np.reshape(tmp_img, shape_img)

    return img

def ReconstructPat(pat_inv, L_max_idx, L_max_idx_notnan, shape_img):
    L_recons_pat = []
    for j in range(21) :
        dict_tmp = {}
        slices_pat = pat_inv[sum(L_max_idx[:j]):sum(L_max_idx[:j+1])]
        slice = ReconstructImg(slices_pat, L_max_idx_notnan[j], shape_img) 

        dict_tmp['MI'] = slice
        L_recons_pat.append(dict_tmp)

    return L_recons_pat

def ComputeRecons(recons_error, set, decoded, reverse=False, txt=""):
    L_mse = []
    L_mse = zip(*sorted( [(x,i) for (i,x) in enumerate(recons_error)], 
                reverse=reverse )[:5] )
    _, mse = L_mse
    mse = np.array(mse)
    print("\nReconstruction error : 5 "+txt+" recons (%) :\n", recons_error[mse])

    img_highest_dec = decoded[mse]
    img_highest_set = set[mse]

    return mse, img_highest_dec, img_highest_set
    

#######################################################
# Plot latent space
####################################################### 

def InteractiveLatentSpace(emb, points, d1, d2, cm):
    fig1, ax1 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax1.axis('off')
    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(emb[:,d1], emb[:,d2], c=cm)
    ax1.scatter(points[:,d1], points[:,d2], color='red', s=30, marker='o')
    ax1.set_xlabel('dim '+str(d1))
    ax1.set_ylabel('dim '+str(d2))
    klicker = clicker(ax1, ["Marker1", "Marker2", "Marker3"],
                            markers=["s", "^", "*"], 
                            colors=['magenta','magenta','magenta'])
    # plt.show()
    return klicker

def PlotReconsSigmaPoints_2D(points, inv_pts, idx_notnan, d1, d2, shape_img):
    est_img = np.full([shape_img[0], shape_img[1], len(points)], fill_value=0, dtype=np.float16)
    fig, ax = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax.axis('off')

    for i in range (inv_pts.shape[0]) :
        img = ReconstructImg(inv_pts[i,:], idx_notnan, shape_img)
        # Normalisation
        img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

        est_img[:,:,i] = img

        num = [1,3,5,7,9,10,8,6,4,2]
        ax = fig.add_subplot(5,2,num[i])
        im = ax.imshow(est_img[:,:,i], cmap='gray', vmin=0, vmax=1)
        plot_colorbar(fig, ax, im)
        ax.set_title(str(int(points[i][d1]))+" ; "+str(int(points[i][d2])))
    
    cbar_ax = fig.add_axes([0.5, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    # plt.show()

def PlotReconsSigmaPoints_3D(points, inv_pts, ref, L_max_idx, L_max_idx_notnan, d1, d2, shape_img) :
    center = (0,0)         # center of bulls eye
    angOrig = np.pi*2/3    # origine (LV-RV junction) angle
    resAng = 120           # angle resolution for polygons
    resRad = 21            # radial resolution for polygons

    fig, ax = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax.axis('off')

    for i in range (inv_pts.shape[0]) :
        print("Recons point",i+1,"/", inv_pts.shape[0])
        pat_inv = inv_pts[i,:]

        recons_pat = ReconstructPat(pat_inv, L_max_idx, L_max_idx_notnan, shape_img)

        # Seuillage
        for s in range (len(recons_pat)) :
            recons_pat[s]['MI'][recons_pat[s]['MI'] < 0] = 0
            recons_pat[s]['MI'][recons_pat[s]['MI'] > 1] = 1

        # # Normalisation
        # L_tmp = []
        # for d in recons_pat :
        #     min_val = np.nanmin(d['MI'])
        #     max_val = np.nanmax(d['MI'])
        #     L_tmp.append([min_val, max_val])

        # L_norm = [np.min(np.array(L_tmp)[:,0]), np.max(np.array(L_tmp)[:,1])]
        # for s in range (len(recons_pat)) :
        #     recons_pat[s]['MI'] = (recons_pat[s]['MI'] - L_norm[0]) / (L_norm[1] - L_norm[0])


        segAll = SegCollectionObj() # object of a collection of several cases
        segAll.add_case(str(i), recons_pat)  # add a case
        segAll.get_polygon_positions(ref.roi.coordinates, resAng=resAng) # get the positions of all polygons

        # for each case of the collection, get values of polygons, method: mean over pixels in polygon
        polygonVals = segAll.get_bullsEye_individual(['MI'], cases=[str(i)], mtd='mean') 

        # plot the bulls eye
        num = [1,3,5,7,9,10,8,6,4,2]
        ax = fig.add_subplot(5,2,num[i])
        ax.set_title(str(int(points[i][d1]))+" ; "+str(int(points[i][d2])))
        bullsEye(resAng = resAng, resRad = resRad, 
                radLimits = (ref.radEndoList[0], ref.radEpiList[-1]),
                center = center, angle_base = angOrig, 
                fill_values = np.array(polygonVals[str(i)]['MI']), # pass the polygon values
                v_min = 0, v_max = 1, cmap = "Reds", colorbar=0, fig = fig, ax = ax)

    # plt.show()

def PlotKlicker_2D(klicker, msrkMulti, emb, points, cm, d1, d2, idx_notnan, shape_img, save=None):
    fig2, ax2 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax2.axis('off')
    fig1, ax1 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax1.axis('off')

    cpt = 1
    n_region = len(klicker.get_positions())
    n_m1 = int(klicker.get_positions()['Marker1'].size /2)
    n_m2 = int(klicker.get_positions()['Marker2'].size /2)
    n_m3 = int(klicker.get_positions()['Marker3'].size /2)
    n_pts =  n_m1 + n_m2 + n_m3

    all_pts = []
    all_pts.append([p for _, pts in klicker.get_positions().items() \
                        for p in pts])
    all_pts = np.squeeze(all_pts)

    for name, pts in klicker.get_positions().items() :
        if pts.size == 0 :
            continue

        inv_pts = msrkMulti.predict(pts)
        est_img = np.full([shape_img[0], shape_img[1], len(pts)], fill_value=0, dtype=np.float16)
        for i in range (inv_pts.shape[0]) :
            img = ReconstructImg(inv_pts[i,:], idx_notnan, shape_img)
            # Normalisation
            img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

            est_img[:,:,i] = img

            ax2 = fig2.add_subplot(n_region,int(np.ceil(n_pts/n_region)),cpt)
            im = ax2.imshow(est_img[:,:,i], cmap='gray')#, vmin=0, vmax=1)
            plot_colorbar(fig2, ax2, im)
            ax2.set_title(name+' : '+str(int(pts[i][d1]))+" ; "+str(int(pts[i][d2])))
            cpt+=1


    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(emb[:,d1], emb[:,d2], c=cm)
    ax1.scatter(points[:,d1], points[:,d2], color='red', s=30, marker='o')
    ax1.scatter(all_pts[:n_m1,d1], all_pts[:n_m1,d2], color='magenta', s=30, marker='s')
    ax1.scatter(all_pts[n_m1:n_m1+n_m2,d1], 
                all_pts[n_m1:n_m1+n_m2,d2], color='magenta', s=30, marker='^')
    ax1.scatter(all_pts[n_m1+n_m2:,d1], 
                all_pts[n_m1+n_m2:,d2], color='magenta', s=30, marker='*')
    ax1.legend(['Latent space', 'sigma', 'marker1', 'marker2', 'marker3'])

    if save != None : fig1.savefig(os.path.join(save, "LS_ReconsPoints.png"))
    if save != None : fig2.savefig(os.path.join(save, "ReconsInteractivePoints.png"))

    # plt.show()

def PlotKlicker_3D(klicker, msrkMulti, emb, points, ref, cm, d1, d2, L_max_idx, L_max_idx_notnan, shape_img, save) :
    fig2, ax2 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax2.axis('off')
    fig1, ax1 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax1.axis('off')

    cpt = 1
    n_region = len(klicker.get_positions())
    n_m1 = int(klicker.get_positions()['Marker1'].size /2)
    n_m2 = int(klicker.get_positions()['Marker2'].size /2)
    n_m3 = int(klicker.get_positions()['Marker3'].size /2)
    n_pts =  n_m1 + n_m2 + n_m3

    all_pts = []
    all_pts.append([p for _, pts in klicker.get_positions().items() \
                        for p in pts])
    all_pts = np.squeeze(all_pts)

    center = (0,0)         # center of bulls eye
    angOrig = np.pi*2/3    # origine (LV-RV junction) angle
    resAng = 120           # angle resolution for polygons
    resRad = 21            # radial resolution for polygons


    for name, pts in klicker.get_positions().items() :
        if pts.size == 0 :
            continue

        inv_pts = msrkMulti.predict(pts)
        for i in range (inv_pts.shape[0]) :
            ax2 = fig2.add_subplot(n_region,int(np.ceil(n_pts/n_region)),cpt)
            ax2.set_title(name+' : '+str(int(pts[i][d1]))+" ; "+str(int(pts[i][d2])))
            cpt+=1

            print("Recons point",i+1,"/", inv_pts.shape[0])
            pat_inv = inv_pts[i,:]

            recons_pat = ReconstructPat(pat_inv, L_max_idx, L_max_idx_notnan, shape_img)

            # Seuillage
            for s in range (len(recons_pat)) :
                recons_pat[s]['MI'][recons_pat[s]['MI'] < 0] = 0
                recons_pat[s]['MI'][recons_pat[s]['MI'] > 1] = 1

            # # Normalisation
            # L_tmp = []
            # for d in recons_pat :
            #     min_val = np.nanmin(d['MI'])
            #     max_val = np.nanmax(d['MI'])
            #     L_tmp.append([min_val, max_val])

            # L_norm = [np.min(np.array(L_tmp)[:,0]), np.max(np.array(L_tmp)[:,1])]
            # for s in range (len(recons_pat)) :
            #     recons_pat[s]['MI'] = (recons_pat[s]['MI'] - L_norm[0]) / (L_norm[1] - L_norm[0])


            segAll = SegCollectionObj() # object of a collection of several cases
            segAll.add_case(str(i), recons_pat)  # add a case
            segAll.get_polygon_positions(ref.roi.coordinates, resAng=resAng) # get the positions of all polygons

            # for each case of the collection, get values of polygons, method: mean over pixels in polygon
            polygonVals = segAll.get_bullsEye_individual(['MI'], cases=[str(i)], mtd='mean') 

            # plot the bulls eye
            bullsEye(resAng = resAng, resRad = resRad, 
                    radLimits = (ref.radEndoList[0], ref.radEpiList[-1]),
                    center = center, angle_base = angOrig, 
                    fill_values = np.array(polygonVals[str(i)]['MI']), # pass the polygon values
                    v_min = 0, v_max = 1, cmap = "Reds", colorbar=0, fig = fig2, ax = ax2)


    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(emb[:,d1], emb[:,d2], c=cm)
    ax1.scatter(points[:,d1], points[:,d2], color='red', s=30, marker='o')
    ax1.scatter(all_pts[:n_m1,d1], all_pts[:n_m1,d2], color='magenta', s=30, marker='s')
    ax1.scatter(all_pts[n_m1:n_m1+n_m2,d1], 
                all_pts[n_m1:n_m1+n_m2,d2], color='magenta', s=30, marker='^')
    ax1.scatter(all_pts[n_m1+n_m2:,d1], 
                all_pts[n_m1+n_m2:,d2], color='magenta', s=30, marker='*')
    ax1.legend(['Latent space', 'sigma', 'marker1', 'marker2', 'marker3'])

    if save != None : fig1.savefig(os.path.join(save, "LS_ReconsPoints.png"))
    if save != None : fig2.savefig(os.path.join(save, "ReconsInteractivePoints.png"))

    # plt.show()

def PlotKlicker_VAE(klicker, vae, emb, points, cm, n_comp, d1, d2, save=None):
    fig2, ax2 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax2.axis('off')
    fig1, ax1 = plt.subplots(figsize=(12,10), constrained_layout=True)
    ax1.axis('off')

    cpt = 1
    n_region = len(klicker.get_positions())
    n_m1 = int(klicker.get_positions()['Marker1'].size /2)
    n_m2 = int(klicker.get_positions()['Marker2'].size /2)
    n_m3 = int(klicker.get_positions()['Marker3'].size /2)
    n_pts =  n_m1 + n_m2 + n_m3

    all_pts = []
    all_pts.append([p for _, pts in klicker.get_positions().items() \
                        for p in pts])
    all_pts = np.squeeze(all_pts)

    for name, pts in klicker.get_positions().items() :
        if pts.size == 0 :
            continue

        mean_pts = np.array([np.mean(emb[:,i]) for i in range(n_comp)], dtype=float)
        # mean_pts = np.array([0 for i in range(n_comp)], dtype=float)
        points_emb = np.array([mean_pts for _ in range (len(pts))], dtype=float)
        points_emb[:,d1] = pts[:,0]
        points_emb[:,d2] = pts[:,1]

        inv_img = vae.decoder.predict(points_emb)
        for i in range (inv_img.shape[0]) :
            # Normalisation
            # img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))
            
            # inv_img[i] *= 2
            ax2 = fig2.add_subplot(n_region,int(np.ceil(n_pts/n_region)),cpt)
            im = ax2.imshow(inv_img[i], cmap='gray', vmin=0, vmax=1)
            plot_colorbar(fig2, ax2, im)
            ax2.set_title(name+' : '+str(int(pts[i][0]))+" ; "+str(int(pts[i][1])))
            cpt+=1
        
    if save :
        fig2.savefig(save+'['+str(d1)+'-'+str(d2)+']_recons_points.png')

    ax1 = fig1.add_subplot(1,1,1)
    ax1.scatter(emb[:,d1], emb[:,d2], c=cm)
    ax1.scatter(points[:,d1], points[:,d2], color='red', s=30, marker='o')
    ax1.scatter(all_pts[:n_m1,0], all_pts[:n_m1,1], color='magenta', s=30, marker='s')
    ax1.scatter(all_pts[n_m1:n_m1+n_m2,0], 
                all_pts[n_m1:n_m1+n_m2,1], color='magenta', s=30, marker='^')
    ax1.scatter(all_pts[n_m1+n_m2:,0], 
                all_pts[n_m1+n_m2:,1], color='magenta', s=30, marker='*')
    ax1.legend(['Latent space', 'sigma', 'marker1', 'marker2', 'marker3'])
    ax1.set_xlabel('dim '+str(d1))
    ax1.set_ylabel('dim '+str(d2))

    if save :
        fig1.savefig(save+'['+str(d1)+'-'+str(d2)+']_embedding.png')

    # plt.show()

def PlotRecons (img_dec, img_set, closest_pts, txt=""):
    fig, ax = plt.subplots(figsize=(12,10),constrained_layout=True)
    ax.axis('off')
    for i in range (len(img_dec)) :
        ax = fig.add_subplot(5, 5, i+1) 
        ims = ax.imshow(img_dec[i, :, :], cmap='gray', vmin=0, vmax=1)
        plot_colorbar(fig, ax, ims)

        for j in range (3) :
            ax = fig.add_subplot(5, 5, i+(j+1)*5+1) 
            ims = ax.imshow(closest_pts[i, j, :, :], cmap='gray', vmin=0, vmax=1)
            plot_colorbar(fig, ax, ims)

        ax = fig.add_subplot(5, 5, i+20+1) 
        ims = ax.imshow(img_set[i, :, :], cmap='gray', vmin=0, vmax=1)
        plot_colorbar(fig, ax, ims)

    fig.suptitle(txt+' reconstruction\nFirst row : Reconstructed images\nNext 3 rows : 3 closest points in latent space\nLast row : Groundtruth')







