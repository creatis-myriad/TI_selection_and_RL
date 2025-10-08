import argparse
import csv
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.append('../') 

from datetime import datetime
from keras.callbacks import EarlyStopping, TensorBoard
from nn_libr.callbacks import BestModelCallback, ImagesCallback
from nn_libr.common.functions import ComputeInfarctExtent
from nn_libr.model import decoder, encoder, vae
from nn_libr.paths.loading_data import Load_dataset
from nn_libr.preprocess import normalization, unwrap
from nn_libr.paths.IO_paths import GetPaths
from nn_libr.plots import plot_loss
from scipy import spatial

seed = 1
np.random.seed(seed)



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



def main(args):
    save_dir = os.path.join(args.outputDir, str(datetime.now())[:18])
    if not os.path.isdir(save_dir) :
        os.mkdir(save_dir)

    with open(args.ref, 'rb') as file:
        ref = pickle.load(file)


    L_pat_name = os.listdir(args.inputDir)
    coeff_split_data = args.split_data
    idx_split_pat = int(len(L_pat_name)*coeff_split_data)+1
    conv_x = args.convX
    coordinates = ref.roi.coordinates[:,:,:,0]
    radial, theta = coordinates[:,:,0], coordinates[:,:,1]
    coords_data = np.where(~np.isnan(radial))
    shape=(512-2*(conv_x-1), 16)


    paths = GetPaths(args.inputDir)
    if args.dcm :
        L_paths = paths['dcm.pkl']
        input_seg = False
    if args.roi :
        L_paths = paths['roi.pkl']
        input_seg = True

    input_type = "MAG"
    L_dataset, L_name_pat, L_idx_split = Load_dataset(
        L_paths, 
        L_pat_name, 
        idx_split_pat, 
        input_type=input_type,
        input_seg=input_seg,
    ) 
    idx_split_data = L_idx_split[0]-1

    # Normalization
    min_max_ = [ 
        np.nanmin(L_dataset),
        np.nanmax(L_dataset)
    ]
    if args.dcm :
        L_dataset, L_myo_pixel = normalization.Normalization_unwrap_dataset(
            L_dataset, 
            input_type=input_type,
            method="Norm_01_offset",
            offset=min_max_,
        ) 
    if args.roi :
        L_dataset[L_dataset<0.5] =0
        L_dataset[L_dataset>=0.5]=1
        L_myo_pixel = [ComputeInfarctExtent(data[idx_split_data:][coords_data]) for data in L_dataset]

    # Unwrapping data
    vertices_p2c, weights_p2c, idx_extended_nega, idx_extended_posi = \
        unwrap.Get_WeightsVertices(radial, theta, shape=shape, method="P2C")
    L_dataset = [unwrap.Polar2Cartesian(
        vertices_p2c, weights_p2c, 
        np.concatenate(
            [data[:,:,0][coords_data], 
                data[:,:,0][coords_data][idx_extended_nega], 
                data[:,:,0][coords_data][idx_extended_posi]]),
        convX=conv_x, shape=shape,
    ) for data in L_dataset]



    L_dataset = np.expand_dims(L_dataset, axis=-1)
    training_set = L_dataset[idx_split_data:]
    testing_set  = L_dataset[:idx_split_data]

    print("\nShape dataset :", L_dataset.shape)
    print("Shape training set :", training_set.shape)
    print("Shape testing set :", testing_set.shape, "\n")


    # -----------------------------------------------
    # Parameters during training
    # -----------------------------------------------

    latent_dim = args.latent_dim
    field_names = ['epochs', 'batch_size', 'lr', 'beta', 'latent_dim', 'convX',
                   'recons_error_mean', 'recons_error_std', 'quartile 1',
                   'recons_error_median', 'quartile 3', '5_worst_recons', 
                   '5_best_recons',
    ]
    params = {
        "epochs": args.epoch,
        "batch_size": args.batch_size,
        "lr": args.LR,
        "beta": args.beta,
        'latent_dim': latent_dim,
        'convX':conv_x,
    }


    # -----------------------------------------------
    # Callbacks
    # -----------------------------------------------

    callbacks_list = []
    callback_images = ImagesCallback.ImagesCallback(
                x=testing_set, 
                z_dim=latent_dim, 
                nb_images=1, 
                from_z=True, 
                from_random=False, 
                run_dir=os.path.join(save_dir, 'save_imgs'),
    )
    callback_bestmodel   = BestModelCallback.BestModelCallback(
                os.path.join(save_dir, 'save_model/best_model.h5'),
    )
    callback_tensorboard = TensorBoard(
                log_dir=os.path.join(save_dir, 'logs/'+str(datetime.now())[:18]), 
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=False,
                update_freq="epoch",
                profile_batch=0,
                embeddings_freq=0,
                embeddings_metadata=None,
    )
    callbacks_earlystopping = EarlyStopping(
                monitor="loss",
                min_delta=0.05,
                patience=25,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
    )

    callbacks_list.append(callback_images)
    callbacks_list.append(callback_bestmodel)
    callbacks_list.append(callback_tensorboard)
    callbacks_list.append(callbacks_earlystopping)


    # -----------------------------------------------
    # Training
    # -----------------------------------------------

    enc = encoder.Encoder_unwrap(latent_dim, training_set.shape[1:])
    enc.compile()
    dec = decoder.Decoder_unwrap(latent_dim, training_set.shape[1:])
    dec.compile()

    my_vae = vae.VAE(enc, dec, params["beta"],)
    my_vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]))

    history = my_vae.fit(
        training_set, 
        epochs=params["epochs"], 
        batch_size=params["batch_size"], 
        callbacks=callbacks_list, 
        shuffle=True,
        verbose=1,
    )

    plot_loss.get_history(history,  plot={"Loss":['loss','r_loss', 'kl_loss']}, save=save_dir)

    my_vae.reload(f'{save_dir}/save_model/best_model')


    # -----------------------------------------------
    # Erreur reconstruction
    # -----------------------------------------------

    testing_emb, z_var, emb  = my_vae.encoder.predict(testing_set)
    training_emb, z_var, emb = my_vae.encoder.predict(training_set)
    testing_decoded = my_vae.decoder.predict(testing_emb)


    diff_imgs_recons = testing_set - testing_decoded
    recons_error = (np.sqrt(np.square(diff_imgs_recons)).sum(axis=(1,2,3)))
    nb_pixel_err = np.size(testing_set[0])
    recons_error = ( recons_error / nb_pixel_err ) * 100
    params["recons_error_mean"] = recons_error.mean()
    params["recons_error_std"]  = np.std(recons_error)
    params["quartile 1"] = np.quantile(sorted(recons_error), q=0.25)
    params["recons_error_median"] = np.median((recons_error))
    params["quartile 3"] = np.quantile(sorted(recons_error), q=0.75)


    # WORST
    highest_mse, img_highest_dec, img_highest_set = ComputeRecons(recons_error, testing_set, testing_decoded, reverse=True, txt='worst')
    params["5_worst_recons"] = recons_error[highest_mse]

    distance, index = spatial.KDTree(training_emb).query(testing_emb[highest_mse,:], k=3)
    closest_pts = []
    for idx in index :
        decoded_pts = training_set[idx]
        closest_pts.append(decoded_pts)
    closest_pts = np.array(closest_pts)


    # BEST
    lowest_mse, img_lowest_dec, img_lowest_set = ComputeRecons(recons_error, testing_set, testing_decoded, reverse=False, txt='best')
    params["5_best_recons"] = recons_error[lowest_mse]

    distance, index = spatial.KDTree(training_emb).query(testing_emb[lowest_mse,:], k=3)
    closest_pts = []
    for idx in index :
        decoded_pts = training_set[idx]
        closest_pts.append(decoded_pts)
    closest_pts = np.array(closest_pts)


    # save params
    with open(os.path.join(save_dir,'params.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        writer.writerows([params])

    dict_patients_latent_info = {
        "emb" : my_vae.encoder.predict(L_dataset),
        "pat_name" : L_name_pat,
        "split_data" : args.split_data,
        "idx_split_data": idx_split_data,
        "color" : L_myo_pixel,
        "convX" : conv_x,
    }

    with open(os.path.join(save_dir, "latent_var_patients.pkl"), 'wb') as handle:
        pickle.dump(dict_patients_latent_info, handle, protocol=pickle.HIGHEST_PROTOCOL)  



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')

    inputs = parser.add_argument_group('Input group')
    inputs.add_argument("--inputDir", help="input directory that contains all the folder's patients", type=str, default="")
    inputs.add_argument("--ref", help="reference to get polar coords", type=str, default="")
    inputs.add_argument("--outputDir", help="input directory where to save the parameters of the training", type=str, default="")

    params = parser.add_argument_group('Parameters group')
    params.add_argument("--latent_dim", help="Latent dimension", type=int, default=2)
    params.add_argument("--convX", help="Latent dimension", type=int, default=9)
    params.add_argument("--beta", help="Loss weight for the KL", type=float, default=1)
    params.add_argument("--epoch", help="Latent dimension", type=int, default=30)
    params.add_argument("--batch_size", help="Latent dimension", type=int, default=32)
    params.add_argument("--LR", help="Latent dimension", type=float, default=1e-4)
    params.add_argument("--split_data", help="between 0 and 1 value", type=float, default=0.5)

    types_ds = parser.add_argument_group('Type data/seg')
    types_ds = types_ds.add_mutually_exclusive_group(required=True)
    types_ds.add_argument('--dcm', help="training with data type", type=bool, default=None)
    types_ds.add_argument('--roi', help="training with segm type", type=bool, default=None)


    args = parser.parse_args()
    main(args)    





























