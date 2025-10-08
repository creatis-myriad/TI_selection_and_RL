import numpy as np
import matplotlib.pyplot as plt


def Norm_0_1 (array, offset=None) :
    if offset : return (array - offset[0])/(offset[1] - offset[0])
    else: return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))

def Center_reduce(array) :
    return (array - np.nanmean(array)) / np.nanstd(array)


def Normalization_dataset(dataset, input_type=None, method=None, offset=None) :
    print("Preprocessing", input_type, "input data ...")
    L_myo_pixel = []
    L_new_dataset = []
    for data in dataset :
        if input_type == "BE" :
            data[data >= 0.999] = 1
            new_data = 1-data
            myo_pixel = np.sum(new_data) / len(new_data[new_data !=0])
            new_data[new_data == 0] = -1    

            L_new_dataset.append(new_data)
            L_myo_pixel.append(myo_pixel)      


        elif input_type == "LGE_2D" :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            for i in range(np.shape(new_data)[-1]) :
                img = new_data[:,:,i]
                myo_pixel = np.nansum(img) / len(img[np.where(~np.isnan(img))])
                img[np.where(np.isnan(img))] = -1

                L_new_dataset.append(img)
                L_myo_pixel.append(myo_pixel)

        
        elif input_type == "LGE_3D" :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            img_3D = new_data[:,:,7:14]
            myo_pixel = np.nansum(img_3D) / len(img_3D[np.where(~np.isnan(img_3D))])
            img_3D[np.where(np.isnan(img_3D))] = -1

            L_new_dataset.append(img_3D)
            L_myo_pixel.append(myo_pixel)    


        elif input_type in ["T1", "T1_pre", "T1_post", "MAG"] :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            img = new_data[:,:,0]
            myo_pixel = np.nansum(img) / len(img[np.where(~np.isnan(img))])
            img[np.where(np.isnan(img))] = -1

            L_new_dataset.append(img)
            L_myo_pixel.append(myo_pixel)  
    
            
        else :
            print("\nThis modality has not been implemented yet\n")
            exit()

    return np.array(L_new_dataset), L_myo_pixel


def Normalization_unwrap_dataset(dataset, input_type=None, method=None, offset=None) :
    print("Preprocessing", input_type, "input data ...")
    L_myo_pixel = []
    L_new_dataset = []
    for data in dataset :
        if input_type == "LGE_2D" :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            for i in range(np.shape(new_data)[-1]) :
                img = new_data[:,:,i]
                myo_pixel = np.sum(img) / (img.size)
                L_new_dataset.append(img)
                L_myo_pixel.append(myo_pixel)

        
        elif input_type == "LGE_3D" :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            img_3D = new_data[:,:,7:14]
            myo_pixel = np.sum(img_3D) / img_3D.size
            L_new_dataset.append(img_3D)
            L_myo_pixel.append(myo_pixel)    


        elif input_type in ["T1", "T1_pre", "T1_post", "MAG"] :
            if method == "Norm_01" :
                new_data = Norm_0_1(data)
            if method == "Norm_01_offset" :
                new_data = Norm_0_1(data, offset=offset)
            if method == "Center_reduce" :
                new_data = Center_reduce(data)

            myo_pixel = np.sum(new_data) / (new_data.size)
            L_new_dataset.append(new_data)
            L_myo_pixel.append(myo_pixel)  
            
        else :
            print("\nThis modality has not been implemented yet\n")
            exit()

    return np.array(L_new_dataset), L_myo_pixel


