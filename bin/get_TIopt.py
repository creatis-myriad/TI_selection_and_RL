import argparse
import os
import pickle
import sys

sys.path.append("../")

from PolarLV.process_func import format_DataAndCoord, get_best_MAG
from PolarLV.common.funFind import GetPaths
from PolarLV.common.funProcess import pairs_sequence



def main(args) :
    path_dicom_dir = args.inputDir_Dicom
    path_roi_dir   = args.inputDir_Roi
    method = args.method
    option = args.option

    L_dicom_dir = sorted(os.listdir(path_dicom_dir))
    L_roi_dir   = sorted(os.listdir(path_roi_dir))

    L_TIopt = []
    L_pat_name = []
    L_seq_name = []
    for dicom_dir, roi_dir in zip(L_dicom_dir, L_roi_dir) :
        """
        Condition moche pour enlever les patients qui n'étaient pas pris 
        en compte lors des expériences + articles
        """
        if dicom_dir in ["0042_M12", "0109_M12", "0111_M12", "0147_M12", "0163_M12"] : continue

        path_dicom = os.path.join(path_dicom_dir, dicom_dir)
        path_roi   = os.path.join(path_roi_dir, roi_dir)

        L_dcm_dir   = sorted(os.listdir(path_dicom))
        L_roi_files = GetPaths(path_roi, name_files=['.csv'], match_patern="post")
        L_roi_files = sorted(L_roi_files[".csv"])

        L_dcm_dir, L_roi_files = pairs_sequence(L_dcm_dir, L_roi_files)

        for dcm_dir, roi_file in zip(L_dcm_dir, L_roi_files) :
            L_pat_name.append(dicom_dir)
            L_seq_name.append(dcm_dir)
            
            path_dcm_dir = os.path.join(path_dicom, dcm_dir)
            path_roi_file = os.path.join(path_roi, roi_file)

            dcm_MAG, roi_MAG = format_DataAndCoord.process(
                path_dcm_dir, 
                path_roi_file,
                folderFormatedData = None,
                folderFigure = None,
                inverseDcm = 0, 
                display=False,
            )     

            idx_best_MAG = get_best_MAG.get_idx_best_MAG(
                dcm_MAG, roi_MAG, 
                method=method, 
                range_graph_MAG=[2000,4000],
                option=option
            ) 
            L_TIopt.append(idx_best_MAG)

    D_TIopt = {
        "pat_name" : L_pat_name,
        "seq_name" : L_seq_name,
        "TIopt" : L_TIopt,
    }
    with open(os.path.join(args.outputDir, "idx_TIopt_MAG_"+method+"_"+option+".pkl"), 'wb') as handle:
        pickle.dump(D_TIopt, handle, protocol=pickle.HIGHEST_PROTOCOL)  



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')

    inputs = parser.add_argument_group('Input group')
    inputs.add_argument("--inputDir_Dicom", help="input directory that contains patients' folder with dicom files", type=str, required=True)
    inputs.add_argument("--inputDir_Roi", help="input directory that contains patients' folder with dicom files", type=str, required=True)
    inputs.add_argument("--method", help="3 methods possible : old_loss; std; std+sat", type=str, default="std+sat")
    inputs.add_argument("--option", help="3 option possible : std; contrast; cnr", type=str, default="std")
    inputs.add_argument("--outputDir", help="output directory path", type=str, required=True)


    args = parser.parse_args()
    main(args)    

















