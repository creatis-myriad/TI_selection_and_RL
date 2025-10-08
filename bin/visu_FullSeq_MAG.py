import argparse
import os
import pickle
import sys

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import uncertainties as unc

sys.path.append("../")

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from nn_libr.model import vae
from nn_libr.model import ar_vae
from nn_libr.paths.IO_paths import GetPaths
from nn_libr.common.functions import ComputeAngleJunction, ComputeEndoSurfArea, ComputeTransmurality, ComputeInfarctExtent
from nn_libr.preprocess import unwrap
from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy import stats



def f_lin(x, a, b):
    return a * x + b

def predband(x, xd, yd, p, func, conf=0.95):
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb

def ComputeRegression(x, y, func) :
    n = len(x)
    popt, pcov = curve_fit(func, x, y)
    a = popt[0]
    b = popt[1]
    r2 = 1.0-(sum((y-func(x,a,b))**2)/((n-1.0)*np.var(y,ddof=1)))

    a,b = unc.correlated_values(popt, pcov)
    px = np.linspace(np.min([x, y]),np.max([x, y]),9)
    py = a*px+b
    
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    lpb, upb = predband(px, x, y, popt, func, conf=0.95)
    return px, nom, std, lpb, upb, r2

def find_nearest_arr(array, arr_):
    array = np.asarray(array)
    idx = (np.sum(np.abs(array - arr_), axis=-1)).argmin()
    return idx

def norm_0_1 (array, offset=None) :
    if offset : return (array - offset[0])/(offset[1] - offset[0])
    else: return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))





# Pour Run le code sans utiliser la GPU 
# sinon marche pas sur le server
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def main(args, app):
    with open(args.ref, 'rb') as file:
        ref = pickle.load(file)

    path_LS = args.input_LS_var
    with open(path_LS, 'rb') as file_lsv:
        ls_var = pickle.load(file_lsv)

    paths_MAG = GetPaths(args.inputDir, name_files=['dcm.pkl', 'roi.pkl'])
    paths_LGE = GetPaths(args.data_to_project, name_files=['dcm.pkl', 'roi.pkl'])
    L_paths_MAG, L_paths_LGE = sorted(paths_MAG["dcm.pkl"]), sorted(paths_LGE["dcm.pkl"])
    L_paths_MAG_roi, L_paths_LGE_roi = sorted(paths_MAG["roi.pkl"]), sorted(paths_LGE["roi.pkl"])

    D_data = {}
    if args.vae :    my_vae = vae.VAE()
    if args.ar_vae : my_vae = ar_vae.AR_VAE()
    my_vae.reload(os.path.join(args.load_nn_model,"best_model"))
    idx_split_data = ls_var["idx_split_data"]
    if args.fMAG : slices_ = 40
    else :         slices_ = 3

    if args.TIopt :
        with open(args.TIopt, 'rb') as file_TIopt:
            TIopt_file = pickle.load(file_TIopt)
        L_TIopt = np.array(TIopt_file["TIopt"])
    else : L_TIopt = np.array([1 for _ in range (int(len(L_paths_MAG)/slices_))])

    conv_x = ls_var["convX"]
    coordinates = ref.roi.coordinates[:,:,:,0]
    radial, theta = coordinates[:,:,0], coordinates[:,:,1]
    coords_data = np.where(~np.isnan(radial))
    shape=(512-2*(conv_x-1), 16)
    vertices_p2c, weights_p2c, idx_extended_nega, idx_extended_posi = \
        unwrap.Get_WeightsVertices(radial, theta, shape=shape, method="P2C")
    vertices_c2p, weights_c2p, _, _ = \
        unwrap.Get_WeightsVertices(radial, theta, shape=shape, method="C2P")
    

    # Load Embedding
    embedding, _, _ = ls_var["emb"]
    # emb = (embedding - np.mean(embedding, axis=0))/np.std(embedding, axis=0)
    emb = np.copy(embedding)
    emb = np.array([x for _,x in sorted(zip(ls_var["pat_name"],emb))])
    D_data["L_pat_name_MAG"] = sorted(ls_var["pat_name"])

    # Load Data MAG
    D_data["L_data_MAG"] = []
    D_data["L_data_MAG_roi"] = []
    for path_MAG, path_MAG_roi in zip(L_paths_MAG, L_paths_MAG_roi) :
        with open(path_MAG, 'rb') as file_MAG:
            data_MAG = pickle.load(file_MAG)
        with open(path_MAG_roi, 'rb') as file_MAG:
            data_MAG_roi = pickle.load(file_MAG)
        data_MAG = data_MAG.data[:,:,0]
        data_MAG_roi = data_MAG_roi.segments[0]['MI']
        D_data["L_data_MAG"].append(data_MAG)
        D_data["L_data_MAG_roi"].append(data_MAG_roi)
    D_data["L_data_MAG"] = np.array(D_data["L_data_MAG"])
    D_data["L_data_MAG_roi"] = np.array(D_data["L_data_MAG_roi"])

    set_MAG = ["test" for _ in range(idx_split_data+1)] + ["train" for _ in range(len(emb) - idx_split_data+1)]
    set_MAG = [x for _,x in sorted(zip(ls_var["pat_name"],set_MAG))]

    # Normalisation MAG 
    min_max_MAG = [ 
        np.nanmin(D_data["L_data_MAG"]),
        np.nanmax(D_data["L_data_MAG"])
    ]
    D_data["L_data_MAG"] = norm_0_1(D_data["L_data_MAG"], offset=min_max_MAG)
    D_data["L_data_MAG_roi"][D_data["L_data_MAG_roi"]<0.5]  = 0
    D_data["L_data_MAG_roi"][D_data["L_data_MAG_roi"]>=0.5] = 1

    D_data["L_data_MAG_reshape"] = np.reshape(
        D_data["L_data_MAG"], 
        (int(len(D_data["L_data_MAG"])/slices_), slices_, 256, 256)
    )
    D_data["L_data_MAG_roi_reshape"] = np.reshape(
        D_data["L_data_MAG_roi"], 
        (int(len(D_data["L_data_MAG_roi"])/slices_), slices_, 256, 256)
    )
    

    # Load Data LGE
    D_data["L_data_LGE"] = []
    D_data["L_data_LGE_roi"] = []
    D_data["idx_LGE_FL"] = []
    D_data["idx_LGE_DE"] = []
    cpt = 0
    for path_LGE, path_LGE_roi in zip(L_paths_LGE, L_paths_LGE_roi) :
        with open(path_LGE, 'rb') as file_LGE:
            data_LGE = pickle.load(file_LGE)
        with open(path_LGE_roi, 'rb') as file_LGE:
            data_LGE_roi = pickle.load(file_LGE)
        data_LGE = data_LGE.data[:,:,0]
        data_LGE_roi = data_LGE_roi.segments[0]['MI']
        if "fl3d" in path_LGE : D_data["idx_LGE_FL"].append(cpt)
        if "DE" in path_LGE   : D_data["idx_LGE_DE"].append(cpt)
        D_data["L_data_LGE"].append(data_LGE)
        D_data["L_data_LGE_roi"].append(data_LGE_roi)
        cpt+=1
    D_data["L_data_LGE"] = np.array(D_data["L_data_LGE"])
    D_data["L_data_LGE_roi"] = np.array(D_data["L_data_LGE_roi"])
    D_data["L_pat_name_LGE"] = [pn.split("/")[0]+"__"+pn.split("/")[-2][-3:] 
                              for pn in sorted(ls_var["pat_name"])[0::slices_]]

    # Normalisation LGE
    min_max_FL = [ 
        np.nanmin(D_data["L_data_LGE"][D_data["idx_LGE_FL"]]),
        np.nanmax(D_data["L_data_LGE"][D_data["idx_LGE_FL"]])
    ]
    min_max_DE = [ 
        np.nanmin(D_data["L_data_LGE"][D_data["idx_LGE_DE"]]),
        np.nanmax(D_data["L_data_LGE"][D_data["idx_LGE_DE"]])
    ]
    D_data["L_data_LGE"][D_data["idx_LGE_FL"]] = norm_0_1(
        D_data["L_data_LGE"][D_data["idx_LGE_FL"]], offset=min_max_FL)
    D_data["L_data_LGE"][D_data["idx_LGE_DE"]] = norm_0_1(
        D_data["L_data_LGE"][D_data["idx_LGE_DE"]], offset=min_max_DE)
    D_data["L_data_LGE_roi"][D_data["L_data_LGE_roi"]<0.5]  = 0
    D_data["L_data_LGE_roi"][D_data["L_data_LGE_roi"]>=0.5] = 1


    # Compute Metrics All Points
    L_tran_data = np.zeros((len(L_paths_LGE), 2))
    L_esa_data  = np.zeros((len(L_paths_LGE), 2))
    L_ext_data  = np.zeros((len(L_paths_LGE), 2))
    L_ang_data  = np.zeros((len(L_paths_LGE), 2))
    cpt = 0
    for data_MAG, data_LGE, data_MAG_roi, data_LGE_roi in \
        zip(D_data["L_data_MAG_reshape"], D_data["L_data_LGE"], D_data["L_data_MAG_roi_reshape"], D_data["L_data_LGE_roi"]) :       

        data_MAG = data_MAG[L_TIopt[cpt],:,:,]
        data_MAG_roi = data_MAG_roi[L_TIopt[cpt],:,:,]
        # Compute Tran metric
        tran_LGE_data = ComputeTransmurality(data_LGE_roi, theta, radial)
        tran_MAG_data = ComputeTransmurality(data_MAG_roi, theta, radial)

        # Compute ESA  metric
        esa_LGE_data = ComputeEndoSurfArea(data_LGE_roi, radial)
        esa_MAG_data = ComputeEndoSurfArea(data_MAG_roi, radial)

        # Compute Infarct Extent metric
        ext_LGE_data = ComputeInfarctExtent(data_LGE[coords_data])
        ext_MAG_data = ComputeInfarctExtent(data_MAG[coords_data])

        # Compute Angle Junction metric
        ang_LGE_data, _ = ComputeAngleJunction(            
            data_LGE_roi[coords_data],
            coords_data,
            ref.roi.origines[:,0], 
            ref.roi.endoCenters[:,0],
        )
        ang_MAG_data, _ = ComputeAngleJunction(            
            data_MAG_roi[coords_data],
            coords_data,
            ref.roi.origines[:,0], 
            ref.roi.endoCenters[:,0],
        )

        # Add metrics
        L_tran_data[cpt,:] = [tran_MAG_data, tran_LGE_data]
        L_esa_data[cpt,:]  = [esa_MAG_data, esa_LGE_data]
        L_ext_data[cpt,:]  = [ext_MAG_data, ext_LGE_data]
        L_ang_data[cpt,:]  = [ang_MAG_data/360, ang_LGE_data/360]
        cpt+=1

    L_metric = np.stack([
        L_tran_data,
        L_esa_data,
        L_ext_data,
        L_ang_data,
    ], axis=0,    
    )


    # Unwrapping data
    L_unwr_LGE = [unwrap.Polar2Cartesian(
        vertices_p2c, weights_p2c, 
        np.concatenate(
            [data[coords_data], 
             data[coords_data][idx_extended_nega], 
             data[coords_data][idx_extended_posi]]),
        convX=conv_x, shape=shape,
    ) for data in D_data["L_data_LGE"]]

    # Creation embedding LGE 
    D_data["L_emb_LGE"] = []
    for elem in L_unwr_LGE:
        elem = np.expand_dims(elem, axis=0)
        elem = np.expand_dims(elem, axis=-1)
        enc_data, _, _ = my_vae.encoder.predict(elem)
        D_data["L_emb_LGE"].append(np.squeeze(enc_data))
    # D_data["L_emb_LGE"] = (D_data["L_emb_LGE"] - np.mean(embedding, axis=0))/np.std(embedding, axis=0)
    D_data["L_emb_LGE"] = np.array(D_data["L_emb_LGE"])

    # Compute Dist full seq MAG
    i, cpt = 0, 0
    L_dist_seqMAG_LGE  = []
    L_dist_img_aligned = []
    for vecM, img_MAG in zip(emb, D_data["L_data_MAG"]) :
        vecL = D_data["L_emb_LGE"][i]
        img_LGE = D_data["L_data_LGE"][i]
        dist_LS = distance.euclidean(vecM, vecL)
        dist_Al = distance.euclidean(img_MAG[coords_data], img_LGE[coords_data])
        cpt+=1
        if cpt%slices_ == 0 : i+=1
        L_dist_seqMAG_LGE.append(dist_LS)
        L_dist_img_aligned.append(dist_Al)
    L_dist_seqMAG_LGE = np.reshape(L_dist_seqMAG_LGE, (len(D_data["L_emb_LGE"]), slices_))
    L_dist_img_aligned = np.reshape(L_dist_img_aligned, (len(D_data["L_emb_LGE"]), slices_))
    L_dist_TInear = np.min(L_dist_seqMAG_LGE, axis=-1)


    # Get emb MAG opt
    emb_best_MAG = np.array([emb[L_TIopt[i]+i*slices_] for i in range(len(D_data["L_data_MAG_reshape"]))])

    # Get emb MAG near
    cpt = 0
    L_idx_LGEnear = []
    for elem in L_dist_seqMAG_LGE :
        idx_min = list(elem).index(L_dist_TInear[cpt])
        L_idx_LGEnear.append(idx_min)
        cpt+=1
    tmp_ = np.reshape(emb, (len(D_data["L_emb_LGE"]), slices_, emb.shape[-1]))
    emb_MAG_near = np.zeros((len(L_idx_LGEnear), emb.shape[-1]))
    for i in range(len(L_idx_LGEnear)): emb_MAG_near[i,:] = tmp_[i, L_idx_LGEnear[i], :]


    app.layout = dbc.Container([
        # Radio Items
        dbc.Row([
            dbc.Col(
                html.Div([
                    "Single vs. All patients :", 
                    dcc.RadioItems(
                        ['Single', 'All'], 
                        'Single',
                        id="ChooseNbPat--raditem",
                    ),
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "Choose Distance Space:", 
                    dcc.RadioItems(
                        ['Dist_Img_Aligned', 'Dist_Vec_LS'], 
                        'Dist_Vec_LS',
                        id="ChooseDistSpace--raditem",
                    ),
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "Choose Pat:", 
                    dcc.Dropdown(
                        D_data["L_pat_name_LGE"], 
                        D_data["L_pat_name_LGE"][0],
                        id="ChoosePat--dropdown",
                    ),
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "Choose Metric:", 
                    dcc.Dropdown(
                        ["Trans","ESA","Extent","Angle",],
                        "Trans",
                        id="ChooseMetric--dropdown",
                    ),
                ]),
                width=2,
            ),
        ]),
        
        # Plots TrajDist + Histo
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='Plot_TrajDist',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
                dcc.Graph(
                    id='Plot_HistTi',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
            ],
                width=6,
            ),
            dbc.Col([
                dcc.Graph(
                    id='PlotRegressionMetric',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
                dcc.Graph(
                    id='ReconsSynthImg',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
            ],
                width=6,
            ),
        ]),


        ], fluid=True)


    # Plot Dist vs TIs
    @app.callback(
        Output('Plot_TrajDist', 'figure'),
        Input('ChooseNbPat--raditem', 'value'),
        Input('ChoosePat--dropdown', 'value'),
        Input('ChooseDistSpace--raditem', 'value'),
    )
    def func(chooseNbPat, choosePat, chooseDistSpace):
        if chooseDistSpace == "Dist_Img_Aligned": dist_space = L_dist_img_aligned
        if chooseDistSpace == "Dist_Vec_LS": dist_space = L_dist_seqMAG_LGE

        L_TIs = [np.arange(0,slices_,1) for _ in range (len(D_data["L_data_LGE"]))]
        L_delta_TI = np.array([np.array(elem)-(L_TIopt[cpt]) for cpt, elem in enumerate(L_TIs)])

        if chooseNbPat == "Single":
            idx_pat = D_data["L_pat_name_LGE"].index(choosePat)
            x_axis_ = np.expand_dims(dist_space[idx_pat, :], axis=0)
            y_axis_ = np.expand_dims(L_delta_TI[idx_pat], axis=0)
        else:
            x_axis_ = dist_space
            y_axis_ = L_delta_TI
        xaxis_title_ = "Distance 8-dims"
        yaxis_title_ = "TIs"


        fig = go.Figure()
        for i in range (len(x_axis_)) :
            plot_dist_dist = go.Scatter(
                x=x_axis_[i], 
                y=y_axis_[i], 
                mode='lines+markers',
                marker=dict(color="blue"),
                name=D_data["L_pat_name_LGE"][i],
                opacity=0.75,
            )
            fig.add_trace(plot_dist_dist)

        fig.update_layout(
            height=550,
            autosize=True,
            xaxis_title=xaxis_title_,
            xaxis_range=[0,np.max(dist_space)+0.5],
            yaxis_title=yaxis_title_,
            yaxis_range=[np.min(L_delta_TI)-0.5,np.max(L_delta_TI)+0.5],
            margin=dict(t=25, b=0, l=0, r=0),
            uirevision=True,
        )
        return fig


    # Plot Histo TIs
    @app.callback(
        Output('Plot_HistTi', 'figure'),
        Input('ChooseDistSpace--raditem', 'value'),
    )
    def func(chooseDistSpace):
        if chooseDistSpace == "Dist_Img_Aligned": dist_space = L_dist_img_aligned
        if chooseDistSpace == "Dist_Vec_LS": dist_space = L_dist_seqMAG_LGE

        L_closest_TI = np.argmin(dist_space, axis=-1)
        delta_TI     = L_closest_TI - (np.array(L_TIopt))

        fig = px.histogram(delta_TI, nbins=42)
        fig.update_layout(
            height=550,
            autosize=True,
            xaxis_title="bins TIs",
            xaxis_range=[-10,10],
            yaxis_title="count histo",
            margin=dict(t=25, b=0, l=0, r=0),
            uirevision=True,
        )
        return fig


    # Plot Regression Metrics
    @app.callback(
        Output('PlotRegressionMetric', 'figure'),
        Input('ChooseMetric--dropdown', 'value'),
    )
    def func(ChooseMetric):
        if ChooseMetric ==  "Trans" : idx_ = 0
        if ChooseMetric ==  "ESA"   : idx_ = 1
        if ChooseMetric ==  "Extent": idx_ = 2
        if ChooseMetric ==  "Angle" : idx_ = 3

        fig = go.Figure()

        # MAGopt :
        px_, nom, std, lpb, upb, r2 = ComputeRegression(
            emb_best_MAG[:,idx_], L_metric[idx_,:,0], f_lin
        )

        data_pts_MAGopt = go.Scatter(
            x=emb_best_MAG[:,idx_], 
            y=L_metric[idx_,:,0], 
            mode='markers',
            marker=dict(color="blue"),
            # opacity=0.75,
            name="Points MAGopt ",
            text=set_MAG[0::slices_],
        )
        fig.add_trace(data_pts_MAGopt)

        linear_regression_MAG = go.Scatter(
            x=px_,
            y=nom,
            line=dict(color="darkblue"),
            name="Linear regression r2="+str(r2)[:5],
            marker_symbol="diamond",
        ) 
        fig.add_trace(linear_regression_MAG)

        # LGE :
        px_, nom, std, lpb, upb, r2 = ComputeRegression(
            D_data["L_emb_LGE"][:,idx_], L_metric[idx_,:,1], f_lin
        )
        data_pts_LGE = go.Scatter(
            x=D_data["L_emb_LGE"][:,idx_], 
            y=L_metric[idx_,:,1], 
            mode='markers',
            marker=dict(color="red"),
            # opacity=0.75,
            name="Points LGE ",
            marker_symbol="square",
            # text=D_data["L_pat_name_MAG"][0::slices_],
        )
        fig.add_trace(data_pts_LGE)

        linear_regression_LGE = go.Scatter(
            x=px_,
            y=nom,
            line=dict(color="darkred"),
            name="Linear regression r2="+str(r2)[:5],
            marker_symbol="diamond",
        ) 
        fig.add_trace(linear_regression_LGE)


        fig.update_layout(
            height=550,
            autosize=True,
            margin=dict(t=25, b=0),
            uirevision=True,
            template="plotly_white"
        )
        return fig



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')

    inputs = parser.add_argument_group('Input group')
    inputs.add_argument("--input_LS_var", help="Latent space var", type=str, default="")
    inputs.add_argument("--load_nn_model", help="", type=str, default="")

    inputs.add_argument("--inputDir", help="Data", type=str, default="")
    inputs.add_argument("--data_to_project", help="", type=str, default="")
    inputs.add_argument("--ref", help="load ref to have polar coords", type=str, default="")

    inputs.add_argument("--fMAG", help="bool, if full sequence of MAG is loaded", type=bool, default=False)
    inputs.add_argument("--TIopt", help="bool, if full TIopt file or not", type=str, default="")

    nn_model = parser.add_argument_group('choose model type')
    nn_model = nn_model.add_mutually_exclusive_group(required=True)
    nn_model.add_argument('--vae', help="VAE nn model", type=bool, default=None)
    nn_model.add_argument('--ar_vae', help="AR VAE nn model", type=bool, default=None)

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    args = parser.parse_args()
    main(args, app)    
    app.run_server(debug=True, port=8053)

