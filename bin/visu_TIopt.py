import argparse
import os
import pickle
import sys

import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append('../')

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from PolarLV.process_func import format_DataAndCoord
from pydicom import dcmread
from scipy import stats



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def norm_0_1 (array, offset=None) :
    if offset : return (array - offset[0])/(offset[1] - offset[0])
    else: return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))

def plot_metric(val=None, color_pts=[], idx_min=0, slider=0, title="") :
    color_pts[idx_min] = "yellow"
    color_pts[slider] = "red"

    plot = go.Scatter(
        y=val, 
        mode='lines+markers',          
        marker=dict(color=color_pts),
        name=title,
    )
    fig = go.Figure()
    fig.add_trace(plot)
    fig.update_layout(
        height=600,
        autosize=True,
        title=title,
    )
    return plot, fig


def main(args, app) :
    dcm_MAG_data = None
    roi_MAG_data = None
    dcm_LGE_data  = None
    roi_LGE_data  = None

    if args.load_MAG_dicoms_from_dir and args.load_T1_roi_file :
        dcm_MAG, roi_MAG = format_DataAndCoord.process(
            args.load_MAG_dicoms_from_dir, 
            args.load_T1_roi_file,
            folderFormatedData = None,
            folderFigure = None,
            inverseDcm = 0, 
            display=False,
        )     
        dcm_MAG_data = dcm_MAG.data
        roi_MAG_data = roi_MAG.segments

    if args.load_LGE_file :
        with open(os.path.join(args.load_LGE_file, 'dcm.pkl'), 'rb') as filedcm:
            dcm_LGE = pickle.load(filedcm) 
            dcm_LGE_data = dcm_LGE.data
        with open(os.path.join(args.load_LGE_file, 'roi.pkl'), 'rb') as filedcm:
            roi_LGE = pickle.load(filedcm) 
            roi_LGE_data = roi_LGE.segments

    idx_slice = find_nearest(dcm_LGE.USliceLocation, dcm_MAG.USliceLocation[0])


    # MAG process
    myocard_zone_MAG = roi_MAG_data[0]['non-MI']
    remote_zone_MAG  = myocard_zone_MAG - roi_MAG_data[0]['MI']
    infarct_zone_MAG = roi_MAG_data[0]['MI']

    myocard_label_MAG = np.where(myocard_zone_MAG >= 0.1)
    remote_label_MAG  = np.where(remote_zone_MAG >= 0.1)
    infarct_label_MAG = np.where(infarct_zone_MAG >= 0.1)

    L_myocard_data_MAG = []
    for i in range (dcm_MAG_data.shape[-1]) :
        L_myocard_data_MAG.append(dcm_MAG_data[:,:,i][myocard_label_MAG])

    range_graph_MAG = [2000, 4000]
    size_bin_MAG = 75


    # LGE process
    myocard_zone_LGE = roi_LGE_data[idx_slice]['non-MI']
    remote_zone_LGE  = myocard_zone_LGE - roi_LGE_data[idx_slice]['MI']
    infarct_zone_LGE = roi_LGE_data[idx_slice]['MI']

    myocard_label_LGE = np.where(myocard_zone_LGE >= 0.1)
    remote_label_LGE  = np.where(remote_zone_LGE >= 0.1)
    infarct_label_LGE = np.where(infarct_zone_LGE >= 0.1)

    L_myocard_data_LGE = []
    for i in range (dcm_LGE_data.shape[-1]) :
        L_myocard_data_LGE.append(dcm_LGE_data[:,:,i][myocard_label_LGE])

    range_graph_LGE = [
        int(min(np.min(L_myocard_data_LGE, axis=1))),
        int(max(np.max(L_myocard_data_LGE, axis=1)))
    ]
    size_bin_LGE = 5



    app.layout = dbc.Container([
        # Plot img + hist + (box)
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='visu_data_MAG',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
            ], width=3,
            ),
            dbc.Col([
                dcc.Graph(
                    id='visu_data_LGE',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}

                ),
            ], width=3,
            ),
            dbc.Col([
                dcc.Graph(id='graph_histogram_MAG'),
            ], width=3,
            ),
            dbc.Col([
                dcc.Graph(id='graph_histogram_LGE'),
            ], width=3,
            ),
            # dbc.Col([
            #     dcc.Graph(id='graph_box'),
            # ], width=3,
            # ),
        ]),

        # Plot slider Instants/Slices
        dbc.Row([
            dbc.Col(
                html.Div([
                    "Instants :", 
                    dcc.Slider(
                        0,
                        np.shape(dcm_MAG_data)[-1]-1,
                        step=1,
                        id='slice--slider',
                        value=0,
                    ),
                ]),
                width=6,
            ),
            dbc.Col(
                html.Div([
                    "Normalization :", 
                    dcc.RadioItems(
                        ['without_norm', 'with_norm'],
                        'without_norm',
                        id='norm--raditem',
                    ),
                ]),
                width=1,
            ),
            dbc.Col(
                html.Div([
                    "Path to save figures :", 
                    dcc.Input(
                        id="path_save_folder--input", 
                        type="text", 
                        placeholder="", 
                        debounce=True
                    ),
                ]),
                width=1,
            ),
            dbc.Col(
                html.Div([
                    "Select figures to download :", 
                    dcc.Dropdown(
                        ['Histogram MAG', 'Histogram LGE',
                        'grad+std metrics', 'Combined metrics'],
                        multi=True,
                        id="graphs_to_save--dropdown",
                    )
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "Download figures :", 
                    dbc.Button("Download", id='download--button'),
                ]),
                width=1,
            ),
        ]),

        # graph Values row 1
        dbc.Row([
            dbc.Col(
                dcc.Graph(id="graph_combined_values"),
                width=6,
            ),
            dbc.Col(
                dcc.Graph(id="graph_metrics_values"),
                width=6,
            ),
        ]),


    dcc.Download(id="download-component"),

    ], fluid=True)



    @app.callback(
        Output("download-component", "data"),
        Input("download--button", "n_clicks"),
        Input("path_save_folder--input", "value"),
        Input("graphs_to_save--dropdown", "value"),
        Input('graph_histogram_MAG', 'figure'),
        Input('graph_histogram_LGE', 'figure'),
        Input('graph_metrics_values', 'figure'),
        Input('graph_combined_values', 'figure'),

        prevent_initial_call=True,
    )
    def func(n_clicks, path_save_folder, graphs_to_save, 
        graph1, graph2,
        graph3, graph4,
        ):
        if n_clicks != None :
            L_graphs = [
                graph1, graph2,
                graph3, graph4,
            ]
            L_graphs_MAGtle = [grph['layout']['title']['text'] for grph in L_graphs]

            for title_graph in graphs_to_save :
                idx_graph = L_graphs_MAGtle.index(title_graph)
                graph2dwnld = go.Figure(L_graphs[idx_graph])
                if path_save_folder != None :
                    graph2dwnld.write_image(os.path.join(path_save_folder, title_graph+".svg"))


    # Callbacks for DICOM data
    @app.callback(
        Output('visu_data_MAG', 'figure'),
        Output('visu_data_LGE', 'figure'),
        Input('slice--slider', 'value'),
        Input('norm--raditem', 'value'),
    )
    def update_graph(slice, norm):
        dcm_d_MAG = dcm_MAG_data[:,:,slice]
        dcm_d_LGE = dcm_LGE_data[:,:,idx_slice]
        range_color_MAG = [np.min(dcm_d_MAG), np.max(dcm_d_MAG)]
        range_color_LGE = [np.min(dcm_d_LGE), np.max(dcm_d_LGE)]

        if norm == "with_norm":
            dcm_d_MAG = (dcm_d_MAG - np.min(dcm_d_MAG[myocard_zone_MAG == 1])) / (np.max(dcm_d_MAG[myocard_zone_MAG == 1]) - np.min(dcm_d_MAG[myocard_zone_MAG == 1]))
            dcm_d_LGE = (dcm_d_LGE - np.min(dcm_d_LGE[myocard_zone_LGE == 1])) / (np.max(dcm_d_LGE[myocard_zone_LGE == 1]) - np.min(dcm_d_LGE[myocard_zone_LGE == 1]))
            range_color_MAG = [0,1]
            range_color_LGE = [0,1]


        fig_MAG = px.imshow(
            dcm_d_MAG,
            color_continuous_scale='gray',
            range_color=range_color_MAG,
        )
        fig_MAG.update_layout(
            height=600,
            autosize=True,
            title="Instant files from MAG ({}) :".format(str(dcm_MAG.USliceLocation[0])[:5]),
        )
        
        fig_LGE = px.imshow(
            dcm_d_LGE,
            color_continuous_scale='gray',
            range_color=range_color_LGE,
        )
        fig_LGE.update_layout(
            height=600,
            autosize=True,
            title="Corresponding LGE slice ({}) :".format(str(dcm_LGE.USliceLocation[idx_slice])[:5]),
        )

        return fig_MAG, fig_LGE


    # Callbacks for Histogram plot
    @app.callback(
        Output('graph_histogram_MAG', 'figure'),
        Output('graph_histogram_LGE', 'figure'),
        Input('slice--slider', 'value'),
    )
    def update_graph(slice):
        dcm_d_MAG = dcm_MAG_data[:,:,slice]
        dcm_d_LGE = dcm_LGE_data[:,:,idx_slice]

        remote_data_MAG  = dcm_d_MAG[remote_label_MAG]
        infarct_data_MAG = dcm_d_MAG[infarct_label_MAG]

        remote_data_LGE  = dcm_d_LGE[remote_label_LGE]
        infarct_data_LGE = dcm_d_LGE[infarct_label_LGE]

        fig_MAG = go.Figure()
        fig_MAG.add_trace(go.Histogram(
            x=infarct_data_MAG.ravel(),
            name="Infarct Zone",
            xbins={"start":range_graph_MAG[0], "end":range_graph_MAG[1], "size":size_bin_MAG},
        ))
        fig_MAG.add_trace(go.Histogram(
            x=remote_data_MAG.ravel(),
            name="Remote Zone",
            xbins={"start":range_graph_MAG[0], "end":range_graph_MAG[1], "size":size_bin_MAG},
        ))
        fig_MAG.update_layout(
            height=600,
            autosize=True,
            title="Histogram MAG",
            barmode='overlay',
        )
        fig_MAG.update_traces(opacity=0.75)
        fig_MAG.update_xaxes(range=range_graph_MAG)


        fig_LGE = go.Figure()
        fig_LGE.add_trace(go.Histogram(
            x=infarct_data_LGE.ravel(),
            name="Infarct Zone",
            xbins={"start":range_graph_LGE[0], "end":range_graph_LGE[1], "size":size_bin_LGE},
        ))
        fig_LGE.add_trace(go.Histogram(
            x=remote_data_LGE.ravel(),
            name="Remote Zone",
            xbins={"start":range_graph_LGE[0], "end":range_graph_LGE[1], "size":size_bin_LGE},
        ))
        fig_LGE.update_layout(
            height=600,
            autosize=True,
            title="Histogram LGE",
            barmode='overlay',
        )
        fig_LGE.update_traces(opacity=0.75)
        fig_LGE.update_xaxes(range=range_graph_LGE)

        return fig_MAG, fig_LGE


    # Callbacks for p-U values MW plot
    @app.callback(
        Output('graph_metrics_values', 'figure'),
        Output('graph_combined_values', 'figure'),
        Input('slice--slider', 'value'),
    )
    def update_graph(slider):
        L_p_value = []
        L_U_value = []
        L_med_value = []
        L_cen_value = []
        L_std = []
        L_val_remote_left = []
        L_val_remote_right = []
        plotbins = list(np.arange(start=range_graph_MAG[0], stop=range_graph_MAG[1]+size_bin_MAG, step=size_bin_MAG))
        for i in range (dcm_MAG_data.shape[-1]) :
            dcm_d_MAG = dcm_MAG_data[:,:,i]
            remote_data_MAG = dcm_d_MAG[remote_label_MAG]
            infarct_data_MAG = dcm_d_MAG[infarct_label_MAG]
            myocard_data_MAG = dcm_d_MAG[myocard_label_MAG]
            if np.all(myocard_data_MAG) == 0 : continue

            U1, p = stats.mannwhitneyu(infarct_data_MAG, remote_data_MAG)
            U2 = len(infarct_data_MAG)*len(remote_data_MAG) - U1
            U = min(U1,U2)
            D_med = abs(np.median(infarct_data_MAG) - np.median(remote_data_MAG)) / range_graph_MAG[1]
            D_center = ((np.median(infarct_data_MAG) + np.median(remote_data_MAG))/2)/ \
                       ((range_graph_MAG[1]+range_graph_MAG[0])/2)
            val_std  = np.std(myocard_data_MAG)

            val_remote_mean = np.mean(remote_data_MAG)
            val_remote_std  = np.std(remote_data_MAG)
            val_remote_left  = (val_remote_mean - 2*val_remote_std) - range_graph_MAG[0]
            val_remote_right = (val_remote_mean + 2*val_remote_std) - range_graph_MAG[1]


            if val_remote_left < 0: L_val_remote_left.append(1)
            else: L_val_remote_left.append(0)

            if val_remote_right > 0: L_val_remote_right.append(1)
            else: L_val_remote_right.append(0)

            L_p_value.append(p)
            L_U_value.append(U)
            L_med_value.append(D_med)
            L_cen_value.append(D_center)
            L_std.append(val_std)


        norm_U_value   = list(norm_0_1(L_U_value))
        idx_max = norm_U_value.index(max(norm_U_value))

        norm_med_value = list(1 - np.array(L_med_value))
        norm_cen_value = list(abs(1 - np.array(L_cen_value)))
        norm_std  = list(norm_0_1(max(L_std) - L_std))
        Loss = list(np.sum([norm_U_value, norm_med_value, norm_cen_value], axis=0))


        # Condition in case there are only zero values
        if np.all(L_val_remote_left[idx_max:])  == 0 : L_val_remote_left[idx_max] = 1
        if np.all(L_val_remote_right[idx_max:]) == 0 : L_val_remote_right[-1] = 1

        idx_last_one_left = np.max(np.nonzero(L_val_remote_left)) + 1
        idx_first_one_right = np.min(np.nonzero(L_val_remote_right))
        slope, intercept, _, _, _ = stats.linregress([idx_last_one_left, idx_first_one_right],[0,1])


        L_sat = list(np.concatenate((
            np.ones(idx_last_one_left), 
            slope*np.arange(0,idx_first_one_right-idx_last_one_left+1,1), 
            np.ones(len(L_val_remote_right)-idx_first_one_right-1),
        )))

        lbd = 0.75
        L_custom_func = list((1-lbd)*np.array(L_sat) + lbd*np.array(norm_std))



        U_plot, fig_U = plot_metric(
            norm_U_value, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=norm_U_value.index(min(norm_U_value[idx_max:]), idx_max),
            slider=slider,
            title="U values",
        )
        Med_plot, _ = plot_metric(
            norm_med_value, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=norm_med_value.index(min(norm_med_value[idx_max:]), idx_max),
            slider=slider,
            title="Med values",
        )
        Cen_plot, _ = plot_metric(
            norm_cen_value, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=norm_cen_value.index(min(norm_cen_value[idx_max:]), idx_max),
            slider=slider,
            title="Cen values",
        )
        Loss_plot, _ = plot_metric(
            Loss, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=Loss.index(min(Loss[idx_max:]), idx_max),
            slider=slider,
            title="Loss values",
        )
        sat_plot, _ = plot_metric(
            L_sat, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=L_sat.index(min(L_sat[idx_max:]), idx_max),
            slider=slider,
            title="Saturation values",
        )
        std_plot, _ = plot_metric(
            norm_std, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=norm_std.index(min(norm_std[idx_max:]), idx_max),
            slider=slider,
            title="std values",
        )
        custom_func_plot, _ = plot_metric(
            L_custom_func, 
            color_pts=["blue" for _ in range(dcm_MAG_data.shape[-1])], 
            idx_min=L_custom_func.index(min(L_custom_func[idx_max:]), idx_max),
            slider=slider,
            title="Custom func values",
        )

        slice_list = sorted(os.listdir(args.load_MAG_dicoms_from_dir))
        L_invTime = [dcmread(os.path.join(args.load_MAG_dicoms_from_dir, slice_)).InversionTime for slice_ in slice_list]
        
        fig_grad_std = make_subplots(specs=[[{"secondary_y": True}]])
        fig_grad_std.add_trace(std_plot)
        fig_grad_std.add_trace(sat_plot)
        fig_grad_std.add_trace(custom_func_plot)
        fig_grad_std.update_layout(
            height=600,
            autosize=True,
            title="sat+std metrics | lbd="+str(lbd),
            xaxis = dict(
                tickmode = 'array',
                tickvals = [i for i in range(len(L_invTime))],
                ticktext = [str(elem) for elem in L_invTime],
            ),
        )
        fig_metrics = make_subplots(specs=[[{"secondary_y": True}]])
        fig_metrics.add_trace(U_plot)
        fig_metrics.add_trace(Med_plot)
        fig_metrics.add_trace(Cen_plot)
        fig_metrics.add_trace(Loss_plot)
        fig_metrics.update_layout(
            height=600,
            autosize=True,
            title="Combined metrics",
        )

        return fig_grad_std, fig_metrics



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')

    inputs = parser.add_argument_group('Input group')
    inputs.add_argument("--load_MAG_dicoms_from_dir", help="Dicom files in a folder for a specific patient", type=str, default="")
    inputs.add_argument("--load_T1_roi_file", help="csv file of the segmentation", type=str, default="")
    inputs.add_argument("--load_LGE_file", help="formated file (.dcm) of the specific patient", type=str, default="")

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    args = parser.parse_args()
    main(args, app)    
    app.run_server(debug=True)

















        