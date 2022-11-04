# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:01:15 2022

@author: JPCli
"""


from dash import Dash, dcc, html, dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import io
import numpy
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import base64
import plotly.graph_objs as go
import plotly.express as px

class CNNDetector:
    def __init__(self, checkpoint_file, input_shape=(64,64) ):        
        self.input_shape = input_shape
        self.model = load_model(checkpoint_file)        
        
    def predict_image_file(self, img):        
        # img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(int(self.input_shape[0]),int(self.input_shape[1]))) 
        img_converted = img.reshape(1,64,64,3)
        return self.model.predict(img_converted)

CHECKPOINT_FILE = r'model\simplenet_cracks8020_weights.01-0.04.hdf5'
INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64
cnn = CNNDetector(CHECKPOINT_FILE, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2('Upload Images'),
    html.Br(),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Dropdown(
        id='output_dropdown',
        # options=None,
        value=None,
        style= {'display':'none'}
        
    ),
    html.Div(id='output-image-upload'),
    html.Br(),
    
    html.H2('Upload Position Data'),
    html.Br(),
    dcc.Upload(
        id='upload-position-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='all_position_table'),
    html.Div(id='all_position'),
    html.Br(),
    
    html.Button('Cracks Detection', id='detection_button', n_clicks=0),
    html.Br(),
    html.Div(id='results'),
    html.Br(),
    html.Button('Check Crack Distribution', id='crack_pos_button', n_clicks=0),
    html.Br(),
    html.Div(id='cracks_position'),
])


@app.callback(Output('output_dropdown', 'style'),
              Input('upload-image', 'contents'),
               State('upload-image', 'filename'))
def update_dropdown_visibility(list_of_contents, list_of_names):
    if list_of_names is None:
        return  {'display':'none'}
    else:
        return {'display':'block'}

@app.callback(Output('output_dropdown', 'options'),
              Input('upload-image', 'contents'),
               State('upload-image', 'filename'))
def update_dropdown_options(list_of_contents, list_of_names):
    if list_of_names is None:
        return [{'label': 1, 'value': 1}, {'label': 2, 'value': 2}]
    else:
        return [{'label': i, 'value': i} for i in list_of_names]
    
@app.callback(Output('output_dropdown', 'value'),
              Input('upload-image', 'contents'),
               State('upload-image', 'filename'))
def update_dropdown_value0(list_of_contents, list_of_names):
    if list_of_names is None:
        return  None
    else:
        return list_of_names[0]
    

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    Input('output_dropdown', 'value'))
def update_dropdown_image(list_of_contents, list_of_names, name):
    if list_of_names is None:
        return None
    else:
        idx = list_of_names.index(name)
        return html.Div([html.Img(src=list_of_contents[idx])])


@app.callback(
    Output('all_position_table', 'children'),
    Input('upload-position-data', 'contents'),
    State('upload-image', 'filename'))
def update_position_tbl(contents, filenames):
    if contents is None:
        return None
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        rslt_df = df[df['filename'].isin(filenames)] 
        
        return html.Div(
            dash_table.DataTable(
                # id='datatable-interactivity',
                # , "presentation": "markdown"
                columns=[
                {"id": name, "name": name}
                for name in rslt_df.columns
            ],
            # data=rslt_df,
            data=rslt_df.to_dict('records'),
            # markdown_options={"html": True},  # dangerous but works
            editable=True,
            filter_action="native",
            sort_action="native",
                # columns=[
                #     {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
                # ],
                
                
                # sort_mode="multi",
                # column_selectable="single",
                # row_selectable="multi",
                # row_deletable=True,
                # selected_columns=[],
                # selected_rows=[],
                # page_action="native",
                # page_current= 0,
                # page_size= 10,
            ))
    

@app.callback(
    Output('all_position', 'children'),
    Input('upload-position-data', 'contents'),
    State('upload-image', 'filename'))
def update_position_img(contents, filenames):
    if contents is None:
        return None
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        rslt_df = df[df['filename'].isin(filenames)] 
        fig = go.Figure(data=[go.Scatter(x=rslt_df['x'], 
                                         y=rslt_df['y'], 
                                         mode='markers',
                                         text=rslt_df['filename'])])
        fig.update_layout(title='Test Sample Distribution in the Pavement (x-y coordinates)',
                          xaxis=dict(title="Longitudinal of the Pavement"),
                          yaxis=dict(title="transversal of the Pavement") 
                          )
        return html.Div(dcc.Graph(figure=fig))







@app.callback(Output('results', 'children'),
              Input('upload-image', 'contents'),
               State('upload-image', 'filename'),
               Input('detection_button', 'n_clicks'))
def update_results(list_of_contents, list_of_names, n_clicks):
    if list_of_contents is None or n_clicks==0:
        return None
    else:
        
        data = []
        pred_res = []
        for i, raw_image in enumerate(list_of_contents):
            
            # raw_image = list_of_contents[0]
            base64_image = raw_image.split(',')[1]
            bytes_image = base64.b64decode(base64_image)
            np_image = np.frombuffer(bytes_image, dtype=np.uint8)
            img = cv2.imdecode(np_image, flags=1)
            pred = cnn.predict_image_file(img)
            if pred[0][0]>0.5:
                res = 'Non-Cracks'
            else:
                res = 'Cracks'
                # f"<img src={raw_image}>"
            # if raw_image[-1] == '=':
            #     raw_image = raw_image[:-1]
            data.append({'Index': i, 'filename': list_of_names[i], 'Image': f"<img src={raw_image[:-10]}>", 
                         'Detection_Results': res, 'Non-Crack Probability (%)': round(pred[0][0]*100,1), 
                         'Crack Probability (%)': round(pred[0][1]*100, 1)})
            
            pred_res.append({'Index': i, 'filename': list_of_names[i], 
                         'Detection_Results': res, 'Non-Crack Probability (%)': round(pred[0][0]*100,1), 
                         'Crack Probability (%)': round(pred[0][1]*100, 1)})
            
        pred_res = pd.DataFrame(pred_res)
        pred_res.to_csv('pred_res.csv')
        return html.Div(
            dash_table.DataTable(
                # id='datatable-interactivity',
                columns=[
                {"id": name, "name": name, "presentation": "markdown"}
                for name in data[0]
            ],
            data=data,
            markdown_options={"html": True},  # dangerous but works
            editable=True,
            filter_action="native",
            sort_action="native"
                # columns=[
                #     {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
                # ],
                # data=df.to_dict('records'),
                
                # editable=True,
                # filter_action="native",
                # sort_action="native",
                # sort_mode="multi",
                # column_selectable="single",
                # row_selectable="multi",
                # row_deletable=True,
                # selected_columns=[],
                # selected_rows=[],
                # page_action="native",
                # page_current= 0,
                # page_size= 10,
            ))

@app.callback(Output('cracks_position', 'children'),
              Input('upload-position-data', 'contents'),
              # Input('upload-image', 'contents'),
                State('upload-image', 'filename'),
               Input('crack_pos_button', 'n_clicks'))
def update_crack_pos(contents, filenames, n_clicks):
    if contents is None or n_clicks < 1:
        return None
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
        pos_df = df[df['filename'].isin(filenames)] 
        
        df = pd.read_csv('pred_res.csv')
        
        new_df = pd.merge(df, pos_df,on='filename',how='left')

        fig = px.scatter(new_df, x="x", y="y", color="Detection_Results",
        text=new_df['filename'],
                title="Cracks Distribution")
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            xaxis=dict(title="Longitudinal of the Pavement"),
            yaxis=dict(title="transversal of the Pavement") 
        )
        fig.update_xaxes(title_font_family="Arial")
        return html.Div(dcc.Graph(figure=fig))




if __name__ == '__main__':
    app.run_server(debug=True,port=3005)
