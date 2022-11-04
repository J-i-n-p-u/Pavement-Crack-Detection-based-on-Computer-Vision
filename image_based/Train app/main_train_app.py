# Run this app with `python app.py` and
# visit http://127.0.0.1:3001/ in your web browser.

from dash import dcc, html, Input, Output, dash_table

import dash
# import numpy as np
import glob
import os
import base64
import pandas as pd
from dash.long_callback import DiskcacheLongCallbackManager
## Diskcache
import diskcache

from split_dataset import split_data
import evaluate_model
import train_model

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)


list_of_images_p = []
list_of_images_n = []
list_of_images_test_p = []
list_of_images_test_n = []

image_directory_p = r'cracks_splitted8020/train_set/Positive/'
image_directory_n = r'cracks_splitted8020/train_set/Negative/'
image_directory_test_p = r'cracks_splitted8020/test_set/Positive/'
image_directory_test_n = r'cracks_splitted8020/test_set/Negative/'

MODEL_NAME="simplenet_cracks8020_i"

app = dash.Dash(__name__, long_callback_manager=long_callback_manager)

app.layout = html.Div([html.H2("Split Data into Train and Test Sets"), 
                       html.Div(children=[html.Button('Generate Train and Test Data from Folder', id='generate_tt', n_clicks=0),
                                          html.Div(id = 'generating_label'),
                                          html.Div(id = 'split_finish_label'),
                                          html.Button('Check the Train and Test Data', 
                                                      id='check_train_test_bt', 
                                                      n_clicks=0)]),
                       html.Div([
                           html.Div(children=[
                               html.H3("Train Data", id='train_data_label'),
                               html.Br(),
                               html.Div(id='num_train_cracks_label'),

                               dcc.Dropdown(id='image-dropdown_p'),
                               html.Div(id='image_p'),
                               html.Br(),
                               html.Div(id='num_train_non_cracks_label'),

                               dcc.Dropdown(id='image-dropdown_n'),
                               html.Div(id='image_n'),
                               html.Br(),
                           ], 
                               style={'padding': 10, 'flex': 1, 'width': '28%', 'display': 'inline-block'}),
                           
                           html.Div(children=[
                               html.H3("Test Data",  id='test_data_label'),
                               html.Br(),
                               html.Div(id='num_test_cracks_label'),


                               dcc.Dropdown(id='image-dropdown_test_p'),
                               html.Div(id='image_test_p'),
                               html.Br(),
                               html.Div(id='num_test_non_cracks_label'),

                               dcc.Dropdown(id='image-dropdown_test_n',),
                               html.Div(id='image_test_n'),
                               html.Br(),
                           ], 
                           style={'padding': 10, 'flex': 1, 'width': '28%', 'display': 'inline-block'})
                       ], 
                           style={'display': 'flex', 'flex-direction': 'row'}),
                       
                       html.H2("Train and Evaluate the Model"), 
                       html.Div(id='Train Model label', children='Input the number of traning epoch:'),
                       html.Div([dcc.Input(id='number_training_epoch', 
                                           value='1', 
                                           type='text',
                                          style={'width':'30%'})]),
                       html.Div([html.Button('Train Model', id='train_button', n_clicks=0),
                                 # html.Button('Cancel Training', id='cancel_train_button', n_clicks=0),
                                 ]),
                       
                       html.Div(id='Train Model text',
                            children='Click the button to train the model'),
                       html.Div(id='Train_model_done_label',),
                       html.Br(),
                       
                       html.Div(id='Evaluate Model label', children='Input the Number of Test Samples'),
                       html.Div([dcc.Input(id='number_test_sample', 
                                           value='20', 
                                           type='text',
                                          style={'width':'30%'})],
                                # style={'display':'table-cell','padding':5, 'verticalAlign':'middle'}
                                ),
                       html.Button('Evaluate Model in the test set', id='test_button', n_clicks=0),
                       html.Br(),
                       
                       html.Div(id='Test model text',
                              children='Click the button to evaluate the model'),
                       html.Button('Check evaluate result', id='check_test_res_button', n_clicks=0),
                       
                       
                       html.H2("Evaluation Results"),
                       html.Div(id='evaluate_done'),
                       
                        html.Div(children = [html.Div(id='model_plot',
                                                      style={'padding': 10, 'flex': 1, 'width': '30%', 'height':'30%',
                                                             'display': 'inline-block'}), 
                                             html.Div(id='train_history_plot',
                                                      style={'padding': 10, 'flex': 1, 'width': '40%', 'display': 'inline-block'}), 
                                             html.Div(id='confusion_matrix_plot', 
                                                      style={'padding': 10, 'flex': 1, 'width': '30%', 'display': 'inline-block'})],
                                 style={'display': 'flex', 'flex-direction': 'row'}
                                 # style={'padding': 10, 'flex': 1, 'width': '28%', 'display': 'inline-block'}
                                 ), 
                       
                       html.Div(id='evaluate_result_table'),
                       
                       ])

@app.callback(Output('generating_label', 'children'),
              Input('generate_tt', 'n_clicks'))
def update_generating_label(n_clicks):
    if n_clicks < 1:
        return  None
    else:
        return 'Spliting the dataset into train set (80%) and test set (20%)...'

@app.long_callback(output=[ Output('split_finish_label', 'children'),
                           Output('check_train_test_bt', 'style')],
                   inputs= Input('generate_tt', 'n_clicks'),
                   # manager=long_callback_manager,
                   )
def update_split_finish_label(n_clicks):
    if n_clicks < 1:
        return  None, {'display':'none'}
    else:
        dataset_dir = 'dataset/'
        train_ratio = 80
        test_ratio = 20
        output_dir_name = 'cracks_splitted8020'
        split_data(dataset_dir, train_ratio, test_ratio, output_dir_name)
        return 'Done!', {'display':'block'}

@app.callback(Output('train_data_label', 'style'),
              Output('num_train_cracks_label', 'children'),
              Output('num_train_non_cracks_label', 'children'),
              
              Output('test_data_label', 'style'),
              Output('num_test_cracks_label', 'children'),
              Output('num_test_non_cracks_label', 'children'),
              
              Input('check_train_test_bt', 'n_clicks'))
def update_train_test_vis(n_clicks):
    if n_clicks < 1:
        return  {'display':'none'}, None, None, {'display':'none'}, None, None
    else:
        list_of_images_p = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_p))]
        list_of_images_n = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_n))]
        list_of_images_test_p = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_p))]
        list_of_images_test_n = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_n))]
        # html.Label('Cracks: ' + str(len(list_of_images_p)), id='num_train_cracks_label'),
        # html.Label('Non-Cracks: '+str(len(list_of_images_n)), id='num_train_non_cracks_label'),
        # html.Label('Cracks: ' + str(len(list_of_images_test_p)), id='num_test_cracks_label'),
        # html.Label('Non-Cracks: '+str(len(list_of_images_test_n)), id='num_test_non_cracks_label'),
        return {'display':'block'}, 'Cracks: ' + str(len(list_of_images_p)), 'Non-Cracks: '+str(len(list_of_images_n)), \
            {'display':'block'}, 'Cracks: ' + str(len(list_of_images_test_p)), 'Non-Cracks: '+str(len(list_of_images_test_n))


@app.callback(Output('image-dropdown_p', 'options'),
              Output('image-dropdown_p', 'value'),
              Output('image-dropdown_p', 'style'),
              
              Output('image-dropdown_n', 'options'),
              Output('image-dropdown_n', 'value'),
              Output('image-dropdown_n', 'style'),
              
              Output('image-dropdown_test_p', 'options'),
              Output('image-dropdown_test_p', 'value'),
              Output('image-dropdown_test_p', 'style'),
            
              Output('image-dropdown_test_n', 'options'),
              Output('image-dropdown_test_n', 'value'),
              Output('image-dropdown_test_n', 'style'),
              
              Input('check_train_test_bt', 'n_clicks'))
def update_dropdown(n_clicks):
    if n_clicks < 1:
        return  [], [], {'display':'none'}, \
            [], [], {'display':'none'}, \
                [], [], {'display':'none'}, \
                    [], [], {'display':'none'},
    else:
        list_of_images_p = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_p))]
        list_of_images_n = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_n))]
        list_of_images_test_p = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_p))]
        list_of_images_test_n = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory_test_n))]

        return [{'label': i, 'value': i} for i in list_of_images_p], list_of_images_p[0], {'display':'block'}, \
            [{'label': i, 'value': i} for i in list_of_images_n], list_of_images_n[0], {'display':'block'}, \
                [{'label': i, 'value': i} for i in list_of_images_test_p], list_of_images_test_p[0], {'display':'block'}, \
                    [{'label': i, 'value': i} for i in list_of_images_test_n], list_of_images_test_n[0], {'display':'block'}


@app.callback(
    Output('image_p', 'children'),
    Input('image-dropdown_p', 'value'))
def update_image_src_p(image_path):
    if image_path == [] or image_path is None:
        return None
    else:
        image_path = image_directory_p+image_path
        encoded_image = base64.b64encode(open(image_path, 'rb').read())
        return html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()))])
    
@app.callback(
    Output('image_n', 'children'),
    Input('image-dropdown_n', 'value'))
def update_image_src_n(image_path):
    if image_path == [] or image_path is None:
        return None
    else:
        image_path = image_directory_n+image_path
        encoded_image = base64.b64encode(open(image_path, 'rb').read())
        return html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()))])


@app.callback(
    Output('image_test_p', 'children'),
    Input('image-dropdown_test_p', 'value'))
def update_image_src_test_p(image_path):
    if image_path == [] or image_path is None:
        return None
    else:
        image_path = image_directory_test_p+image_path
        encoded_image = base64.b64encode(open(image_path, 'rb').read())
        return html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()))])
    
@app.callback(
    Output('image_test_n', 'children'),
    Input('image-dropdown_test_n', 'value'))
def update_image_src_test_n(image_path):
    if image_path == [] or image_path is None:
        return None
    else:
        image_path = image_directory_test_n+image_path
        encoded_image = base64.b64encode(open(image_path, 'rb').read())
        return html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image.decode()))])


@app.callback(
    Output('Train Model text', 'children'),
    Input('train_button', 'n_clicks'),
)
def update_output_train(n_clicks):
    if n_clicks>=1:
        return 'the button has been clicked {} times, the model is training...'.format(
            n_clicks
        )

@app.long_callback(
    output=Output("Train_model_done_label", "children"),
    inputs=[Input("train_button", "n_clicks"),
            Input("number_training_epoch", "value")],
)
def update_train_done_label(n_clicks, number_of_epochs):
    if n_clicks>=1:
        
        if not os.path.exists('model-checkpoints/'):
            os.mkdir('model-checkpoints/')
        if not os.path.exists('training_logs/'):
            os.mkdir('training_logs/')
        if not os.path.exists('tensorboard_logs/'):
            os.mkdir('tensorboard_logs/')
            os.mkdir('tensorboard_logs/train/')
            os.mkdir('tensorboard_logs/validation/')
            
        model = train_model.build_simplenet()
        train_model.plot_model(model, to_file=f'{MODEL_NAME}.jpg', show_shapes=True)
        train_model.train_simplenet(model,
                target_size=(64,64),
                dataset_path="cracks_splitted8020/",
                training_path_prefix="train_set",
                test_path_prefix="test_set",
                history_file_path="training_logs/",
                history_filename=MODEL_NAME+".csv",
                checkpoint_path="model-checkpoints/",
                checkpoint_prefix=MODEL_NAME,
                number_of_epochs=int(number_of_epochs), 
                tensorboard_log_path="tensorboard_logs/",
                batch_size = 256
                )
        
        fig = train_model.plot_learning_curves_from_history_file("training_logs/"+MODEL_NAME+".csv")
        fig.savefig('learning_curves.jpg')
        return 'Model training completed!'

@app.callback(
    Output('Test model text', 'children'),
    Input('test_button', 'n_clicks'),
)
def update_output_eval_text(n_clicks):
    if n_clicks>=1:
        return 'the button has been clicked {} times, the model is evaluating...'.format(
            n_clicks
        )

@app.long_callback(
    output=Output("evaluate_done", "children"),
    inputs=[Input("test_button", "n_clicks"),
            Input("number_test_sample", "value")],
)
def evaluate(n_clicks, value):
    if n_clicks>=1:
        eval_num = [int(value), int(value)]
        Input_shape = (64, 64)
        random_seed = 1
        list_of_files = glob.glob('model-checkpoints/*.hdf5') 
        CHECKPOINT_FILE = max(list_of_files, key=os.path.getctime) # last checkpoint
        
        df= evaluate_model.eval_res(CHECKPOINT_FILE, Input_shape, eval_num, image_directory_test_p, image_directory_test_n, random_seed)
        df.to_csv(f'{eval_num[0]}_{eval_num[1]}_eval_table.csv')
        score = evaluate_model.plot_cm(df['True Label'], df['Predicted Label'], f'{eval_num[0]}_{eval_num[1]}_cm.jpg')
        
        return f'Evaluation done: F1-score: {score}'


@app.callback(
    Output('model_plot', 'children'),
    Output('train_history_plot', 'children'),
    Output('confusion_matrix_plot', 'children'),
    Input('check_test_res_button', 'n_clicks'),
    Input('number_test_sample', 'value')
)
def update_output_eval_cm(n_clicks, value):
    if n_clicks>=1:
        eval_num = [int(value), int(value)]
        image_path = f'{eval_num[0]}_{eval_num[1]}_cm.jpg'
        if not os.path.exists(image_path):
            image_path = 'loading.jpg'
        encoded_image_cm = base64.b64encode(open(image_path, 'rb').read())
        
        image_path = f'{MODEL_NAME}.jpg'
        encoded_image_model = base64.b64encode(open(image_path, 'rb').read())
        
        image_path = 'learning_curves.jpg'
        encoded_image_curve = base64.b64encode(open(image_path, 'rb').read())
        
        return html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_model.decode()),
                                  style={'height':'20%', 'width':'50%'})]), html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_curve.decode()),
                                      style={'height':'100%', 'width':'100%'})]), html.Div([html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_cm.decode()),
                                  style={'height':'80%', 'width':'80%'})])
    else:
        return None, None, None
    
    
@app.callback(
    Output('evaluate_result_table', 'children'),
    Input('check_test_res_button', 'n_clicks'),
    Input('number_test_sample', 'value')
)
def update_output_eval_table(n_clicks, value):
    if n_clicks>=1:
        eval_num = [int(value), int(value)]
        
        df = pd.read_csv(f'{eval_num[0]}_{eval_num[1]}_eval_table.csv', index_col=0)
        
        data = []
        for i, img_path in enumerate(df['Fig_path']):
            encoded_image = base64.b64encode(open(df['Fig_path'].iloc[i], 'rb').read())
            base64_img = 'data:image/jpg;base64,{}'.format(encoded_image.decode())
            
            data.append({'Index': i, 'filename': df['Fig_name'].iloc[i],
                         'Image': f"<img src={base64_img[:-10]}>", 
                         'True Label': 'Cracks' if df['True Label'].iloc[i]==1 else 'Non Cracks', 
                         'Predicted Label': 'Cracks' if df['Predicted Label'].iloc[i]==1 else 'Non Cracks', 
                         'Correct or not': 'Yes' if df['True Label'].iloc[i] == df['Predicted Label'].iloc[i] else 'No',
                         'Non-Crack Probability (%)': round(df['Non_Cracks Confidence'].iloc[i]*100,1), 
                         'Crack Probability (%)': round(df['Cracks Confidence'].iloc[i]*100,1), })
        
        return html.Div(
            dash_table.DataTable(
                id='datatable-interactivity',
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'lineHeight': '15px'
                    },
                columns=[
                {"id": name, "name": name, "presentation": "markdown"}
                for name in data[0]
                ],
                data = data,
                markdown_options={"html": True},  # dangerous but works
                editable=True,
                filter_action="native",
                sort_action="native",
                fixed_rows={'headers': True},
                # page_size=10,
                page_size=10,  # we have less data in this example, so setting to 20
                style_table={'height': '600px', 'overflowY': 'auto'}
            ))



if __name__ == '__main__':
    app.run_server(debug=True, port=3001)