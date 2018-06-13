import base64
import io
import os
import time
import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


IMAGE_DATASETS = ('mnist_3000', 'cifar_gray_3000', 'fashion_3000')
WORD_EMBEDDINGS = ('wikipedia_3000', 'twitter_3000', 'crawler_3000')


def merge(a, b):
    return dict(a, **b)


def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64


def Card(children, **kwargs):
    return html.Section(
        children,
        style=merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid'
        }, kwargs.get('style', {})),
        **omit(['style'], kwargs)
    )


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={'margin': '25px 5px 30px 0px'},
        children=[
            f"{name}:",
            html.Div(style={'margin-left': '5px'}, children=[
                dcc.Slider(id=f'slider-{short}',
                           min=min,
                           max=max,
                           marks=marks,
                           step=step,
                           value=val)
            ])
        ])


def NamedInlineRadioItems(name, short, options, val):
    return html.Div(
        id=f'div-{short}',
        style={
            'display': 'inline-block'
        },
        children=[
            f'{name}:',
            dcc.RadioItems(
                id=f'radio-{short}',
                options=options,
                value=val,
                labelStyle={
                    'display': 'inline-block',
                    'margin-right': '7px'
                },
                style={
                    'display': 'inline-block',
                    'margin-left': '7px'
                }
            )
        ]
    )


demo_layout = html.Div(
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem',
        'padding': '10px 30px'
    },
    children=[
        # Header
        html.Div(className="row", children=[
            html.H2(
                't-SNE Explorer',
                id='title',
                style={
                    'float': 'left',
                    'margin-top': '20px',
                    'margin-bottom': '0',
                    'margin-left': '7px'
                }
            ),

            html.Img(
                src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
                style={
                    'height': '100px',
                    'float': 'right'
                }
            )
        ]),

        # Body
        html.Div(className="row", children=[
            html.Div(className="eight columns", children=[
                dcc.Graph(
                    id='graph-3d-plot-tsne',
                    style={'height': '95vh'}
                )
            ]),

            html.Div(className="four columns", children=[
                Card([
                    dcc.Dropdown(
                        id='dropdown-dataset',
                        searchable=False,
                        options=[
                            {'label': 'MNIST', 'value': 'mnist_3000'},
                            {'label': 'Fashion MNIST', 'value': 'fashion_3000'},
                            {'label': 'Wikipedia', 'value': 'wikipedia_3000'},
                            {'label': 'Web Crawler', 'value': 'crawler_3000'}
                        ],
                        placeholder="Select a dataset"
                    ),

                    NamedSlider(
                        name="Number of Iterations",
                        short="iterations",
                        min=250,
                        max=1000,
                        step=None,
                        val=500,
                        marks={i: i for i in [250, 500, 750, 1000]}
                    ),

                    NamedSlider(
                        name="Perplexity",
                        short="perplexity",
                        min=3,
                        max=100,
                        step=None,
                        val=30,
                        marks={i: i for i in [3, 10, 30, 50, 100]}
                    ),

                    NamedSlider(
                        name="Initial PCA Dimensions",
                        short="pca-dimension",
                        min=25,
                        max=100,
                        step=None,
                        val=50,
                        marks={i: i for i in [25, 50, 100]}
                    ),

                    NamedSlider(
                        name="Learning Rate",
                        short="learning-rate",
                        min=10,
                        max=200,
                        step=None,
                        val=100,
                        marks={i: i for i in [10, 50, 100, 200]}
                    )
                ]),

                Card([
                    html.Div(id='div-plot-hover-message'),

                    html.Img(
                        id='img-plot-hover-display',
                        style={'height': '20vh',
                               'display': 'block',
                               'margin': 'auto'}
                    ),
                ])
            ])
        ])
    ]
)


def demo_callbacks(app):
    def generate_figure_image(groups):
        data = []

        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val['x'],
                y=val['y'],
                z=val['z'],
                text=[idx for _ in range(val['x'].shape[0])],
                textposition='top',
                mode='markers',
                marker=dict(
                    size=2.5,
                    symbol='circle-dot'
                )
            )
            data.append(scatter)

        figure = go.Figure(
            data=data,
            layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        )

        return figure

    def generate_figure_word_vec(embedding_df):
        embedding_df = embedding_df[:1000]
        scatter = go.Scatter3d(
            name=embedding_df.index,
            x=embedding_df['x'],
            y=embedding_df['y'],
            z=embedding_df['z'],
            text=embedding_df.index,
            textposition='middle-center',
            showlegend=False,
            mode='text',
            marker=dict(
                size=2.5,
                symbol='circle-dot'
            )
        )

        figure = go.Figure(
            data=[scatter],
            layout=go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        )

        return figure

    @app.server.before_first_request
    def load_image_data():
        global data_dict, _figure

        data_dict = {
            'mnist_3000': pd.read_csv("data/mnist_3000_input.csv"),
            'fashion_3000': pd.read_csv("data/fashion_3000_input.csv"),
            'cifar_gray_3000': pd.read_csv("data/cifar_gray_3000_input.csv"),
            'wikipedia_3000': pd.read_csv('data/wikipedia_3000.csv'),
            'crawler_3000': pd.read_csv('data/crawler_3000.csv'),
            # 'twitter_3000': pd.read_csv('data/twitter_3000.csv')
        }


    @app.callback(Output('graph-3d-plot-tsne', 'figure'),
                  [Input('dropdown-dataset', 'value'),
                   Input('slider-iterations', 'value'),
                   Input('slider-perplexity', 'value'),
                   Input('slider-pca-dimension', 'value'),
                   Input('slider-learning-rate', 'value')])
    def generate_plot(dataset, iterations, perplexity, pca_dim, learning_rate):
        if dataset:
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv', index_col=0)

            except FileNotFoundError as error:
                print(error, "The dataset was not found. Please generate it using generate_demo_embeddings.py")
                return go.Figure()

            # For Image datasets
            if dataset in IMAGE_DATASETS:
                embedding_df['label'] = embedding_df.index

                groups = embedding_df.groupby('label')
                figure = generate_figure_image(groups)

            # Everything else is word embeddings
            else:
                figure = generate_figure_word_vec(embedding_df)

            return figure

    @app.callback(Output('img-plot-hover-display', 'src'),
                  [Input('graph-3d-plot-tsne', 'clickData')],
                  [State('dropdown-dataset', 'value'),
                   State('slider-iterations', 'value'),
                   State('slider-perplexity', 'value'),
                   State('slider-pca-dimension', 'value'),
                   State('slider-learning-rate', 'value')])
    def display_click_point_image(clickData,
                                  dataset,
                                  iterations,
                                  perplexity,
                                  pca_dim,
                                  learning_rate):
        if clickData and dataset in IMAGE_DATASETS:
            # Load the same dataset as the one displayed
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv')

            except FileNotFoundError as error:
                print(error, "The dataset was not found. Please generate it using generate_demo_embeddings.py")
                return

            # Convert the point hovered into float64 numpy array
            hover_point_np = np.array([clickData['points'][0][i] for i in ['x', 'y', 'z']]).astype(np.float64)
            # Create a boolean mask of the point hovered, truth value exists at only one row
            bool_mask_hover = embedding_df.loc[:, 'x':'z'].eq(hover_point_np).all(axis=1)
            # Retrieve the index of the point hovered
            hovered_idx = embedding_df[bool_mask_hover].index[0]

            # Retrieve the image corresponding to the index
            image_vector = data_dict[dataset].iloc[hovered_idx]
            if dataset == 'cifar_gray_3000':
                image_np = image_vector.values.reshape(32, 32).astype(np.float64)
            else:
                image_np = image_vector.values.reshape(28, 28).astype(np.float64)

            image_b64 = numpy_to_b64(image_np)

            return 'data:image/png;base64, ' + image_b64

        else:
            return

    @app.callback(Output('div-plot-hover-message', 'children'),
                  [Input('dropdown-dataset', 'value')])
    def display_hover_message(dataset):
        if dataset in IMAGE_DATASETS:
            return "Click a data point to display its image:"
        elif dataset in WORD_EMBEDDINGS:
            return
