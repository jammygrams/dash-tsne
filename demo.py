import base64
import io
import os
import time
import json
import pickle
import re

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go

from config import iterations_ls, perplexity_ls, pca_dim_ls, learning_rate_ls
DATASETS = ('doc2vec', 'tfidf', 'bert_250_word_mean')

with open('demo_description.md', 'r') as file:
    demo_md = file.read()


def merge(a, b):
    return dict(a, **b)


def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def Card(children, **kwargs):
    return html.Section(
        children,
        style=merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
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


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f'div-{short}',
        style=merge({
            'display': 'inline-block'
        }, kwargs.get('style', {})),
        children=[
            f'{name}:',
            dcc.RadioItems(
                id=f'radio-{short}',
                options=options,
                value=val,
                labelStyle={
                    'display': 'inline-block',
                    'margin-right': '7px',
                    'font-weight': 300
                },
                style={
                    'display': 'inline-block',
                    'margin-left': '7px'
                }
            )
        ],
        **omit(['style'], kwargs)
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
                    style={'height': '98vh'}
                )
            ]),

            html.Div(className="four columns", children=[
                Card([
                    dcc.Dropdown(
                        id='dropdown-dataset',
                        searchable=False,
                        options=[
                            # TODO: Generate more data
                            # {'label': 'MNIST Digits', 'value': 'mnist_3000'},
                            # {'label': 'Fashion MNIST', 'value': 'fashion_3000'},
                            # {'label': 'CIFAR 10 (Grayscale)', 'value': 'cifar_gray_3000'},
                            # {'label': 'Twitter (GloVe)', 'value': 'twitter_3000'},
                            {'label': 'BERT Word Mean (768 dimensions)', 'value': 'bert_250_word_mean'},
                            {'label': 'TFIDF (SVD 200 dimensions)', 'value': 'tfidf'},
                            {'label': 'Doc2Vec (200 dimensions)', 'value': 'doc2vec'},
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
                        marks={i: i for i in iterations_ls}
                    ),

                    NamedSlider(
                        name="Perplexity",
                        short="perplexity",
                        min=3,
                        max=100,
                        step=None,
                        val=30,
                        marks={i: i for i in perplexity_ls}
                    ),

                    NamedSlider(
                        name="Initial PCA Dimensions",
                        short="pca-dimension",
                        min=25,
                        max=200,
                        step=None,
                        val=100,
                        # If 'no_pca', set on slider at value 200
                        marks={(200 if i is 'none' else i): i for i in pca_dim_ls}
                    ),

                    NamedSlider(
                        name="Learning Rate",
                        short="learning-rate",
                        min=10,
                        max=200,
                        step=None,
                        val=100,
                        marks={i: i for i in learning_rate_ls}
                    ),

                    html.Div(id='div-wordemb-controls', style={'display': 'none'}, children=[
                        NamedInlineRadioItems(
                            name="Display Mode",
                            short="wordemb-display-mode",
                            options=[
                                {'label': ' Regular', 'value': 'regular'},
                                {'label': ' Top-100 Neighbors', 'value': 'neighbors'}
                            ],
                            val='regular'),

                        dcc.Dropdown(id='dropdown-word-selected', placeholder='Select word to display its neighbors')
                    ])
                ]),

                Card(style={'padding': '5px'}, children=[
                    html.Div(id='div-plot-click-message',
                             style={'text-align': 'center',
                                    'margin-bottom': '7px',
                                    'font-weight': 'bold'}
                             ),

                    html.Div(id='div-plot-click-image'),

                    html.Div(id='div-plot-click-wordemb')
                ])
            ])
        ]),

        # Demo Description
        html.Div(
            className='row',
            children=html.Div(
                style={
                    'width': '75%',
                    'margin': '30px auto',
                },
                children=dcc.Markdown(demo_md)
            )
        )
    ]
)


def demo_callbacks(app):
    def generate_figure_image(groups, layout):
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
                    size=3,
                    symbol='circle'
                )
            )
            data.append(scatter)

        figure = go.Figure(
            data=data,
            layout=layout
        )

        return figure


    @app.server.before_first_request
    def load_image_data():
        global data_dict

        data_dict = {
            # 'mnist_3000': pd.read_csv("data/mnist_3000_input.csv"),
            # 'fashion_3000': pd.read_csv("data/fashion_3000_input.csv"),
            # 'cifar_gray_3000': pd.read_csv("data/cifar_gray_3000_input.csv"),
            # 'wikipedia_3000': pd.read_csv('data/wikipedia_3000.csv'),
            # 'crawler_3000': pd.read_csv('data/crawler_3000.csv'),
            # 'twitter_3000': pd.read_csv('data/twitter_3000.csv', encoding="ISO-8859-1"),
            # TODO: clean up below
            'tfidf': pd.read_pickle('data/source_text.pkl'),
            'doc2vec': pd.read_pickle('data/source_text.pkl'),
            'bert_250_word_mean': pd.read_pickle('data/source_text.pkl')
        }


    @app.callback(Output('graph-3d-plot-tsne', 'figure'),
                  [Input('dropdown-dataset', 'value'),
                   Input('slider-iterations', 'value'),
                   Input('slider-perplexity', 'value'),
                   Input('slider-pca-dimension', 'value'),
                   Input('slider-learning-rate', 'value'),
                   Input('dropdown-word-selected', 'value'),
                   Input('radio-wordemb-display-mode', 'value')])
    def display_3d_scatter_plot(dataset, iterations, perplexity, pca_dim, learning_rate, selected_word, wordemb_display_mode):
        if dataset:
            # no_pca value is set as 200 above TODO: clean up?
            if pca_dim == 200:
                pca_dim = 'none'
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv', index_col=0, encoding="ISO-8859-1")

            except FileNotFoundError as error:
                print(error, "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py")
                return go.Figure()

            # Plot layout
            axes = dict(
                title='',
                showgrid=True,
                zeroline=False,
                showticklabels=False
            )

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=axes,
                    yaxis=axes,
                    zaxis=axes
                )
            )

            # For Image datasets
            if dataset in DATASETS:
                embedding_df['label'] = embedding_df.index

                groups = embedding_df.groupby('label')
                figure = generate_figure_image(groups, layout)

            else:
                figure = go.Figure()

            return figure


    @app.callback(Output('div-plot-click-image', 'children'),
                  [Input('graph-3d-plot-tsne', 'clickData'),
                   Input('dropdown-dataset', 'value'),
                   Input('slider-iterations', 'value'),
                   Input('slider-perplexity', 'value'),
                   Input('slider-pca-dimension', 'value'),
                   Input('slider-learning-rate', 'value')])
    def display_click_image(clickData,
                            dataset,
                            iterations,
                            perplexity,
                            pca_dim,
                            learning_rate):
        if clickData:
            # no pca value is set as 200 above TODO: clean up?
            if pca_dim == 200:
                pca_dim = 'none'
            # Load the same dataset as the one displayed
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv', encoding="ISO-8859-1")

            except FileNotFoundError as error:
                print(error, "\nThe dataset was not found. Please generate it using generate_demo_embeddings.py")
                return

            # Convert the point clicked into float64 numpy array
            click_point_np = np.array([clickData['points'][0][i] for i in ['x', 'y', 'z']]).astype(np.float64)
            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = embedding_df.loc[:, 'x':'z'].eq(click_point_np).all(axis=1)
            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():
                clicked_idx = embedding_df[bool_mask_click].index[0]

                # Retrieve text corresponding to index
                text = data_dict[dataset].iloc[clicked_idx].values[0]

                return dcc.Markdown(text)

        return None
        

    @app.callback(Output('div-plot-click-message', 'children'),
                  [Input('graph-3d-plot-tsne', 'clickData'),
                   Input('dropdown-dataset', 'value')])
    def display_click_message(clickData, dataset):
        """
        Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        :param clickData:
        :param dataset:
        :return:
        """
        if clickData:
            return "Text Selected"
        else:
            return "Click a data point on the scatter plot to display its corresponding text."

