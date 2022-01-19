import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import altair as alt

from whatlies.transformers import Umap
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")
mpl.style.use('fivethirtyeight')

st.title('Coursework NLP short text')
all_metrics = {}
with open(r"/tmp/metric_dumps/benchmarking_samplings_logreg_metrics_ready.pkl", "rb") as file:
    all_metrics = pickle.load(file)


st.subheader("Dataset overview")
st.write("Here you can see first 20 classes of dataset subsampled and projected on 2d plane using umap")
with open(r"/tmp/dataset_dumps/umaped_dataset_first_20_classes_dataframe.pkl", "rb") as file:
    dataset_subsample = pickle.load(file)
cols = st.columns((1, 14, 1))
cols[0].write(" ")
cols[1].write(alt.Chart(dataset_subsample).mark_point().encode(
        alt.X("X"),
        alt.Y("Y"),
        alt.Color("subreddit"),
        tooltip= [alt.Tooltip("subreddit"),
        alt.Tooltip("title"),
        alt.Tooltip("selftext")]
    ).properties(
        width=1600,
        height=800
    ).interactive())


st.subheader("Sampling strategies effeciency")
train_fractures_values = ["0.001", "0.005", "0.05", "0.1"]
train_fraction = st.select_slider("Select train fraction", options=train_fractures_values,
                                         value="0.001")
metrics = all_metrics[train_fraction]
fig, axs = plt.subplots(ncols=4, figsize=(30, 7))
fig.suptitle(f"Initial {train_fraction} of dataset in train")
for i, (metric, values) in enumerate(metrics.items()):
    axs[i].set_title(metric)
    values.plot(ax=axs[i])
st.pyplot(fig)


