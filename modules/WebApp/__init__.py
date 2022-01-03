import streamlit as st
import pickle
import seaborn as sns
import pandas as pd
import os

from collections import defaultdict

from matplotlib import pyplot as plt

st.title('Coursework NLP short text')
all_metrics = {}
with open(r"C:\Users\TOPAPEC\repos\coursework\metricdict.pkl", "rb") as file:
    all_metrics["0.001"] = pickle.load(file)
with open(r"C:\Users\TOPAPEC\repos\coursework\metricdict005.pkl", "rb") as file:
    all_metrics["0.005"] = pickle.load(file)
with open(r"C:\Users\TOPAPEC\repos\coursework\metricdict05.pkl", "rb") as file:
    all_metrics["0.05"] = pickle.load(file)
with open(r"C:\Users\TOPAPEC\repos\coursework\metricdict1.pkl", "rb") as file:
    all_metrics["0.1"] = pickle.load(file)

metrics_df = {}
for key, val in all_metrics.items():
    metric_dict = defaultdict(dict)
    for metric_name in next(iter(val.values())).keys():
        for method_name, method_metrics in val.items():
            metric_dict[metric_name][method_name] = method_metrics[metric_name]
        metric_dict[metric_name] = pd.DataFrame(metric_dict[metric_name])
    metrics_df[key] = metric_dict

st.subheader("Sampling strategies effeciency")
train_fractures_values = ["0.001", "0.005", "0.05", "0.1"]
train_fraction = st.select_slider("Select train fraction", options=train_fractures_values,
                                         value="0.001")
metrics = metrics_df[train_fraction]
fig, axs = plt.subplots(ncols=4, figsize=(30, 7))
fig.suptitle(f"Initial {train_fraction} of dataset in train")
for i, (metric, values) in enumerate(metrics.items()):
    axs[i].set_title(metric)
    st.write(sns.lineplot(data=values, ax=axs[i]))
