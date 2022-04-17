import pickle
import random
import logging
from functools import partial
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import umap
import umap.plot
import yaml
import importlib.util
from runpy import run_path
from faiss import IndexFlatL2
from modAL.models import ActiveLearner
from modAL.models.base import BaseEstimator
from modAL.uncertainty import entropy_sampling, uncertainty_sampling, margin_sampling
from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from scipy.stats import entropy
from modules.ActiveLearning.Samplings import random_sampling
from modules.ActiveLearning.Heuristics import disputable_points
from modules.models.Linear import LogReg
from modules.models.Wraps import TorchClassifierWrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xxhash import xxh128_hexdigest

st.set_page_config(layout="wide")

random_state = 42
np.random.seed(random_state)
random.seed(random_state)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Functions ------------------------------------------------------------------------------------------------------------

def app_init():
    dataset_types = ["train", "val", "pool"]
    is_labeled = [True, True, False]
    st.session_state["class_count"] = st.session_state["config"]["datasets"]["class_count"]
    for dataset_type, is_labeled in zip(dataset_types, is_labeled):
        with open(st.session_state["config"]["datasets"][f"{dataset_type}_path"], "rb") as dataset_file:
            dataset, X = pickle.load(dataset_file)
            st.session_state[f"dataset_{dataset_type}"] = dataset
            st.session_state[f"X_{dataset_type}"] = X
            if is_labeled:
                le = LabelEncoder()
                st.session_state[f"y_{dataset_type}"] = \
                    le.fit_transform(dataset[st.session_state["config"]["datasets"]["target"]])
                st.session_state[f"le_{dataset_type}"] = dict(zip(le.transform(le.classes_), le.classes_))


@st.cache
def class_mass_centers(X, y):
    _ndx = np.argsort(y)
    _id, _pos, g_count = np.unique(y[_ndx], return_index=True, return_counts=True)
    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count[:, None]
    return g_count, g_mean, np.unique(y[_ndx])


@st.cache
def umap_data(X, y, n_neighbors, min_dist, metric="cosine", verbose=False, cache_keyword="display_umaped_cache"):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric,
                        verbose=verbose)
    X_out = reducer.fit_transform(X)
    y_out = y
    return X_out, y_out


def plot_dataset(id_to_centers, id_to_counts, classes, le, custom_colors=None):
    classes = np.vectorize(le.get)(classes)
    print(classes.shape)
    dataframe = pd.DataFrame({
        "X": id_to_centers[:, 0],
        "Y": id_to_centers[:, 1],
        "count": id_to_counts,
        "class": classes,
        "color": classes if custom_colors is None else custom_colors
    })
    return alt.Chart(dataframe).mark_circle().encode(
        alt.X("X"),
        alt.Y("Y"),
        color=alt.Color("color", legend=None),
        size=alt.Size("count"),
        tooltip=[alt.Tooltip("class"), alt.Tooltip("count")]
    ).properties(
        width=1000,
        height=800
    ).interactive()


def cached_train(model, X, y, cache_keyword="model_train_cache", **model_kwargs):
    if (xxh128_hexdigest(X), xxh128_hexdigest(y)) == st.session_state.get(cache_keyword + "_hashes", ("", "")):
        return pickle.loads(st.session_state[cache_keyword])
    else:
        model.fit(X, y, **model_kwargs)
        st.session_state[cache_keyword] = pickle.dumps(model)
        st.session_state[cache_keyword + "_hashes"] = (xxh128_hexdigest(X), xxh128_hexdigest(y))
        return model


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


# Global vars ----------------------------------------------------------------------------------------------------------


cols = st.columns((2, 10, 4))
st.session_state["sampling_type_choice"] = cols[0].empty()
st.session_state["committee_info_textbox"] = cols[0].empty()
st.session_state["sampling_number_of_samples_choice"] = cols[0].empty()
st.session_state["sampling_result_dataframe_column_slider"] = cols[0].empty()
st.session_state["sampling_button"] = cols[0].empty()
st.session_state["sampling_skip_vis_checkbox"] = cols[0].empty()
st.session_state["Sampled dataset download"] = cols[0].empty()
st.session_state["sampling_result_dataframe_download"] = cols[0].empty()
st.session_state["sampling_result_dataframe_upload"] = cols[0].empty()
st.session_state["dataset_count_above_main_plot"] = cols[1].empty()
st.session_state["main_plot"] = cols[1].empty()
st.session_state["sampling_result_dataframe"] = st.empty()

with open("conf.yml", "r") as config_file:
    st.session_state["config"] = yaml.safe_load(config_file)

# Main -----------------------------------------------------------------------------------------------------------------

if "initialised" not in st.session_state:
    st.session_state["samplings"] = ["Confidence", "Margin", "Entropy",
                                     "Random", "Most Disputable points"]
    st.session_state["sampling_funcs"] = [uncertainty_sampling, margin_sampling, entropy_sampling, random_sampling,
                                          disputable_points]
    st.session_state["n_neighbours"] = 20
    st.session_state["min_dist"] = 0.8
    st.session_state["sampling_number_of_samples_variants"] = [500, 1000, 2000, 3000]
    st.session_state["sampling_max_number_of_samples_to_show"] = 500
    logging.basicConfig(level=logging.INFO)
    app_init()
    spec = importlib.util.spec_from_file_location("model", st.session_state["config"]["sampling_model"]["path"])
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    st.session_state["sampling_model"] = model_module.load_model()
    try:
        st.session_state["committee"] = model_module.CommitteeLoader.get_committee()
        st.session_state["committee_info"] = model_module.CommitteeLoader.models_info()
        st.session_state["samplings"].append("Committee")
    except:
        logging.info("Failed to load committee")
        st.session_state["committee"] = None
        st.session_state["committee_info"] = None
    st.session_state["initialised"] = True

st.session_state["dataset_count_above_main_plot"].write(
    f"Train dataset size is {st.session_state.X_train.shape[0]}, "
    f"val dataset size is {st.session_state.X_val.shape[0]}, "
    f"pool dataset size is {st.session_state.X_pool.shape[0]}")

st.session_state["current_sampling_choice"] = st.session_state["sampling_type_choice"].radio(
    "Choose sampling strategy:", st.session_state["samplings"])
number_to_sample = st.session_state["sampling_number_of_samples_choice"] \
    .select_slider("Select number of samples:", st.session_state["sampling_number_of_samples_variants"])
number_of_samples_to_show_in_df = st.session_state["sampling_result_dataframe_column_slider"] \
    .slider("Number of samples to show in table:", min_value=0,
            max_value=st.session_state["sampling_max_number_of_samples_to_show"], value=50)

with st.session_state["main_plot"]:
    X, y = umap_data(st.session_state["X_train"], st.session_state["y_train"], st.session_state["n_neighbours"],
                     st.session_state["min_dist"], verbose=False)
    id_to_counts, id_to_centers, classes = class_mass_centers(X, y)
    plot = plot_dataset(id_to_centers, id_to_counts, classes, st.session_state["le_train"])
    st.write(plot)

skip_vis = st.session_state["sampling_skip_vis_checkbox"].checkbox('Sample and add to dataset, skip visualisation')

if st.session_state["current_sampling_choice"] == "Committee":
    st.session_state["committee_info_textbox"].write(
        "Current committee members: \n {}".format(
            "\n".join("- {}".format(info) for info in st.session_state["committee_info"]))

    )

if st.session_state["sampling_button"].button("Sample 'em all!"):
    st.session_state["main_plot"].write("Training the model, sampling...")
    if st.session_state["current_sampling_choice"] == "Committee":
        learner = st.session_state["committee"]
    else:
        st.session_state["committee_info_textbox"].write("")
        learner = ActiveLearner(
            estimator=st.session_state["sampling_model"],
            query_strategy=st.session_state["sampling_funcs"][
                st.session_state["samplings"].index(st.session_state["current_sampling_choice"])]
        )
    learner._add_training_data(st.session_state["X_train"], st.session_state["y_train"])
    if st.session_state["current_sampling_choice"] == "Committee":
        for i, member in enumerate(learner.learner_list):
            member.estimator = cached_train(member.estimator, st.session_state["X_train"], st.session_state["y_train"],
                         cache_keyword=f"{str(i)}_committee_training")
    else:
        learner.estimator = cached_train(learner.estimator, st.session_state["X_train"], st.session_state["y_train"])

    query_idx, query_inst = learner.query(st.session_state["X_pool"], n_instances=number_to_sample)
    X_train_new = st.session_state["X_pool"][query_idx]
    y_train_new = np.full(query_idx.shape[0], len(st.session_state["le_train"]))
    if not skip_vis:
        X, y = umap_data(np.append(st.session_state["X_train"], X_train_new, axis=0),
                         np.append(st.session_state["y_train"], y_train_new, axis=0), st.session_state["n_neighbours"],
                         st.session_state["min_dist"], verbose=False, cache_keyword="display_umaped_cache_sampling")
        X_train_new, y_train_new = X[st.session_state["X_train"].shape[0]:], y[st.session_state["y_train"].shape[0]:]
        id_to_counts, id_to_centers, classes = class_mass_centers(X[:st.session_state["X_train"].shape[0]],
                                                                  y[:st.session_state["y_train"].shape[0]])
        centers_count = id_to_centers.shape[0]
        new_samples_count = y_train_new.shape[0]
        id_to_counts, id_to_centers, classes = \
            np.append(id_to_counts, np.ones(y_train_new.shape), axis=0), \
            np.append(id_to_centers, X_train_new, axis=0), \
            np.append(classes, y_train_new, axis=0)
        plot = plot_dataset(id_to_centers, id_to_counts, classes, st.session_state["le_train"],
                            custom_colors=np.append(np.zeros(centers_count), np.ones(new_samples_count)))
        with st.session_state["main_plot"]:
            st.write(plot)
        with st.session_state["sampling_result_dataframe"]:
            dataframe = st.session_state["dataset_pool"].iloc[
                        query_idx[np.random.choice(query_idx.shape[0], number_of_samples_to_show_in_df, replace=False)],
                        :]
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                   None, 'display.width', None, 'display.max_colwidth', -1):
                st.table(dataframe)
    else:
        st.session_state["sampling_result_dataframe_download"] \
            .download_button(label="Download sampled dataset",
                             data=convert_df(st.session_state["dataset_pool"].iloc[query_idx]),
                             file_name='tolabel.csv',
                             mime='text/csv')
    file = st.session_state["sampling_result_dataframe_upload"].file_uploader("Choose a file")
