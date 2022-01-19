import pickle
import random
from functools import partial

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import umap
import umap.plot
from sklearn.linear_model import LogisticRegression
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
pd.set_option('display.max_colwidth', -1)


# Functions ------------------------------------------------------------------------------------------------------------

def import_dataset_sampled(num_samples: int = 100000):
    dataset = pd.read_csv("selfpost/rspct.tsv", sep="\t")
    le = LabelEncoder()
    vectorized_output_path = "selfpost/vectorized.npy"
    vectorized_labels_output_path = "selfpost/vectorized_labels.npy"
    with open(vectorized_output_path, "rb") as vect_X, open(vectorized_labels_output_path, "rb") as vect_y:
        X = np.load(vect_X, allow_pickle=True)
        y = np.load(vect_y, allow_pickle=True)
    assert (np.array(dataset["subreddit"][:100]) == y[:100]).all()
    y = le.fit_transform(y)
    sample_indices = np.random.choice(X.shape[0], num_samples, replace=False)
    X = X[sample_indices]
    y = y[sample_indices]
    return X, y, dataset.iloc[sample_indices, :], dict(zip(le.transform(le.classes_), le.classes_)), sample_indices


def class_mass_centers(X, y):
    _ndx = np.argsort(y)
    _id, _pos, g_count = np.unique(y[_ndx], return_index=True, return_counts=True)
    g_sum = np.add.reduceat(X[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count[:, None]
    return g_count, g_mean, np.unique(y[_ndx])

@st.cache
def umap_data(X, y, n_neighbors, min_dist, metric="cosine", verbose=False, cache_keyword="display_umaped_cache"):
    # X_curhash, y_curhash = xxh128_hexdigest(X), xxh128_hexdigest(y)
    #     # if f"{cache_keyword}_hashes" in st.session_state and (X_curhash, y_curhash) == st.session_state[
    #     #     f"{cache_keyword}_hashes"]:
    #     #     (X_out, y_out) = st.session_state[f"{cache_keyword}"]
    #     # else:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric,
                        verbose=verbose)
    X_out = reducer.fit_transform(X)
    y_out = y
    # st.session_state[f"{cache_keyword}"] = (X_out, y_out)
    # st.session_state[f"{cache_keyword}_hashes"] = X_curhash, y_curhash
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
        model.set_state(st.session_state[cache_keyword])
    else:
        model.fit(X, y, **model_kwargs)
        st.session_state[cache_keyword] = model.get_state()
        st.session_state[cache_keyword + "_hashes"] = (xxh128_hexdigest(X), xxh128_hexdigest(y))


# Global vars ----------------------------------------------------------------------------------------------------------

samplings = ["Confidence", "Margin", "Entropy",
             "Random", "Most Disputable points"]
sampling_funcs = [uncertainty_sampling, margin_sampling, entropy_sampling, random_sampling, disputable_points]
n_neighbours = 20
min_dist = 0.8
sampling_number_of_samples_variants = [500, 1000, 2000, 3000]
sampling_max_number_of_samples_to_show = 500

cols = st.columns((2, 10, 4))
st.session_state["dataset_count_above_main_plot"] = cols[1].empty()
st.session_state["main_plot"] = cols[1].empty()
# st.session_state["sampling_result_dataframe_column_title"] = cols[2].empty()
st.session_state["sampling_result_dataframe_column_slider"] = st.empty()
st.session_state["sampling_result_dataframe"] = st.empty()
with st.session_state["sampling_result_dataframe_column_slider"]:
    number_of_samples_to_show_in_df = st.slider("Number of samples to show:",
                                                min_value=0, max_value=sampling_max_number_of_samples_to_show,
                                                value=50)


# Main -----------------------------------------------------------------------------------------------------------------

if "dataset" not in st.session_state:
    X, y, dataset, le, sample_indices = import_dataset_sampled()
    train_indices = np.random.choice(X.shape[0], 20000, replace=False)
    le[len(le)] = "unk"
    st.session_state["dataset"] = X, y, dataset, le, train_indices
else:
    X, y, dataset, le, train_indices = st.session_state["dataset"]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = np.delete(X, train_indices, axis=0), np.delete(y, train_indices, axis=0)
st.session_state["dataset_count_above_main_plot"].write(f"Train dataset size is {X_train.shape[0]}, test dataset size is {X_test.shape[0]}")

with cols[0]:
    sampling_choice = st.radio("Choose sampling strategy:",
                               samplings)
    st.write(f"Choice: {sampling_choice}")
    number_to_sample = st.select_slider("Select number of samples:", sampling_number_of_samples_variants)

with st.session_state["main_plot"]:
    X, y = umap_data(X_train, y_train, n_neighbours, min_dist, verbose=False)
    id_to_counts, id_to_centers, classes = class_mass_centers(X, y)
    plot = plot_dataset(id_to_centers, id_to_counts, classes, le)
    st.write(plot)

skip_vis = cols[0].checkbox('Sample and add to dataset, skip visualisation')

if cols[0].button("Sample it!"):

    learner = ActiveLearner(
        estimator=TorchClassifierWrap(LogReg(), 100, 300, 500, verbose=False),
        query_strategy=sampling_funcs[samplings.index(sampling_choice)]
    )
    learner._add_training_data(X_train, y_train)
    cached_train(learner.estimator, X_train, y_train)
    query_idx, query_inst = learner.query(X_test, n_instances=number_to_sample)
    X_train_new = X_test[query_idx]
    # print(f"Queryidx{query_idx.shape[0]}")
    y_train_new = np.full(query_idx.shape[0], len(le))
    if not skip_vis:
        X, y = umap_data(np.append(X_train, X_train_new, axis=0), np.append(y_train, y_train_new, axis=0), n_neighbours, min_dist,
                         verbose=False, cache_keyword="display_umaped_cache_sampling")
        X_train_new, y_train_new = X[X_train.shape[0]:], y[y_train.shape[0]:]
        id_to_counts, id_to_centers, classes = class_mass_centers(X[:X_train.shape[0]], y[:y_train.shape[0]])
        centers_count = id_to_centers.shape[0]
        new_samples_count = y_train_new.shape[0]
        # print(id_to_counts.shape, np.ones(y_train_new.shape[0]).shape)
        # print(id_to_centers.shape, X_train_new.shape)
        id_to_counts, id_to_centers, classes = \
            np.append(id_to_counts, np.ones(y_train_new.shape), axis=0), \
            np.append(id_to_centers, X_train_new, axis=0), \
            np.append(classes, y_train_new, axis=0)
        plot = plot_dataset(id_to_centers, id_to_counts, classes, le,
                            custom_colors=np.append(np.zeros(centers_count), np.ones(new_samples_count)))
        with st.session_state["main_plot"]:
            st.write("FUCK YOU NVIDIA")
            st.write(plot)
        with st.session_state["sampling_result_dataframe"]:

            dataset_sub = dataset[~dataset.index.isin(train_indices)]
            dataframe = dataset_sub.iloc[
                query_idx[np.random.choice(query_idx.shape[0], number_of_samples_to_show_in_df, replace=False)], :]
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                   None, 'display.width', None, 'display.max_colwidth', -1):
                st.table(dataframe)
    else:
        pass


