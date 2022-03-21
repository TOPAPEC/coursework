from typing import List
from modAL import Committee, ActiveLearner
from modAL.disagreement import vote_entropy_sampling
from modules.models.Linear import LogReg
from modules.models.Wraps import TorchClassifierWrap
from modules.demo.lib.CommitteeWrap import CommitteeWrap
from sklearn.svm import LinearSVC


class CommitteeLoader(CommitteeWrap):
    @staticmethod
    def models_info(**kwargs) -> List[str]:
        return [
            "Logistic Regression",
            "Support Vector Machine"
        ]

    @staticmethod
    def get_committee(**kwargs) -> Committee:
        learners = [
            ActiveLearner(TorchClassifierWrap(LogReg(), 100, 300, 500, verbose=False)), ActiveLearner(LinearSVC())
        ]
        return Committee(
            learner_list=learners,
            query_strategy=vote_entropy_sampling
        )


def load_model():
    return TorchClassifierWrap(LogReg(), 100, 300, 500, verbose=False)
