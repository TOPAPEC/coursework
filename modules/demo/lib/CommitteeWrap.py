from typing import List
from modAL.models import Committee


class CommitteeWrap:
    def models_info(self) -> List[str]:
        pass

    def get_committee(self) -> Committee:
        pass