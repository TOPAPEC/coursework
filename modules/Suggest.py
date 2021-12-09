class Suggest:
    """
    from_to:
        oracle - test data should be labeled according to preexisting labels.
        labeling - test data should be labeled according to generated labels.
        relabeling - relabel data in train according to generated labels.
    """
    def __init__(self, indices=None, labels=None, from_to="oracle", new_field_name=None):
        self.indices = indices
        self.labels = labels
        self.from_to = from_to
        self.new_field_name = new_field_name
