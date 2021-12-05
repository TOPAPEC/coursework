class Suggest:
    def __init__(self, indices=None, labels=None, from_to="oracle", new_field_name=None):
        self.indices = indices
        self.labels = labels
        self.from_to = from_to
        self.new_field_name = new_field_name
