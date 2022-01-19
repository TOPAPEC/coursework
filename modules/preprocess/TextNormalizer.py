import spacy
import sentencepiece as spm
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_words(text):
    # Text input is string, returns lowercased strings.
    return [token.lemma_ for token in nlp(text)]

def normalize_text(text):
    return " ".join([word for word in lemmatize_words(sp.decode_ids(sp.tokenize(text)))])
