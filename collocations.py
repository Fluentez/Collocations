# python -m spacy download en_core_web_sm
# pip install spacy
# pip install collocater

from pprint import pprint

import spacy
from collocater import collocater

collie = collocater.Collocater.loader()
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(collie)

text = "If this isn't a bunch of beautiful flowers I don't know what is!"
doc = nlp(text)
print(
    doc._.collocs
)  # returns [bunch of, bunch of beautiful flowers, beautiful flowers]

# Tokens with associated collocations in text:
colls = [(col.text, col.start_char, col.end_char, col.label_) for col in doc._.collocs]
pprint(colls)  # returns [
#                          ('bunch of', 16, 24, 'bunch_noun__prep'),
#                          ('bunch of beautiful flowers', 16, 42, 'flower_noun__quant'),
#                          ('beautiful flowers', 25, 42, 'flower_noun__adj')
#                          ]

print(collie(text))
# {'beautiful flowers': {'coll_type': 'flower_noun__adj', 'location': [7, 9]},
# 'bunch of': {'coll_type': 'bunch_noun__prep', 'location': [5, 7]},
# 'bunch of beautiful flowers': {'coll_type': 'flower_noun__quant',
#                                'location': [5, 9]}}
