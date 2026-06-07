import pandas as pd

# https://korpus.dsl.dk/resources/details/ddo-fullforms.html

def load_dictionary():

    # https://korpus.dsl.dk/resources/details/ddo-fullforms.html

    # load data
    df = pd.read_csv('all_words.csv',sep='\t')[['bøjningsform', 'ordklasse']]

    # only keep words with alphabetic characters (only standard form, so no é or ñ)
    df = df[df['bøjningsform'].str.match(r'^[a-zåæøA-ZÅÆØ]+$', na=False)]

    # only consider lowercase words
    df['bøjningsform'] = df['bøjningsform'].str.lower()

    # remove duplicates
    df = df.drop_duplicates(subset=['bøjningsform'])

    # remove following "ordklasse": 'fork.','egennavn','præfiks','symbol','formelt subjekt', 'infinitivpartikel
    df = df[~df['ordklasse'].isin(['fork.','egennavn','præfiks','symbol','formelt subjekt', 'infinitivpartikel'])]

    # remove words 15 or fewer characters or just 1 character long
    df = df[df['bøjningsform'].str.len() > 1]
    df = df[df['bøjningsform'].str.len() <= 15]

    df['bøjningsform'].to_list()

    words = df['bøjningsform'].to_list()
    word_dict = {}
    for w in words:
        key = str(len(w))
        word_dict.setdefault(key, []).append(w)
        
    return word_dict