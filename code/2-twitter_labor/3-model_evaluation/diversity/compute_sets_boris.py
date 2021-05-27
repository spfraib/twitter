import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import os
import re
import argparse

def regex_match_string(ngram_list, regex_list, mystring):
    if any(regex.search(mystring) for regex in regex_list):
        return 1
    elif any(regex in mystring for regex in ngram_list):
        return 1
    else:
        return 0

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_code", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args_from_command_line()

    path_data = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/random_samples/random_samples_splitted/{args.country_code}/evaluation'
    output_path = f'/scratch/mt4493/twitter_labor/twitter-labor-data/data/keyword_model/sample_diversity/{args.country_code}'
    combined_list_dict = {
        'MX': ['me despidieron', 'perd[i|í] mi trabajo',
               'me corrieron', 'me qued[e|é] sin (trabajo|chamba|empleo)',
               'ya no tengo (trabajo|empleo|chamba)',
               'consegui[\w\s\d]*empleo', 'nuevo trabajo',
               'nueva chamba', 'encontr[e|é][.\w\s\d]*trabajo',
               'empiezo[\w\s\d]*trabajar', 'primer d[i|í]a de trabajo',
               'estoy desempleado', 'sin empleo', 'sin chamba', 'nini', 'no tengo (trabajo|empleo|chamba)',
               'necesito[\w\s\d]*trabajo', 'busco[\w\s\d]*trabajo', 'buscando[\w\s\d]*trabajo',
               'alguien[\w\s\d]*trabajo', 'necesito[\w\s\d]*empleo',
               'empleo', 'contratando', 'empleo nuevo', 'vacante', 'estamos contratando'],
        'BR': ['perdi[.\w\s\d]*emprego', 'perdi[.\w\s\d]*trampo',
               'fui demitido', 'me demitiram',
               'me mandaram embora', 'consegui[.\w\s\d]*emprego', 'fui contratad[o|a]',
               'começo[.\w\s\d]*emprego', 'novo emprego|emprego novo',
               'primeiro dia de trabalho', 'novo (emprego|trampo)',
               'estou desempregad[a|o]', 'eu[.\w\s\d]*sem[.\w\s\d]*emprego',
               'gostaria[.\w\s\d]*emprego', 'queria[.\w\s\d]*emprego', 'preciso[.\w\s\d]*emprego',
               'procurando[.\w\s\d]*emprego', 'enviar[.\w\s\d]*curr[í|i]culo', 'envie[.\w\s\d]*curr[í|i]culo',
               'oportunidade[.\w\s\d]*emprego',
               'temos[.\w\s\d]*vagas']}

    random_df = pd.concat([pd.read_parquet(path) for path in Path(path_data).glob('*.parquet')])
    random_df['text_lower'] = random_df['text'].str.lower()

    ngram_list = combined_list_dict[args.country_code]
    regex_list = [re.compile(regex) for regex in ngram_list]
    random_df['seedlist_keyword'] = random_df['text_lower'].apply(
        lambda x: regex_match_string(ngram_list=ngram_list, regex_list=regex_list, mystring=x))
    sample_df = random_df.loc[random_df['seedlist_keyword'] == 1]
    sample_df.to_parquet(os.path.join(output_path, f'boris_{args.country_code}.parquet'), index=False)