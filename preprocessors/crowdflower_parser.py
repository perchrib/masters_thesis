import pandas as pd
from helpers.global_constants import CROWDFLOWER_CSV_PATH
# pd.set_option('max_columns', 100)
# pd.set_option('expand_frame_repr', False)
pd.set_option('max_rows', 1)

"""
Crowdflower Tweet Gender Dataset Parser

"""


def parse_crowdflower(file_path=CROWDFLOWER_CSV_PATH):
    dataset = pd.read_csv(file_path)

    pd.set_option('display.max_rows', 10)
    # pd.set_option('display.max_columns', None)

    print(dataset['gender'].value_counts())
 
    # pd.reset_option('display.max_rows')
    # pd.reset_option('display.max_columns')


if __name__ == '__main__':
    parse_crowdflower()
