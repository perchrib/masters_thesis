from config.global_constants import TEXT_DATA_DIR
from preprocessors.dataset_characteristics import get_data

if __name__ == '__main__':
    authors, female_texts, male_texts = get_data(TEXT_DATA_DIR)
