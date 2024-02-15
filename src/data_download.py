import os, re
import zipfile
from shutil import copyfile
from loguru import logger
from utilities import database_collection
from utilities.database_map_and_filter import gz_unzipper, tar_file_to_folder

logger.info('Import OK')

if __name__ == "__main__":

    url = 'https://zenodo.org/records/10602864/' 
    output_folder = 'data/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Download file from repository
    database_collection.download_resources(filename=f'Figures.zip', url=f'{url}/files/0_raw_original_collated.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}Figures.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')

    database_collection.download_resources(filename=f'example_data.zip', url=f'{url}/files/1_specific_datasets.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}example_data.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')
    #download the readme
    database_collection.download_resources(filename=f'example_data.zip', url=f'{url}/files/README_data.md?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}example_data.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')