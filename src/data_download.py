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


    #download the readme
    database_collection.download_resources(filename=f'README_data.md', url=f'{url}/files/README_data.md?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}README_data.md', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')

    # Download the raw data
    database_collection.download_resources(filename=f'0_raw_original_collated.zip', url=f'{url}/files/0_raw_original_collated.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}0_raw_original_collated.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')

    #download the data use to train each model
    database_collection.download_resources(filename=f'1_specific_datasets.zip', url=f'{url}/files/1_specific_datasets.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}1_specific_datasets.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')

    #download the models themselves
    database_collection.download_resources(filename=f'Models.zip', url=f'{url}/files/Models.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}Models.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')
        
    #Download the Results output from each of the src scripts
    database_collection.download_resources(filename=f'Results.zip', url=f'{url}/files/Results.zip?download=1', resource_folder=output_folder) 
    with zipfile.ZipFile(f'{output_folder}Results.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{output_folder}')
