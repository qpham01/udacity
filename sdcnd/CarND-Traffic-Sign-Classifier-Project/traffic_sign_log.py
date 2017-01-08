"""
Logging of traffic sign model execution results
"""
from dldata import dllog as log

DATASET_NAME = 'German Traffic Sign Dataset'
DL_RUN = "Traffic Sign Classification"
DL_NETWORK = 'Lenet Variation for Traffic Sign'
DL_MODEL_FILE_PATH = 'traffic_sign_model.py'

def log_run_start(run_type, hyper_dict, version_description):
    """
    Log the start info of a neural network run
    """
    dl_data = (DATASET_NAME, run_type)
    return log.dl_run_start(DL_RUN, DL_NETWORK, DL_MODEL_FILE_PATH, dl_data, \
        hyper_dict, version_description)

def log_run_end(id_run, accuracy, training_loss):
    """
    Log the end info of a neural network run
    """
    log.dl_run_end(id_run, accuracy, training_loss)
