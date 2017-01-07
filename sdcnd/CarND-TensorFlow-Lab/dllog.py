"""
Contains utility methods for logging deep learning network runs
"""
DEFAULT_USER = ('Quoc', 'Pham')
DEFAULT_ENV = 'Beholder'

def dl_run_start(network, model_file_path, data, hyper_dict, user=DEFAULT_USER, \
    environment=DEFAULT_ENV):
    """ Log the start of a deep learning network execution """
    print("Run started with network %s with model source %s on dataset %s using \
        hyper-params %s by user %s on %s." % (network, model_file_path, data, hyper_dict, \
        user, environment))
    return 0

def dl_run_end(run_id, accuracy):
    """ Log the end of a deep learning network execution """
    print("Run %s ended with accuracy %f" % (run_id, accuracy))
