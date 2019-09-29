import logging

def init(config):
    config = str(config)
    if config.lower() == 'info':
        logging.basicConfig(level=logging.INFO)
    elif config.lower() == 'debug':
        logging.basicConfig(level=logging.DEBUG)
    elif config.lower() == 'error':
        logging.basicConfig(level=logging.ERROR)
    else:
        assert False, "{}: Logging config missing".format(config)

def get_logger(name, config='debug'):
    init(config)
    return logging.getLogger(name)