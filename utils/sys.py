import sys
import os
import logging


def debugger_is_active() -> bool:
    """
    Return if the debugger is currently active
    https://stackoverflow.com/a/67065084
    Tested on VSCode
    """
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def initialize_logger(output_path=None):
    if output_path is None:
        output_path = './results'

    log_filename = f'{output_path}/log.txt'

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(
        filename=log_filename, encoding='utf-8', filemode='w', level=logging.INFO)
