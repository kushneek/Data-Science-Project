import os
import logging
from datetime import datetime

file_dir=os.getcwd()
file_name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_dir=os.path.join(file_dir, "logs")

os.makedirs(log_dir, exist_ok=True)
log_file_path=os.path.join(log_dir, file_name)

logging.basicConfig(
    filename=log_file_path, 
    format='[%(asctime)s] %(lineno)d-%(message)s', 
    level=logging.INFO)