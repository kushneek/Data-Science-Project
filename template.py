import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name="mlproject"

list_of_files=[
".github/workflows/.gitkeep",
f"src/{project_name}/__init__.py",
f"src/{project_name}/components/__init__.py",
f"src/{project_name}/components/data_ingestion.py",
f"src/{project_name}/components/data_transformation.py",
f"src/{project_name}/components/model_training.py",
f"src/{project_name}/components/model_monitoring.py",
f"src/{project_name}/pipeline/__init__.py",
f"src/{project_name}/pipeline/training_pipeline.py",
f"src/{project_name}/pipeline/prediction_pipeline.py",
f"src/{project_name}/exception.py",
f"src/{project_name}/logger.py",
f"src/{project_name}/utils.py",
"Requirements.txt",
"setup.py",
"app.py",
"Dockerfile"
]

for filePath in list_of_files:
    filePath=Path(filePath)
    fileDir, fileName= os.path.split(filePath)

    if fileDir!="":
        os.makedirs(fileDir, exist_ok=True)
        logging.info(f"Creating Directory: {fileDir} for the {fileName}")
    
    if((not os.path.exists(filePath)) or (os.path.getsize(filePath)==0)):
        with open(filePath,"w") as f:
            pass 
            logging.info(f"Creating empty file: {filePath}")
    
    else:
        logging.info(f"{fileName} already Exists !!!")
