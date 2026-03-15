from setuptools import setup, find_packages

def get_requirements(file_path:str)->list[str]:
    '''
    This function will get the requirements of the project into list 
    '''
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    return requirements


setup ( 
name="mlproject",
author="Kushagra Neekhra",
author_email="kush.neek18@gmail.com",
install_requires=get_requirements("Requirements.txt"),
version="0.0.1",
packages=find_packages()
)

