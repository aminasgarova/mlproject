# Importing tools to build and package the Python project
from setuptools import find_packages, setup  
from typing import List  # Used to specify the return type of our function (a list of strings)

# Constant used to detect editable install reference in requirements.txt
HYPEN_E_DOT = "-e ."

# Function to read and clean the list of required packages from requirements.txt
def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file:
        requirements = file.readlines()   # Read all lines from the file
        requirements = [req.replace("\n", "") for req in requirements]  # Remove newline characters
        #requirements = [req.strip() for req in requirements] 
        if HYPEN_E_DOT in requirements:  # If '-e .' is in the list, remove it
            requirements.remove(HYPEN_E_DOT)
        return requirements  # Return clean list of package names

# Package setup configuration
setup(
    name="mlproject",  # The name of your Python package
    version="0.0.1",  # Initial version of your package
    author="Your Name",  # Replace with your actual name
    author_email="youremail@example.com",  # Replace with your actual email
    packages=find_packages(),  # Automatically find all Python packages (directories with __init__.py(inside src folder))
    install_requires=get_requirements("requirements.txt")  # Install dependencies listed in requirements.txt
)
