# Python Code

This folder named "python" contains all relevant files needed to run the regression models on each dataset in python. Each dataset should have its own ipython notebook that contains the implimentations of all relevant models. 

## Steps to Run Code

1. Create a virtual environment in the python directory. Many commands can be used to do this. I personally use: virtaulenv venv
1. Activate virtaul environment. Command is normally: source venv/bin/activate for macOS/Linux. Command is normally: venv/Scripts/activate for Windows
1. Use the requirements.txt file in "python" directory to load all necessary packages. Command is: pip install -r requirements.txt . However, if you do not wish to make a virtual environment you can run the same command and just install the packages on your local machine. 
1. Choose a python notebook that corresponds to the dataset you are interested in. Then set the kernel to pull from wherever you installed the packages from the requirments.txt. (Either virtual environment or base environemnt on your local machine)
1. Run the entire python notebook or the select parts of the notebook you are interested in. Keep in mind if you do not choose to run the entire notebook, you will at least need to load the libraries and datasets at the beginning of the notebooks for the models and selection techniques to run.