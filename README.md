This template is designed to give a framework for public distributions of "science" projects. 
It is a guideline, showing the minimum things recommended to include with your public repository, 
to make your results easily replicable. 
It is not exhaustive by any means, nor is everything here strictly required in all cases! 
Please consider this as a loose list of things considered "nice to have", and as reference material above all. 

# DeepSkies Science Repo Template 
Include status links to different outside resources, such as build info, paper info, license, etc. 
You can select your license from the [choose a license page.](https://choosealicense.com/licenses/), and then change the name of the license in the badge and link included. 
For workflows, change the name of the repo listed in the img.shields link to point to your repo and workflows.

[![status](https://img.shields.io/badge/arXiv-000.000-red)](arxiv link if applicable)
[![status](https://img.shields.io/badge/PyPi-0.0.0.0-blue)](pypi link if applicable)
[![status](https://img.shields.io/badge/License-MIT-lightgrey)](MIT or Apache 2.0 or another requires link changed)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/owner/repo/build-repo)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/owner/repo/test-repo?label=test)

Your overview should contain a brief summary of the project, and figures and examples showing input and output. 

## Installation 
Information about install. 
We recommend publishing to pypi using a poetry package management system (described below) but we also provide instructions for using python virtual environments and showyourwork with conda integration. 

Example of what your installation instructions should look like: 

To install with pip: 
> pip install git+https://github.com/DeepSkies/science_template.git
>
This will set up a virtual environment, which can b  e run with on mac or linux
> source venv/bin/activate
>
Or on windows with 
> venv\Scripts\activate.bat

Verify installation is functional is all tests are passing
> pytest

Additionally, include how to install from source (via git clone) and associated setup. 

### poetry 
Poetry is our recommended method of handling a package environment as publishing and building is handled by a toml file that handles all possibly conflicting dependencies. 
Full docs can be found [here](https://python-poetry.org/docs/basic-usage/).

Install instructions: 

Add poetry to your python install 
> pip install poetry

Install the pyproject file
> poetry install 

To add another package to your environment
> poetry add (package name)

To run within your environment 
>poetry run (file).py

If you wish to start from scratch: 
> pip install poetry
> poetry init

### virtual environment
At the bare minimum, project dependencies must be contained and strictly defined and shared for replication purposes. 
The easiest way to do this is to use a python virtual environment. 
Full instructions are found [here.](https://docs.python.org/3/library/venv.html)

To initialize an environment:
> python3 -m venv /path/to/env
> 
To activate it: 
Linux and Mac: 
> source venv/bin/activate
> 
Windows: 
> venv\Scripts\activate.bat

And use pip as normal to install packages. 

In order to produce a file to share with your version of dependencies, produce a requirements.txt. 
This can later be installed in full to a new system using `pip install -r requirements.txt`. 
Note that this does not manage any versioning conflicts and can very quickly become depreciated. 
> pip freeze >requirements.txt 

### show your work with conda
We also supply a ["show your work"](https://github.com/showyourwork/showyourwork) workflow to use with a conda venv which can compile the example tex file in `DeepTemplate-Science/src/tex/ms.tex`

To execute this workflow: 
>showyourwork build

This will build your project and install the conda venv associated with the project (or just compile the document if you haven't been using it) and output the document as a pdf. 
If you would like to integrate with overleaf to push your work remotely, you can do that by adding the following lines to your showyourwork.yml file: 
> 
>   overleaf: 
> 
>       id: URL identifying your project
>       push:
>           - src/tex/figures
>           - src/tex/output
>       pull:
>           - src/tex/ms.tex
>           - src/tex/bib.bib

And adding the system variables `$OVERLEAF_EMAIL` and `$OVERLEAF_PASSWORD` with your credentials. 
To do this, use a bash terminal to input the command `export OVERLEAF_EMAIL='youremail@server.org`, and do the same for your password. 
To verify these are set correctly, run `echo $OVERLEAF_EMAIL`and `echo $OVERLEAF_PASSWORD`. 
To complete this setup, run `showyourwork build` as if you were compiling a project.
The above snippet of the yaml file will then push anything in the `src/tex/figures` and `src/tex/output` folders to the remote, under the `images` folder.  

The existing yaml file is set up to modify the [template project](*https://www.overleaf.com/read/fsjwntpjmdzw). 
The differences in the ID in the template and the url you'll see is due to the fact that only project owners have access to that ID (even if I want to share). 
This limits the person who can build the project to the person that owns the overleaf page, at least until Latex sets up token authentication. 
The workaround for this is account sharing, but this is not recommended. 

For more information please see the [showyourwork page on the topic](https://show-your.work/en/latest/overleaf/).



## Quickstart
Description of the immediate steps to replicate your results, pointing to a script with cli execution. 
You can also point to a notebook if your results are highly visual and showing plots in line with code is desired.

Example: 

To run full model training: 
> python3 train.py --data /path/to/data/folder

To evaluate a single ""data format of choice""
> python3 eval.py --data /path/to/data

## Documentation 
Please include any further information needed to understand your work. 
This can include an explanation of different notebooks, basic code diagrams, conceptual explanations, etc. 
If you have a folder of documentation, summarize it here and point to it. 

## Citation 
Include a link to your bibtex citation for others to use. 

```
@article{key , 
    author = {You :D}, 
    title = {title}, 
    journal = {journal}, 
    volume = {v}, 
    year = {20XX}, 
    number = {X}, 
    pages = {XX--XX}
}

```

## Acknowledgement 
Include any acknowledgements for research groups, important collaborators not listed as a contributor, institutions, etc. 
