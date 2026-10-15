# Example Scientific Computing Projects

Directories are sorted by when I made the projects in my undergraduate degree. The "Independent" directory contains projects that I made outside of any university class/study.

# How to run the code
The code is intended for reading only however there are README files for some of the projects in the senior honours directory. 
For python files all that is needed is the latest versions of matplotlib, numpy and scipy (as of 15/10/2026). These can be installed via:
```Bash pip3 install matplotlib numpy scipy```
I like to separate the "functional" code from the "user interface" code and as such all of the projects have a python file with the user interaction logic
(which is the one that should be run) and a file that contains most of the interesting code. 

i.e. in `project_examples/senior_honours/ising_model/version_1` there is `run.py` and `dynamics.py`; the interesting code is contained inside the latter but the former
would be run like:
```Bash python3 run.py```

All of the senior honours projects have help menus via the `--help` flag as well as their own README files present in the directories.
```Bash python3 run.py --help```

# Examples
Most of the directories contain example outputs of the code. It will be found that the data/figures is formatted to a much higher standard in `senior_honors` projects as these
were formally assessed. The independent project were made for my own experimentation/practice only and as such little time was taken to polish the outputs.

# Optimisation
The 'version_2' projects in `senior_honours` were submitted for coursework/examination and 'version_1' was my first draft code. Feel free to compare the two versions.
