from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Knowledge exploration with NLP'
LONG_DESCRIPTION = 'Python package geared towards extracting knowledge from PDF documents (specially papers) using ' \
                   'Spacy and Haystack amongst others. Guides and examples available at https://github.com/Davegdd/knowlexp ' \
		   '(manually install dependencies farm-haystack and PyMuPDF to use this package). ' \

setup(

    name="knowlexp",
    version=VERSION,
    author="Dave Dominguez",
    url = 'https://github.com/Davegdd/knowlexp',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],


    keywords=['Python', 'NLP', 'Spacy', 'Haystack', 'knowledge', 'papers'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Information Analysis",

    ]
)



