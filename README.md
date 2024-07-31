# PeptideBarcode_Generator
# PeptideBarcode_Generator.py
PeptideBarcode_Generator is a Python script that uses a built-in function to generate peptide barcodes of various lengths with a user defined detection probability with the help of DeepMSPeptide python script.

# Contact
Vasileios Cheras - vasileios.cheras@bsse.ethz.ch Kobi Benenson - kobi.benenson@bsse.ethz.ch

# Cite DeepMSPeptide python script:
Guillermo Serrano and others, DeepMSPeptide: peptide detectability prediction using deep learning, Bioinformatics, Volume 36, Issue 4, February 2020, Pages 1279‚Äì1280, https://doi.org/10.1093/bioinformatics/btz708

# License to use DeepMSPeptide script as part of PeptideBarcode_Generator python script
MIT Licence
Copyright (c) 2019 Guillermo Serrano Sanz
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#Installation
This script need no installation, but has the following requirements:
    -Tensorflow 1.13.1 or above (https://www.tensorflow.org/install/pip)
    -Python 3.6.5 or above

# How to use PeptideBarcode_Generator
python PeptideBarcode_Generator.py /path/to/peptide/file.txt
python3 PeptideBarcode_Generator.py PeptideBarcodes.txt 
To run the script, copy PeptideBarcode_Generator.py and model_2_1D.h5 to your local drive in the folder. 

