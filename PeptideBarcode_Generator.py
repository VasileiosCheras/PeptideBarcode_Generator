"""
############ INTRODUCTION ############

PeptideBarcode_Generator is a Python script that uses function "Create_barcode_left_to_right", defined within the script, to generate peptide sequences of various lengths.
Then the DeepMSPeptide Python script (1) is used to predict the detectability of the generated peptide sequences in mass spectrometry analysis.

After installing the necessary packages and importing the appropriate python tools, the function "Create_barcode_left_to_right" is defined.
In short this function generates a list of random peptide sequences based on the following user specifications.
The user can define:
       a. the min and max length of the peptides that will be generated 
       b. maximum number of consecutive amino-acid (aa) repetitions
       c. The aa library that will be used to generate the peptide barcode

Next,the function is called with the user-defined arguments (line 109) and the generated peptide sequences are stored in the PeptideBarcode.txt file. 

In the next step the MS detectability of the generated peptides is calculated using the DeepMSPeptide Python script.
In short, this script uses the pre-compiled and pre-trained one dimension Convolutional Neural Network, in order to calculate the detection probability of the input peptide sequences. 
The script is then filtering the peptides and proceed only with those that have a detection probability higher than a certain threshold. This default threshold value is set at 0.8.
The user can change it by modifying the line 199. The prediction algorithm output before and after applying the detectability cut-off is stored in two text files (PeptideBarcodes_Predictions.txt and PeptideBarcodes_Predictions_cutoff.txt)
in the working directory. Thus the user has access to the detectability of all the generated peptides. The molecular weight of the peptides is also calculated. Peptides sharing the same weight are compared for their detection probability.
Only the ones with the highest detection probability are kept and a list of peptide sequences with unique molecular weights are saved in the PeptideBarcodes_Final.csv, located in the working directory.

A series of messages are also printing on the command line to inform the user about the:

	a.The number of peptides with detection probability higher than the set detection probability threshold (default 0.8)
	b.The number of peptides of which the weight was calculated
	c.The minimum and maximum weights of the generated peptide sequences
	d.The number of unique weights
	e.Peptides saved in PeptideBarcodes_Final.csv


[1]"Guillermo Serrano et al, DeepMSPeptide: peptide detectability prediction using deep learning, Bioinformatics, Volume 36, Issue 4, February 2020, Pages 1279–1280 https://doi.org/10.1093/bioinformatics/btz708"

For copyright license to use DeepMSPeptide in this script, please refer to the README file 
"""

############ SCRIPT ############
# Install the following packages

#pip install biopython
#pip install tensorflow


#import tools that will be used later
import random
from random import choice
import csv
import numpy as np
from tensorflow import keras


#Change to the oworking directory containing the model.
#import os
#path='Desktop/PeptideBarcode_Generator/Curated_list'
#os.chdir(path)

"""
"Create_barcode_left_to_right" is a function which returns barcodes of various lengths in a range from 10 to a defined
maximum length (max_length). The barcodes consist of elements from "alphabet" specificed by the user when the function 
is called and with a maximum of consecutive letters (max_repetitions).
By changing the number inside the randrange parenthesis (here: 10) and selecting a max_repetitions number when
calling the function, the user can define the min and max length of the peptide barcodes that will be generated
    :param alphabet: string of all elements/letters of which the barcode consists of, e.g. single letter
    amino acides
    :param max_length: maximum length of barcode
    :param max_repetitions: maximal number of consecutive repetitions of the same letter
    :return: number in scientific notation
"""

def create_barcode_left_to_right(alphabet, max_length, max_repetitions):
    # start with empty barcode
    barcode = ""
    # last char is char to the left of the currently created position
    last_char = ""
    # counter for the number of consecutive repetitions
    n = 0
    # length of the generated barcode (will be in the range specified below) 
    length = random.randrange(10, max_length)
    
    while len(barcode) <= length:
        char = choice(alphabet)

        # count the number of consecutive repetitions
        if char == last_char:
            n += 1
        else:
            n = 1

        # only add letter to barcode if the maximum number of consecutive repetitions is not exceeded
        if n <= max_repetitions:
            barcode += char
            last_char = char

    # return the generated barcode and add at the end either the amino acid K or R
    return (barcode + choice("KR"))

# create empty list to store the generated barcodes
L = []

# create n barcodes and save them in list L and in .txt file named PeptideBarcodes.txt
# Edit n to specify how many peptides will be generated
# Edit the string inside the parameter alphabet to specify the aa composition
# Edit the parameter max_repetitions to specify the maximum number of consecutive aa
n = 20000
with open('PeptideBarcodes.txt', 'a') as output_file:
    for i in range(n):
        new_barcode = create_barcode_left_to_right(alphabet= 'ADEFGILNPQSTVWY', max_length=32, max_repetitions=4)
        L.append(new_barcode)
        output_file.write(new_barcode + '\n')

"""
"load_pep_and_codify" is a function that uses a deep learning method to predict peptide detectability exclusively 
based on the peptide amino acid sequences (see introduction for reference to DeepMSPeptide script)
    :param file: /path/to/peptide/file.txt (specified when running the script e.g. python PeptideBarcode_Generator.py PeptideBarcodres.txt)
    :param max_length: maximum length of barcodes
It returns the pre-processed peptides (as 2D Numpy array), number of peptides whose detectability was predicted 
#and the peptide sequences
"""
def load_pep_and_codify(file, max_len):
    aa_dict={'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,
        'P':15,'O':16,'S':17,'U':18,'T':19,'W':20,'Y':21,'V':22}
    # open file and read it line by line
    with open(file, 'r') as inf:
        lines = inf.read().splitlines() 
    # create empty list, named pep_codes  in which the peptides will be saved as their dictionary codes   
    pep_codes=[]                            
    long_pep_counter = 0  
    # create empty list, named newLines  in which the peptides will be saved                 
    newLines = []  
    # associate each peptide with the appropriate dictionary codes and save the code into pep_codes list                     
    for pep in lines:                       
        if not len(pep) > max_len:          
            current_pep=[]                  
            for aa in pep:                  
                current_pep.append(aa_dict[aa])             
            pep_codes.append(current_pep) 
            # add to the newLines list the peptide sequence
            newLines.extend([pep])  
        # count how many peptides were skipped because they were longer than the specified max length (e.g. here: 81) and the prediction will not be made     
        else:
            long_pep_counter += 1
    #processing peptides to predict detectability later by transforming the list of peptide sequences into a 2D Numpy array 
    #pad_sequences is used to ensure that all sequences in a list have the same length by padding 0 in the 
    #              beginning of each sequence until each sequence has the same length as the longest sequence.
    predict_data = keras.preprocessing.sequence.pad_sequences(pep_codes, value=0, padding='post', maxlen=max_len)
    return predict_data, long_pep_counter, newLines

#Load the model
print('Loading model...')
model_2_1D = keras.models.load_model('model_2_1D.h5')

#Load input peptides into the model, use of "load_pep_and_codify" function 
print('Loading input peptides')
predict_data, skipped,  lines = load_pep_and_codify("PeptideBarcodes.txt", 81)
print('Succesfully loaded {0} peptides and skipped {1}'.format(len(lines), str(skipped)))

#Generate predictions for the detactability of the peptides
print('Making predictions')
model_2_1D_pred = model_2_1D.predict(predict_data)
model_2_1D_pred = np.hstack((np.array(lines).reshape(len(lines), 1),model_2_1D_pred)).tolist()

#Save detectability predictions in pred_output list and 
# add Detectability code 0 if detactability < 0.5 and 1 if detactability > 0.5 
Pred_output = []
for pred in model_2_1D_pred:
    if float(pred[1]) > 0.5:
        # pred.extend('0')
        Pred_output.append([pred[0], str(1-float(pred[1])), '0'])
    else:
        Pred_output.append([pred[0], str(1-float(pred[1])), '1'])
        # pred.extend('1')

#Save detection probability and detectability (0 or 1) in .txt file, then notify user 
outFile = 'PeptideBarcodes_Predictions.txt'
print('Saving predictions to file {}'.format(outFile))
with open(outFile, 'w') as outf:
    outf.write('Peptide\tProb\tDetectability\n')
    outf.writelines('\t'.join(i) + '\n' for i in Pred_output)

#Return a list containing each line in the file as a list item. 
a_file = open("PeptideBarcodes_Predictions.txt", "r")
lines = a_file.readlines()[1:]
a_file.close()

counter_total= 0    # number of total peptides generated and whose detection probability was calculated
counter_final = 0   # number of peptides whose detection probability is above desired threshold 

#Open new file and add the header line
new_file = open("PeptideBarcodes_Predictions_cutoff.txt", "w")
new_file.write('Peptide\tProb\tDetectability\n')

#Save peptides with detection probability above 0.80 into the new file
#The user can change 0.80 to the desired cutoff (e.g 0.90 or 0.60)
for line in lines:
    counter_total = counter_total + 1
    if float(line.split("\t")[1]) >= 0.80:
        new_file.write(line)
        counter_final = counter_final + 1

new_file.close()


#Introducing the weights of the amino acids
weights = {'A': 71.0371, 'D': 115.0269, 'E': 129.0425, 'F': 147.0684, 'G': 57.0214, 'K': 128.0949,       
           'I': 113.0840, 'L': 113.0840, 'N': 114.0429, 'P': 97.0527, 'Q': 128.0585, 
           'S': 87.0320, 'T': 101.0476, 'V': 99.0684,  'W': 186.0793, 'Y': 163.0633 , 'R': 156.1011}

 #'C': 103.0091, 'M': 131.0404, , 'H': 137.0589

count_weights = 0
count_weights1 = 0

#Introducing the lists and variables that will be used to calculate the weights of the peptides
weight_list = []
sequence_weight_list = []
weight_int = 0 
weight_int_list = []
weight = 0
listpep=[('Peptide','Prob','Weight')]
mx = None
mn = None

#Open the previously created file and read each line
#It returns a list containing each line in the file as a list item. 
#Here it is saved in the list named lines
b_file = open("PeptideBarcodes_Predictions_cutoff.txt", "r")
lines = b_file.readlines()[1:]
b_file.close()

#Function to return index of item within a list of lists
def index(x, lst):
    for i, row in enumerate(lst):
        for j, element in enumerate(row):
            if element == x:
                return (i)
    return (-1, -1)

#For each peptide saved in list lines, calculate the weight
for line in lines:
    sequence = line.split("\t")[0]
    prediction1 = line.split("\t")[1]

    for x in sequence:
        weight = sum(weights[x] for x in sequence) + 18.3212 
        weight_int = int(weight) 

        #reports max and min weights
        if mx is None or weight > mx:
            mx = round(weight, 2)
        if mn is None or weight < mn:
            mn = round(weight,2)

    count_weights = count_weights + 1

    #if two peptides share the same weight, keep only the peptide with the heighest detection probability
    if any(sub[2]==weight_int and sub[1] <= prediction1 for sub in listpep):
        a = index(weight_int, listpep)
        listpep.append([sequence, prediction1,weight_int])
        listpep.pop(a)
    elif any(sub[2]==weight_int and sub[1] > prediction1 for sub in listpep):
        continue
    else:
        listpep.append([sequence, prediction1,weight_int])

 
    weight_list.append(weight_int)

#Save peptides with unique weights into a csv file
with open("PeptideBarcodes_Final.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(listpep)

#Sanity check point
print ("- Number of peptides generated: " + str(counter_total))
print ("- Number of peptides with detection probability higher than 0.80: " + str(counter_final))
print ("- Number of peptides of which the weight was calculated: " + str(count_weights))
print ("- The minimum and maximum weights are: {} Da, {} Da".format(mn,mx))

weight_set = set(weight_list)
print ("- Number of unique weights: " + str(len(weight_set)))
print ("- Peptides saved in final file are:" + str(len(listpep)-1))

    