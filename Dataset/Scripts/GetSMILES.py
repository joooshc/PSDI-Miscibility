import os
import pandas as pd
import numpy as np
import pubchempy as pcp

# function to convert table to np arrays
def to_np_arrays(chemical_pairs):
    # Creating arrays for each compound to merge later
    compound1_names = chemical_pairs['Compound1'].tolist()
    compound1_cas = chemical_pairs['CAS1'].tolist()
    compound2_names = chemical_pairs['Compound2'].tolist()
    compound2_cas = chemical_pairs['CAS2'].tolist()
    compounds = np.array([compound1_names, compound1_cas, compound2_names, compound2_cas])

    return compounds

# function to get smiles codes and saves it to a file
def to_smiles(compounds):
    
    count = 1
    print("Getting SMILES codes, please wait...")
    smiles_list_1 = []; smiles_list_2 = []
    for cas1, cas2 in zip(compounds[1], compounds[3]):
        compounds1 = pcp.get_compounds(cas1, 'name')
        compounds2 = pcp.get_compounds(cas2, 'name')
        if compounds1:
            smiles_list_1.append(compounds1[0].canonical_smiles)
        else:
            smiles_list_1.append('NaN')
        if compounds2:
            smiles_list_2.append(compounds2[0].canonical_smiles)
        else:
            smiles_list_2.append('NaN')
        
        count += 1
        print(f"Progress: {np.round(100*(count/len(compounds[1])), 2)}%")
    
    with open('smiles1.txt', 'w') as file:
        for smiles in smiles_list_1:
            file.write(smiles + '\n')
    with open('smiles2.txt', 'w') as file:
        for smiles in smiles_list_2:
            file.write(smiles + '\n')
    
    print(f"SMILES codes saved to smiles1.txt and smiles2.txt")
        
    return smiles_list_1, smiles_list_2

# function to import smiles codes from a file
def from_smiles():
    smiles1 = []
    smiles2 = []
    with open('smiles1.txt', 'r') as file:
        for line in file:
            smiles = line.strip()
            smiles1.append(smiles)
    with open('smiles2.txt', 'r') as file:
        for line in file:
            smiles = line.strip()
            smiles2.append(smiles)

    return smiles1, smiles2
