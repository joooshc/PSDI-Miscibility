############################## IMPORTANT ###############################
"""
Keep the GetSMILES.py file in the same directory.

###### To fetch PubChem properties use: ######

pc_properties_1 = fetch_pubchem_props(s1, "pubchem_properties1")
pc_properties_2 = fetch_pubchem_props(s2,"pubchem_properties2")

###### To generate and save MACCs keys use: ######

create_maccs_dataset(chemical_pairs, maccs_descriptors(s1), maccs_descriptors(s2), "MACCs_dataset")
"""
########################################################################

import os, csv, GetSMILES, time, warnings
import pandas as pd
import numpy as np
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from sklearn.preprocessing import MinMaxScaler
import MACCs_PCA

# function to get PubChem properties
def fetch_pubchem_props(smiles_list, file_name):
    ToRetrieve = ["MolecularFormula", "MolecularWeight", "XLogP", "ExactMass", 
                  "MonoisotopicMass", "TPSA", "Complexity", "HBondDonorCount", 
                  "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount", 
                  "Volume3D", "XStericQuadrupole3D", "YStericQuadrupole3D", 
                  "ZStericQuadrupole3D", "FeatureCount3D", "FeatureAcceptorCount3D", 
                  "FeatureDonorCount3D", "FeatureAnionCount3D", "FeatureRingCount3D", 
                  "FeatureHydrophobeCount3D", "ConformerModelRMSD3D", "EffectiveRotorCount3D", 
                  "ConformerCount3D"]
    
    # Loop to get properties with smiles codes. Also checks for problematic compounds
    property_list = []
    for smiles in tqdm(smiles_list, desc="Getting PubChem properties"):
        try:
            RawProperty = pcp.get_properties(ToRetrieve, smiles, "smiles") 
            PropertyDict = RawProperty[0]
        except pcp.BadRequestError:
            PropertyDict = {prop: 'NaN' for prop in ToRetrieve}
        
        property_list.append(PropertyDict)
    property_df = pd.DataFrame(property_list, columns=ToRetrieve)

    property_df.to_csv(f"{file_name}.csv", quoting=csv.QUOTE_NONNUMERIC, float_format="%.8f")

    return property_df

def import_pubchem_props(file_name):
    properties = pd.read_csv(f"{file_name}.csv")
    return properties

# function to get RDKit descriptors
def maccs_descriptors(smiles_list):
    MACCs_list = []
    for smiles in smiles_list:
        try:
            compound = Chem.MolFromSmiles(smiles)
            if compound is None:
                raise Chem.rdchem.MolSanitizeException("Failed to generate compound from SMILES")
            # Computing MACCs descriptors
            maccs_fp = MACCSkeys.GenMACCSKeys(compound)
            # Convert MACCS keys to bit vector
            MACCs_vector = [int(bit) for bit in list(maccs_fp.ToBitString())]
            MACCs_list.append(MACCs_vector)
        except Chem.rdchem.MolSanitizeException as e:
            MACCs_list.append([np.nan]*167)  # Append list of NaNs if there's an error
    
    # Convert to numpy array
    MACCs_df = pd.DataFrame(MACCs_list).apply(pd.Series)

    return MACCs_df

def rdkit_descriptors(smiles_list):
    
    descriptor_dict = {}
    for smiles in smiles_list:
        try:
            compound = Chem.MolFromSmiles(smiles)
            if compound is None:
                raise ValueError('Unable to create compound from SMILES string')
            for descriptor_name, descriptor_fn in Descriptors.descList:
                descriptor_value = descriptor_fn(compound)
                
                if descriptor_name not in descriptor_dict:
                    descriptor_dict[descriptor_name] = []
                
                descriptor_dict[descriptor_name].append(descriptor_value)
        except (ValueError, AttributeError) as e:
            # Handle any compounds that can't be created or descriptors that can't be computed
            for descriptor_name, _ in Descriptors.descList:
                if descriptor_name not in descriptor_dict:
                    descriptor_dict[descriptor_name] = []
                descriptor_dict[descriptor_name].append(np.nan)
    
    Descriptors_df = pd.DataFrame(descriptor_dict)
    return Descriptors_df

def create_maccs_dataset(dataset, m1, m2, file_name):
    df_combined = pd.concat([dataset, m1, m2], axis=1)
    df_combined.to_csv(f"{file_name}.csv")

    return df_combined

def merge_data(dataset, pc1, pc2, rk1, rk2, file_name):

    maccs_dataset = MACCs_PCA.MACCs_PCA()
    print(maccs_dataset.shape, dataset.shape, pc1.shape, pc2.shape, rk1.shape, rk2.shape)

    # removing columns with just 0 values
    pc1 = remove_zero_columns(pc1)
    pc2 = remove_zero_columns(pc2)
    rk1 = remove_zero_columns(rk1)
    rk2 = remove_zero_columns(rk2)

    pc_diff = pc1.subtract(pc2)
    rk_diff = rk1.subtract(rk2)

    if (pc1.shape == pc2.shape) and (rk1.shape == rk2.shape):
        # Creating the combined DataFrame and removing NaN rows
        df_combined = pd.concat([dataset, pc1, pc2, rk1, rk2], axis=1)
        df_combined_diff = pd.concat([dataset, pc_diff, rk_diff], axis=1)
        df_combined.dropna(inplace=True)
        df_combined_diff.dropna(inplace=True)
    
    else:
        print("Error. Could not combine table because there are 0 columns for different features in both compounds lists.")
        print(pc1.shape, pc2.shape, rk1.shape, rk2.shape)
    
    c_df_compounds_df = df_combined.iloc[:, 0:2]
    filtered_maccs_dataset = maccs_dataset[maccs_dataset[['Compound1', 'Compound2']].apply(tuple, axis=1).isin(c_df_compounds_df[['Compound1', 'Compound2']].apply(tuple, axis=1))]
    
    df_combined.reset_index(drop=True, inplace=True)
    filtered_maccs_dataset.reset_index(drop=True, inplace=True)
    df_combined_maccs = pd.concat([df_combined, filtered_maccs_dataset.iloc[:, 9:]], axis=1)
    df_combined_maccs_diff = pd.concat([df_combined_diff, filtered_maccs_dataset.iloc[:, 9:]], axis=1)

    df_combined.to_csv("master_dataset.csv")
    df_combined_diff.to_csv("master_dataset_diff.csv")
    # df_combined_maccs.to_csv("master_dataset_with_maccs.csv")
    # df_combined_maccs_diff.to_csv("master_dataset_with_maccs_diff.csv")

    return df_combined, df_combined_diff, df_combined_maccs, df_combined_maccs_diff

def remove_zero_columns(df):
    non_zero_df = df.loc[:, (df != 0).any(axis=0)]

    return non_zero_df

# The main block (execute this script directly)
if __name__ == '__main__':

    # Importing the dataset and respective smiles codes
    current_directory = os.getcwd()
    main_directory = current_directory[:-len('Scripts')]
    file_directory = f"{main_directory}Dataset"
    os.chdir(file_directory)

    chemical_pairs = pd.read_csv(f"{file_directory}\IUPAC_dataset_combined.csv")
    compounds = GetSMILES.to_np_arrays(chemical_pairs)
    s1, s2 = GetSMILES.from_smiles()

    # pc_properties_1 = fetch_pubchem_props(s1, "pubchem_properties1")
    # pc_properties_2 = fetch_pubchem_props(s2,"pubchem_properties2")

    # Importing the PubChem properties previously fetched for each compound list
    pc_properties_1 = import_pubchem_props("pubchem_properties1")
    pc_properties_2 = import_pubchem_props("pubchem_properties2")
    rk_descriptors_1 = rdkit_descriptors(s1)
    rk_descriptors_2 = rdkit_descriptors(s2)

    c_df, c_df_diff, cwm_df, cwm_diff_df  = merge_data(chemical_pairs, pc_properties_1.iloc[:, 2:], pc_properties_2.iloc[:, 2:], rk_descriptors_1, rk_descriptors_2, "master_dataset")

    print(f"\nMaster dataset shape: {c_df.shape} \nMaster dataset diff shape: {c_df_diff.shape} \nMaster dataset with MACCs shape: {cwm_df.shape} \nMaster dataset diff with MACCs shape: {cwm_diff_df.shape} \n")


