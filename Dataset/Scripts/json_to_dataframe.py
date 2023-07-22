import json, os, time
import pandas as pd
import numpy as np

# Importing the dataset and respective smiles codes
current_directory = os.getcwd()
main_directory = current_directory[:-len('Scripts')]
file_directory = f"{main_directory}Dataset"
os.chdir(file_directory)

with open('IUPAC_dataset_combined.json') as json_file:
    dataset = json.load(json_file)

compound_1 = []
compound_2 = []
for pairs in dataset.keys():
    compounds = pairs.split(' & ')
    compound_1.append(compounds[0])
    compound_2.append(compounds[1])

mole_fractions_lists = []
for entry in dataset.values():
    mole_fractions = []
    for sub_entry in entry:
        mole_fractions.extend(sub_entry['mole fraction'])
    mole_fractions_lists.append(mole_fractions)

corresp_temps_lists = []
for entry in dataset.values():
    temperatures = []
    for sub_entry in entry:
        temperatures.extend(sub_entry['corresp temps (C)'])
    corresp_temps_lists.append(temperatures)

cas_nums_1 = []
cas_nums_2 = []

for entry in dataset.values():
    for sub_entry in entry:
        cas_nums = sub_entry['cas num']
        
        # If the cas_nums is a string, split it
        if isinstance(cas_nums, str):
            cas_nums = cas_nums.split(', ')
        elif isinstance(cas_nums, list) and len(cas_nums) == 1:
            cas_nums = cas_nums[0].split(', ')
        
        cas_nums_1.append(cas_nums[0]) 
        if len(cas_nums) > 1:
            cas_nums_2.append(cas_nums[1])

mf_means = [np.mean(i) for i in mole_fractions_lists]
mf_stds = [np.std(i) for i in mole_fractions_lists]
temp_means = [np.mean(i) for i in corresp_temps_lists]
temp_stds = [np.std(i) for i in corresp_temps_lists]

dataframe = {'Compound1': compound_1, 'Compound2': compound_2, 'CAS1': cas_nums_1, 'CAS2': cas_nums_2, 'MF_mean': mf_means, 'MF_std': mf_stds, 'Temp_mean': temp_means, 'Temp_std': temp_stds}
print(len(compound_1), len(compound_2), len(mf_means), len(mf_stds), len(temp_means), len(temp_stds), len(cas_nums_1), len(cas_nums_2))
df = pd.DataFrame(dataframe)
df.to_csv('IUPAC_dataset_combined.csv', index=False)