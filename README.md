# Machine Learning Models on Miscibility Datasets
Created by [Daniel C](dc1n19@soton.ac.uk), [Josh C](jc10g22@soton.ac.uk) and [Thasmia F](tmab1g21@soton.ac.uk) with help from [Dr Jo Grundy](https://www.southampton.ac.uk/people/5xrlgf/doctor-jo-grundy).  

This project focused on using machine learning (ML) to predict how well different liquids mix together to form a uniform solution. The key to making accurate predictions was having a large and reliable dataset with information about how these substances mix. A property called mole fraction was used, which represents the proportion of each substance present in the mixture. By determining the mole fraction, it might be possible to say how well two liquids can mix at a given temperature.

Various ML models (Linear, Polynomial, Random Forest, Gradient Boosting, and Neural Networks) were trained and tested using IUPAC data, which included information about different mixtures and their mole fractions at specific temperatures. Ensemble learning methods, such as Random Forest and Gradient Boosting, showed promising results. These methods combine the knowledge of many individual models to make better predictions. For example, the random forest model works by taking the average of predictions from different models to form a final reliable prediction, and gradient boosting learns from the mistakes of previous models to improve its predictions. However, the most accurate model was a Feed Forward Neural Network, which processes input data through interconnected inputs, referred to as neurons, to learn patterns in the data necessary to make accurate predictions.

Several challenges were encountered due to limited data on miscibility and the need for manual data preprocessing. The models were tested within specific temperature ranges, potentially affecting their reliability for extrapolation to other temperatures. Nevertheless, the project demonstrated the potential of ML in predicting mole fractions of mixtures. Further research could address the models' limitations and enhance their applicability to a broader range of conditions and substances, opening up new possibilities for optimising mixtures in various industries.

## Documentation
Available in the Wiki

## Installation
Recommended method is to create a copy of the repository on your local machine, then open it as a folder in your preferred code editor.

## Dependencies
- Python 3.11.2 or newer

The following packages are required:  
_All available to install using pip_
- [RDKit](https://www.rdkit.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandas](https://pandas.pydata.org/)
- [CatBoost](https://catboost.ai/)
- [TensorFlow](https://www.tensorflow.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [CIRpy](https://cirpy.readthedocs.io/en/latest/)
- [NumPy](https://numpy.org/)
- [PubChemPy](https://pubchempy.readthedocs.io/en/latest/)