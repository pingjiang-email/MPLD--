Multi-scale Pyramid Deformable Registration with Accurate Similarity Measure for Medical Image Analysis

This is a pytorch unofficial implementation of MPLD paper.


Requirements
CUDA 12.6 
torch 2.0 
Python 3.10

Data

To download the  data at  https://zenodo.org/records/14498769

The image is in the DATA folder

Training

Python train.py 

Configuration parameters for train can be found in configs_MPLDconfigs_MPLD under the models folder

Testing

Python test.py 



/////Detailed description/////


Python train.py ：The training code of MPLD algorithm

Python test.py ：The test code of MPLD algorithm

/////The code for the rest of the files is in the models file////

configs - MPLD. py:  the configuration parameter of the model.

datasets.py: Input form and order of paired registered images.

Eva-model.py and similarity—Evaluator.py: We designed the similarity measurement code, the specific parameters can be changed under these two files

trans.py and transformation.py : The model transformer encoder code

Minmax.py: Data normalization code
