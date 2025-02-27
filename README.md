# M2Diff

This is official repository for paper "M2Diff: Multi-Modality Multi-Task Enhanced Diffusion for MRI-Guided Low-Dose PET Enhancement" submitted to MICCAI 2025.

Monash DaCRA fPET-fMRI dataset: https://openneuro.org/datasets/ds003397/versions/1.2.3
ADNI dataset: https://adni.loni.usc.edu/data-samples/adni-data/

All the hyperparameter we used for our traaining are already set in the file except for batch size.
Batch Size for Monash DaCRA dataset: 1
Batch Size for ADNI dataset: 5

Data needs to be sliced into 2D mat files and data folder can be structured like this:

data
|
--test
|
------LD
------SD
------T1
|
--train
|
------LD
------SD
------T1
|
--val
|
------LD
------SD
------T1
