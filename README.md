# SynapseDiffsion

The purpose of this is to utilize MEG and EEG data recorded on a visual stimuli (image for couple seconds) and decode it and resynthesize the image.

## Important Files

### Data Converter Files

These files convert the raw eeg and meg datafiles into tensor files for ease in training: 
- [eeg_load_batches](eeg_load_batches.py)
- [meg_load_batches](meg_load_batches.py)
- [eeg_meg_comb_load_batches](eeg_meg_comb_load_batches.py)

#### Slurm Script
- [batchLoader.sh](batchLoader.sh)
`Usage:  [sbatch| ] ./batchLoader [eeg|meg|comb]  note: sbatch only enabled on slurm managed clusters`

### Trainer Files
- Specified Experiment Initialization (hyperparamters, optimizer, etc.)
- Train and test the model 
    - Option to use DDP or FSDP if available
- Generates Confusion Matrix with classification results

#### Slurm Training Scripts
These are used to start training on HPC cluster GreatLakes
[EEG](train_eeg.sh),
[MEG](train_meg.sh),
[Fusion Classification](train_fusion_classification.sh)

#### [Trainer](training.py) Class
- Common code in training pipeline

Confusion matricies below are of order validation, test.

#### [EEG Decoder](./eeg_trainer.py)

- [3 Class](EEG_Test_and_Val_CM's_c10,11,16.png): Fruit, Furniture, and Medical equipment
- [2 Class](EEG_Test_and_Val_CM's_c10,11.png): Fruit, Furniture

#### [MEG Decoder](./meg_trainer.py):
Confusion matricies:
- [Classes 10,11](MEG_Test_and_Val_CM's_c10,11.png): Fruit, Furniture
- [Classes 2, 10](MEG_Test_and_Val_CM's_c2,10.png): Body Parts, Furniture

#### [Fusion Decoder](./gen_trainer.py):
Confusion matricies
- To generate confusion matrix

#### Synapse Diffusion
Upcoming

### Hyperparamter selection
Using Ray[Train,Tune] for automated hyperparameter selection 
`note: Ray is a self managed distributed systems training and inference library`

#### Ray Compatible [Trainer](training_ray.py) Class
- TODO: complete config and setup
- TODO: add automated toggle for single gpu or distributed depended on available system

## Data
Current MEG Data Source -> THINGS MEG1: https://openneuro.org/datasets/ds004212 \
Current EEG Data Source -> THINGS EEG: https://openneuro.org/datasets/ds003825/versions/1.2.0
- Things MEG1, from [THINGS](https://things-initiative.org): A global initiative bringing together researchers using the same image database to collect and share behavioral and neuroscience data in object recognition and understanding.
