## ideas

### Training Considerations
Distributed data parallel 
for hyperparameter search. so it would be diff hyperparams for model/optim with a copy of the data
https://www.anyscale.com/blog/deep-dive-data-ingest-in-a-third-generation-ml-architecture

+ model parallel?
+ Other optimaizations

Multi-gpu 
single or multi system



data in batches -> all datapoints in batch run on separate set of blocks
https://d2l.ai/chapter_computational-performance/multiple-gpus.html

https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/
    Use 1,2,4
PyTorch AMP
PyTorch AdamW - fused kernel (fused=True)
Weight-decay explanation: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html#adamw
https://towardsdatascience.com/how-to-solve-data-loading-bottlenecks-in-your-deep-learning-training-1ddfcc24449b


https://github.com/pytorch/examples/blob/main/distributed/FSDP/T5_training.py

### Related work
https://www.cs.cmu.edu/~afluo/

EEG BCI literature review
https://www.sciencedirect.com/science/article/pii/S2665917423001599

fMRI decoding
https://web.eecs.umich.edu/~weimerw/p/weimer-icse2017-preprint.pdf
    root ^: https://web.eecs.umich.edu/~weimerw/fmri.html

Object tracking:
https://arxiv.org/pdf/2103.17154.pdf

Spatiotemporal transformer analysis for attention estimation with eeg:
https://arxiv.org/pdf/2204.07162v1.pdf

EEG connectivity analysis:
https://www.mdpi.com/2306-5354/10/3/372s

things-initiative MEG:
https://elifesciences.org/articles/82580#s4

### Code Resources
https://github.com/analyticalmonk/awesome-neuroscience?tab=readme-ov-file
https://github.com/vlawhern/arl-eegmodels


### Model Architecture
https://www.frontiersin.org/articles/10.3389/fnhum.2023.1169949/full
https://www.reddit.com/r/MachineLearning/comments/18bd0lw/d_what_is_the_latest_with_multimodal/

#### New meg model:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6609925/

#### generation module:
https://github.com/facebookresearch/multimodal/tree/main MULTIMODAL lib 
https://arxiv.org/pdf/2209.08725.pdf: neural wavelet domain diffusion for 3d shape generation
https://github.com/edward1997104/Wavelet-Generation

GATv2Optim Θ_n^+ 
pass through classifier models and both in embedded space? and make eeg and meg CLIP shaped (dim of last dense layer before output)
then concat or add? if concat then make last dense dim half of clip size?
do embedding space alignment with clip space:
    Contrastive learning for embedding space?, needs research 
    Use CLIP model to tokenize/embed the original image
    use simple model or cosine sim or something to align the meg+eeg space to clip via the true output of that image in CLIP
pass to generative model that can use clip and finetune

There could be better other aligned spaces other than CLIP

#### Structure
STGATE (Spatial-Temporal Graph Attention Net): https://www.frontiersin.org/articles/10.3389/fnhum.2023.1169949/full
    Transformer encoder

EEG:
Graph Attention Network + FC
CNN + Transfomer, Transformer + CNN
CNN + Graph Attention Network
Transformer + Graph Attention network
CNN (s) + Transformer (t) + Graph Attention Network

#### Reference
GATv2Conv analysis = https://arxiv.org/pdf/2305.16196.pdf

#### Future
Need to consider doing causal embedding/attention to allow for real time decoding

### Data
https://vscode.dev/github.com/ViCCo-Group/THINGS-data

#### Preprocessing
EEG: Using Matlab (R2020b) and the EEGlab (v14.0.0b) toolbox22, data were filtered using a Hamming windowed FIR filter with 0.1 Hz highpass and 100 Hz lowpass filters, re-referenced to the average reference, and downsampled to 250 Hz. Epochs were created for each individual stimulus presentation ranging from [−100 to 1000 ms] relative to stimulus onset. 


#### Metrics
https://arxiv.org/pdf/2007.15359.pdf
https://arxiv.org/pdf/2206.10935.pdf

##### Q's
No image, just labels with MEG data
    borrow brain module from speech meg paper, if anything useful also from meta-brain-decoding
    

    Speech Meg:
        Contrastive loss to map learned meg encodings with speech encodings and then classify/ get speech sample from meg data
        brain module:
            3D-> 2D map of meg channels.
                Normalize
                

Questions:
    For skip-connections do people always just add the input without transformations (y = f(x) + x), are there other formats of skip-connections that don't add the identity?
    Multi-path feature extraction ex from class: waveform -> conv, logmel -> conv, 2dconv -> wavegram, feature_maps, concat
    Why concat, in what scenarios is concat good/bad
        would you want to use another form of aggregation?
    
    What would you recommend we do, for training our models.
        We have 197GB data -> 55GB EEG; 128GB MEG; 14GB Images;How would you recommend we train with the large amount data?
        To train a small model on a subset, or similar size model how extensive should hyperparamter search be for a small model+subset?
    
    create user group from local 



dwt 1x 63,100 -> 4x 32,50
    4x 32,50 -> 16x 16,25
16x coef shape:16,25 
https://github.com/JDAI-CV/CoTNet/tree/master For after simple conv of wavelet model, pass conv to this block before doing classification
https://github.com/YehLi/ImageNetModel?tab=readme-ov-file More reference