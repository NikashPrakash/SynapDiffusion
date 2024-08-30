Ivan: x_redu [64,271,281] -> 
x_reduced = self.nonlin_in(tf.tensordot(x,self.W,axes=[[1],[0]], 
                                                            name='de-mix') + self.b_in)
x_redu [64,64,281] ->
x_reduced = tf.expand_dims(x_reduced, -2)

x_redu [64,64,1,281]
conv_ = self.nonlin_out(tf.nn.depthwise_conv2d(x_reduced, 
                                                self.filters, strides=[1,1,1,1],
                                                padding='SAME') + self.b)
        self.filters [filter_length,1,n_ls,1]
        Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a filter tensor of shape [filter_height, filter_width, in_channels, channel_multiplier] containing in_channels convolutional filters of depth 1, depthwise_conv2d applies a different filter to each input channel (expanding from 1 channel to channel_multiplier channels for each), then concatenates the results together. The output has in_channels * channel_multiplier channels.

conv_ [64,64,1,281]
pool [64,64,1,141]
return [64,64,141]

Ours:
self.conv = nn.Conv2d(n_ls, n_ls, kernel_size=(filter_length, 1), stride=(stride, 1),padding='same',groups=n_ls)
shape: [filter,1,n_ls,1]: [9,1,64,1]
    Depthwise convolution:
    When groups == in_channels and out_channels == K * in_channels
    In other words, for an input of size (N, C_{in}, L_{in})​,
    a depthwise convolution with a depthwise multiplier K can be performed with the arguments 
    $(C_{in}=C_{in}, C_{out}=C_{in} \times K, ..., \text{groups}=C_{in})$ 
__________________________________
x [64,271,281] ->
x_reduced = torch.einsum('bct,cs->bst', x, self.weights) 

x_redu [64,64,281] ->
x_reduced = x_reduced.unsqueeze(-2) #x_red.shap = batch, 64, 1, 281: batch, feature, groups,time

x_redu [64,64,1,281] -> 
conv_ = self.nonlin_out()(self.conv(x_reduced)).permute(0, 1, 3, 2)

conv_ [64,64,281,1]
pool [64,64,141,1]
return [64,64,141]


## lang meg data
[MEG-MASC](https://www.nature.com/articles/s41597-023-02752-5): 
high-quality magneto-encephalography dataset for evaluating natural speech processing

Stories from Manually Annotated Sub-Corpus (MASC):
- ‘LW1’: a 861-word story narrating an alien spaceship trying to find its way home (5 min, 20 sec)
- ‘Cable Spool Boy’: a 1,948-word story narrating two young brothers playing in the woods (11 min)
- ‘Easy Money’: a 3,541-word fiction narrating two friends using a magical trick to make money (12 min, 10 sec)
- ‘The Black Willow’: a 4,652-word story narrating the difficulties an author encounters during writing (25 min, 50 sec) 

Within each ∼1 h recording session, participants were recorded with a **208 axial-gradiometer** MEG scanner **sampled** at **1,000 Hz**, and *online band-pass filtered* between *0.01 and 200 Hz* while they listened to four distinct stories through binaural tube earphones (Aero Technologies), at a mean level of 70 dB sound pressure level.

Before the experiment, participants were exposed to **20 sec** of each of the distinct speaker voices used in the study to (i) clarify the structure of the session and (ii) familiarize the participants with these voices

The order in which the four stories were presented was assigned pseudo-randomly, thanks to a “Latin-square design” across participants. The story order for each participant can be found in ‘participants.tsv’. This participant-specific order was used for both recording sessions

paricipants responded, every ∼3 min and with a button press, to a two-alternative forced-choice question relative to the story content (e.g. ‘What precious material had Chuck found? Diamonds or Gold’).

To help decorrelate language features from acoustic representations, we varied both voices and speech rate every 5–20 sentences. Specifically, we used three distinct synthetic voices:‘Ava’, ‘Samantha’ and ‘Allison’ speaking between 145 and 205 words per minute. Additionally, we varied the silence between sentences between 0 and 1,000 ms. Both speech rate and silence duration were sampled from a uniform distribution between the min and max values.

Each story was divided into ~3 min sound files. In between these sounds – approximately every 30 s – we played a random word list generated from the unique content words (nouns, proper nouns, verbs, adverbs and adjectives) selected from the preceding 5 min segment presented in random order. "We decided to include word lists to allow data users to compare brain responses to content words within and outside of context, following experimental paradigms of previous studies"

a very small fraction (<1%) of non-words were inserted into the natural sentences, on average every 30 words. We decided to include non-words to allow comparisons between phonetic sequences that do and do not have an associated meaning.    

following the BIDS labeling45, each “task” corresponds to the concatenation of these sentences and word lists. Each subject listened to the exact same set of four tasks, in a different block order.

Dataset Structure:
![alt text](image.png)

- ‘./dataset_description.json’ describes the dataset

- ‘./participants.tsv’ indicates the age and gender of each participant, the order in which they heard the stories, whether they have an anatomical MRI scan, and how many recording sessions they completed

- ‘./stimuli/’ contains the original texts, the modified texts (i.e. with word lists), the synthesized audio tracks.

- Each’./sub-SXXX’ contains the brain recordings of a unique participant divided by session (e.g.’ses-0’ and’ses-1’)

- In each session folder lies the anatomical and the meg data, and the timestamp annotations (see Fig. 4).

- Sessions are numbered by temporal order (s0 is first; s1 is second).

- Tasks are numbered by a unique integer common across participants.
