
# [Speech Understanding] Programming Assignment 2

This readme file contains detailed explanation of the code provided and instructions on how to run each file for successful reproduction of the reported results.

#### Environment requirements
First, note that in order for the provided code to run successfully, an environment with all the required packages must be installed. Consequently, I have provided a *'package_requirements.txt'* file which contains the name and respective version of all the packages used during this programming assignment and are necessary to reproduce the results.

Second, note that the programming assignment is computationally and storage-wise extensive. This implies that it required strong GPU-resources and large storage (for data, models etc.). Consequently, please run the provided code files on such a system only.

#### Pre-requisites
Now, note that there are data-wise and model-wise pre-requisites of the code. This is listed as follows:

##### Data-wise pre-requisites
Now, for each question the required data must be downloaded and located in the specified path.

- For question 01, two datasets are required. First is the VoxCeleb1-H dataset, which should be downloaded and stored under the path 'PA2/Q1/voxceleb1'. Second is the Kathbath dataset for which 'hindi' language data should be downloaded and stored under tha path 'PA2/Q1/kathbath'.
- For question 02, the required generated data is stored under the path 'PA2/Q2/LibriMix/storage_dir/Libri2Mix/wav16k/max/test'.


##### Model-wise pre-requisites
Now, for each question the required model checkpoints must be downloaded and located in the specific path.

- For question 01, three model checkpoints are required as follows: 'hubert_large', 'wavlm_base_plus', and 'wavlm_large'. The model checkpoints should be downloaded and stored under the path 'PA2/Q1/checkpoints/hubert_large', 'PA2/Q1/checkpoints/wavelmbase+', and 'PA2/Q1/checkpoints/wavelmlarge' respectively.

- For question 02, sepformer is downloaded following the instructions provided in the assignment itself.

Please note that each model of question 01 has it package requirements provided on its official github repository (requirements.txt) which should be loaded for its successful execution.

### Instructions

Now, note the following set of instructions for the successful reproduction of provided results.

#### For question 01

- The chosen three pre-trained models are loaded using the protocol mentioned above in 'Pre-requisites/Model-wise pre-requisites/For question 01'.
- EER (%) can be calculated by running the 'main_vox1celeb.py' file. Please note that for each model, the model_name and model_checkpoint must be provided accordingly.
- Comparison of my results and that reported in the official paper is mentioned in the report.
- EER (%) of selected models on the test set of hindi language of the Kathbath dataset can be calculated by running the 'main_kathbath.py' file. Please note that for each model, the model_name and model_checkpoint must be provided accordingly.
- The fine-tune of the best model can be done by utilizing the official github repository of the model.
- Analysis of results along with plausible reasons for the observed outcomes is mentioned in the report.

#### For question 02

- The steps followed to generate LibriMix dataset is mentioned in the report.
- Evaluation of the performance of the pre-traied SepFormer on the testing set can be done by running the 'main_sepformer.py' file with appropriate paths.
- The fine-tune of the sepformer can be done by utilizing the official github repository of the model.
- Analysis of results along with plausible reasons for the observed outcomes is mentioned in the report.


NOTE: Please contact the author in case of any discrepancy.


