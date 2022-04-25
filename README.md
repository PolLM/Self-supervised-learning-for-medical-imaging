# aidl2022_final_project: Self-supervised learning for medical imaging

Authors: Francesc Garcia, Pol Llopart, Sergio Rodriguez

Advisor: Kevin McGuinness

## Project description and results

The description of the project, toghether with the results obtained from all the experiments is explained in the file **Project_report.pdf**.

## Repository Structure

The repository is structured as follows:

_Entry points of the project in **bold font**._

- Project_report.pdf: Report of the project including scope and experiment results.

- selfsup_task: Folder containing all the experiments related with the hyperparameter search and training of the Barlow Twins model.
  - **check_bt_resources.py**: Find the Barlow Twins model execution time under several configurations. 
  - **scan_optimal_lr.py**: Scans optimal lr and compares lr schedulers.
  - **scan_best_transformations.py**: Scans different combinations of transformations.
  - **scan_best_hyperparametrs.py**: Scans different combinations of Barlow Twins Lambda and projector head.
  - hyper_utils.py: Utils related with the self-supervised tasks.
  - **train_barlow_twins.py**: Final self-supervised training.
  
- augmentations: Folder containing all the augmentations tested and used by the self-supervised model.
  - transform_utils.py: Contains all transformations not implemented in torchvison.
  
- dataset_loader: Folder containing functions to load the datasets.
  - CheXpertDataset.py: Containing one class to load the CheXpert dataset
  
- datasets: Folder where the datasets are saved. Added in .gitignore file to avoid pushing the datasets. It is not mandatory to save them on this folder (one can change the folder path on each experiment).
  - Covid dataset:
    - Download LINK: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
    - 'Mask' folders must be removed manually before using it, or the mask images will be used as x-ray images and distort the results.
  - CheXpert dataset:
    - Download LINK: https://stanfordmlgroup.github.io/competitions/chexpert/ (Requires registration)
    - Recommendation to download the _small_ version. The project will resize all the images regardless of the original resolution.
    - The CheXpert dataset contains different kinds of pictures. For that reason, it is required to filter out all the lateral images (Only use the ones marked as 'Frontal' to be able to reproduce the same results as the original work. Recommendation to generate new csv with only the images labeled as 'Frontal'.

- downstream_task: Folder containing all the experiments related with the supervised training of the pre-trained models.
  - **scan_best_supervised_hyperparameters**: Scans optimal parameters for the supervised training.
  - **compare_networks.py**: Containing one class that generates two resnet models: One pretrained (Dict is provided) and one non pre-trained. Allows to train and validate both models sequentially and log into a TensorBoard writer the results of loss/accuracy.
  - **ConfusionMatrix**: Outputs confusion matrix from model predictions.
  - **trained_vs_no_trained_compare**: Executes multiple trainings with different number of samples given a state dict and using the class 'compare_networks0.
  - **train_models_with_frozen_parts.py**: Executes the code to freeze and train models given a state dict. Outputs the loss/accuracy values for every epoch in terminal, and logs them into a TensorBoard writer.
  
- interpretability: Folder containing the interpretability experiments carried out.
  - **Grad-CAM.py**: Applies Grad-CAM to interpret our models. Takes a state dict and an image as inputs.
  
- runs: Folder containing logs of all the experiments carried out. For each experiment we saved the config file (with the hyperparameters), the final and best model state_dict and the tensorboard logs. Each state_dict weights around 50Mb, it is for this reason tht we have added them into the .gitignore file.
  - The pre-trained Resnet-18 weights from the final trainings can be downloaded here: https://drive.google.com/drive/folders/1xQ-mKPM8B-XJI1DcMh4AKMwSZIMYsA1w?usp=sharing

- self_sup_classes: Folder where the BarlowTwins class is defined.
  - barlow.py: Main class to implement Barlow Twins architecture.
 
- utils: Folder containing useful functions used in other parts of the project.
  - training_utils.py: Functions used when training our models.
  - metrics.py: Functions used to compute evaluation metrics.
  - logging_utils.py: Functions used for logging.
  - load_tb_logs.py: Functions used to extract tensorboard data.
