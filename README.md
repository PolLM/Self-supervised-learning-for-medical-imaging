# aidl2022_final_project: Self-supervised learning for medical imaging

Authors: Francesc Garcia, Pol Llopart, Sergio Rodriguez

Advisor: Kevin McGuinness

## Project description and results

The description of the project, toghether with the results obtained from all the experiments is explained in the file **Project_report.pdf **.

## Repository Structure

The repository is structured as follows:

Project_report.pdf - Report of the project including scope and experiment results.
selfsup_task - Folder containing all the experiments related with the hyperparameter search and training of the Barlow Twins model.
augmentations - Folder containing all the augmentations tested and used by the self-supervised model.
dataset_loader - Folder containing one class to load the CheXpert dataset
datasets - Folder where the datasets are saved. Added in .gitignore file to avoid pushing the datasets. It is not mandatory to save them on this folder (one can change the folder path on each experiment).
downstream_task - Folder containing all the experiments related with the supervised training of the pre-trained models.
interpretability- Folder containing the interpretability experiments carried out.
runs - Folder containing logs of all the experiments carried out. For each experiment we saved the config file (with the hyperparameters), the final and best model state_dict and the tensorboard logs. Each state_dict weights around 50Mb, it is for this reason tht we have added them into the .gitignore file.
self_sup_classes- Folder where the BarlowTwins class is defined.
utils - Folder containing useful functions used in other parts of the project.
