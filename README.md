# aidl2022_final_project: Self-supervised learning for medical imaging

Authors: Francesc Garcia, Pol Llopart, Sergio Rodriguez

Advisor: Kevin McGuinness

## Project description and results

The description of the project, toghether with the results obtained from all the experiments is explained in the file **Project_report.pdf**.

## Repository Structure

The repository is structured as follows:

- Project_report.pdf: Report of the project including scope and experiment results.

- selfsup_task: Folder containing all the experiments related with the hyperparameter search and training of the Barlow Twins model.
  - check_bt_resources.py:
  - scan_optimal_lr.py:
  - scan_best_transformations.py:
  - scan_best_hyperparametrs.py
  - hyper_utils.py:
  - train_barlow_twins.py:
  
- augmentations: Folder containing all the augmentations tested and used by the self-supervised model.
  - transform_utils.py:
  - view_generator.py:
  
- dataset_loader: Folder containing one class to load the CheXpert dataset
  - CheXpertDataset.py:
  
- datasets: Folder where the datasets are saved. Added in .gitignore file to avoid pushing the datasets. It is not mandatory to save them on this folder (one can change the folder path on each experiment).

- downstream_task: Folder containing all the experiments related with the supervised training of the pre-trained models.
  - scan_best_supervised_hyperparameters: 
  - compare_networks.py: TODO
  - ConfusionMatrix: TODO
  - trained_vs_no_trained_compare: TODO
  
- interpretability: Folder containing the interpretability experiments carried out.
  - Grad-CAM.py: TODO
  
- runs: Folder containing logs of all the experiments carried out. For each experiment we saved the config file (with the hyperparameters), the final and best model state_dict and the tensorboard logs. Each state_dict weights around 50Mb, it is for this reason tht we have added them into the .gitignore file.

- self_sup_classes: Folder where the BarlowTwins class is defined.
  - barlow.py: 
 
- utils: Folder containing useful functions used in other parts of the project.
  - training_utils.py:
  - metrics.py:
  - logging_utils.py:
  - load_tb_logs.py:
