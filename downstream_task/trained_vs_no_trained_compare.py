model_dict = [...] # Path to the file containing the state dict to be used for the analysis.
batch_size = 128
num_classes = 4

# Modify these two variables to contain all the labels to be scanned.
ListOfCompares = [0.01,0.05,0.10,0.15,0.20,0.25]
ListOfTags=      ["1 percent of samples",
                    "5 percent of samples",
                    "10 percent of samples",
                    "15 percent of samples",
                    "20 percent of samples",
                    "25 percent of samples"]

ListOfDicts_barlow_1 = [None,None,None,None,None,None]
ListOfDicts_barlow = [None,None,None,None,None,None]
ListOfDicts_resnet_1 = [None,None,None,None,None,None]
ListOfDicts_resnet = [None,None,None,None,None,None]

config = {
    'project_path': [...] # Update with the base path of your project.
    'model_path': [...] # Update with the base path of your project.
    'random_seed': 73, 
    'device': torch.device(type='cuda'), 
    'img_res': 224, 
    'num_classes': 4, 
    'train_frac': 0.8, 
    'test_frac': 0.1, 
    'val_frac': 0.1, 
    'transforms_prob': 0.5, 
    'h_flip': 1, 
    'batch_size_sup': 96, 
    'optimizer': 0, 
    'optimizer_weight_decay': 9.870257603212248e-06, 
    'soft_crop': 0, 
    'lr_sup': 0.0015294613483158384, 
    'num_epochs_sup': 15
    }


transform = transforms.Compose([
    
    transforms.Grayscale(),
    transforms.Resize(224),
    # you can add other transformations in this list
    transforms.ToTensor()
])

dataset = ImageFolder("/content/COVID-19_Radiography_Dataset/",transform)


val_len = int(len(dataset)*config["val_frac"])
train_len = int(len(dataset)*config["train_frac"])
test_len = int(len(dataset)-val_len - train_len)
train_dataset, val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len,test_len])

# Update with your log directory
writer = SummaryWriter(log_dir="[...]",comment="Compare trained vs untrained without logsoftmax")

print(f'Total samples: {len(train_dataset)} as Training dataset')
print(f'Total samples: {len(val_dataset)} as Validation dataset')
for idx,perc in enumerate(ListOfCompares):
    
    comparison = compare_networks(model_dict,batch_size,num_classes,config)
    tr_split_len = math.floor(len(train_dataset) * perc)
    print(f'Starting loop {idx}')
    print(f'Taking {perc*100} percent of samples: {tr_split_len}')
    dataset_reduced = torch.utils.data.random_split(train_dataset, [tr_split_len, len(train_dataset)-tr_split_len])[0]
    criterion = nn.CrossEntropyLoss()
    tag = ListOfTags[idx]
    comparison.train(dataset_reduced, val_dataset, config['num_epochs_sup'],criterion,config,writer,tag)    
    ListOfDicts_barlow[idx], ListOfDicts_resnet[idx], ListOfDicts_barlow_1[idx], ListOfDicts_resnet_1[idx] = comparison.get_dicts()

base_path = [...] # Update with the path to save the state dict.

save_checkpoint(ListOfDicts_barlow[0],basepath=base_path,filename='Barlow_15_epocs_1perc.pt')
save_checkpoint(ListOfDicts_barlow[1],basepath=base_path,filename='Barlow_15_epocs_5perc.pt')
save_checkpoint(ListOfDicts_barlow[2],basepath=base_path,filename='Barlow_15_epocs_10perc.pt')
save_checkpoint(ListOfDicts_barlow[3],basepath=base_path,filename='Barlow_15_epocs_15perc.pt')
save_checkpoint(ListOfDicts_barlow[4],basepath=base_path,filename='Barlow_15_epocs_20perc.pt')
save_checkpoint(ListOfDicts_barlow[5],basepath=base_path,filename='Barlow_15_epocs_25perc.pt')


save_checkpoint(ListOfDicts_barlow_1[0],basepath=base_path,filename='Barlow_1_epocs_1perc.pt')
save_checkpoint(ListOfDicts_barlow_1[1],basepath=base_path,filename='Barlow_1_epocs_5perc.pt')
save_checkpoint(ListOfDicts_barlow_1[2],basepath=base_path,filename='Barlow_1_epocs_10perc.pt')
save_checkpoint(ListOfDicts_barlow_1[3],basepath=base_path,filename='Barlow_1_epocs_15perc.pt')
save_checkpoint(ListOfDicts_barlow_1[4],basepath=base_path,filename='Barlow_1_epocs_20perc.pt')
save_checkpoint(ListOfDicts_barlow_1[5],basepath=base_path,filename='Barlow_1_epocs_25perc.pt')

save_checkpoint(ListOfDicts_resnet[0],basepath=base_path,filename='resnet_15_epocs_1perc.pt')
save_checkpoint(ListOfDicts_resnet[1],basepath=base_path,filename='resnet_15_epocs_5perc.pt')
save_checkpoint(ListOfDicts_resnet[2],basepath=base_path,filename='resnet_15_epocs_10perc.pt')
save_checkpoint(ListOfDicts_resnet[3],basepath=base_path,filename='resnet_15_epocs_15perc.pt')
save_checkpoint(ListOfDicts_resnet[4],basepath=base_path,filename='resnet_15_epocs_20perc.pt')
save_checkpoint(ListOfDicts_resnet[5],basepath=base_path,filename='resnet_15_epocs_25perc.pt')

save_checkpoint(ListOfDicts_resnet_1[0],basepath=base_path,filename='resnet_1_epocs_1perc.pt')
save_checkpoint(ListOfDicts_resnet_1[1],basepath=base_path,filename='resnet_1_epocs_5perc.pt')
save_checkpoint(ListOfDicts_resnet_1[2],basepath=base_path,filename='resnet_1_epocs_10perc.pt')
save_checkpoint(ListOfDicts_resnet_1[3],basepath=base_path,filename='resnet_1_epocs_15perc.pt')
save_checkpoint(ListOfDicts_resnet_1[4],basepath=base_path,filename='resnet_1_epocs_20perc.pt')
save_checkpoint(ListOfDicts_resnet_1[5],basepath=base_path,filename='resnet_1_epocs_25perc.pt')


writer.flush()
writer.close()
