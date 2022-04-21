class compare_networks(nn.Module):

    def __init__(self, model_dict_resnet, batch_size, num_classes, config):
        super(compare_networks,self).__init__()
        # Load two models
        # One model is not trained (we set the dict path as None)
        # One model is trained with previous results from trainings (Self supervised).
        self.model_not_pretrained = load_resnet18_with_barlow_weights(None,num_classes).to(config['device'])
        self.model_pretrained = load_resnet18_with_barlow_weights(model_dict_resnet,num_classes).to(config['device'])

        self.batch_size = batch_size


    def train(self, dataset_train, dataset_valid, num_epochs, criterion, config,writer,tag):
        dataloader_train = DataLoader(dataset_train, self.batch_size)
        dataloader_valid = DataLoader(dataset_valid, self.batch_size)

        # Train on pretrained
        optimizer = torch.optim.Adam(self.model_pretrained.parameters(), lr=config["lr"], weight_decay=config["optimizer_weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], verbose=False)

        print(f'Starting the train of pre-trained model.')
        for i in range(num_epochs):
            loss = self.train_one_epoch(self.model_pretrained,dataloader_train,optimizer,criterion,config)
            loss_valid = self.valid_one_epoch(self.model_pretrained,dataloader_valid,config)
            writer.add_scalar(f"Loss/train:pretrained {tag}",loss,i)
            writer.add_scalar(f"Loss/valid:pretrained {tag}",loss_valid,i)
            scheduler.step()
            #print(f'Ended epoch {i+1}')

        # Train on not pretrained
        optimizer = torch.optim.Adam(self.model_not_pretrained.parameters(), lr=config["lr"], weight_decay=config["optimizer_weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], verbose=False)

        print(f'Starting the train of not pre-trained model.')
        for i in range(num_epochs):
            
            loss = self.train_one_epoch(self.model_not_pretrained,dataloader_train,optimizer,criterion,config)
            writer.add_scalar(f"Loss/train:not_pretrained {tag}",loss,i)
            scheduler.step()
            
            loss_valid = self.valid_one_epoch(self.model_not_pretrained,dataloader_valid,config)
            writer.add_scalar(f"Loss/valid:not_pretrained {tag}",loss_valid,i)
            #print(f'Ended epoch {i+1}')

        
    def train_one_epoch(self,model,dataloader_train,optimizer, criterion, config):
        running_loss = 0.0
        logsoft = nn.LogSoftmax()
        epoch_loss = 0.0
        for i, (images,labels) in enumerate(dataloader_train):
            images, labels = images.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()

            # Add a logsoft at the end as activation function because the barlow network does not use it.
            outputs = logsoft(model(images))

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        return running_loss/len(dataloader_train)
    @torch.no_grad()
    def valid_one_epoch(self,model, dataloader_valid, config):
        correct_pred = 0
        total_pred = 0
        logsoft = nn.LogSoftmax()
        for i, (images,labels) in enumerate(dataloader_valid):
            images, labels = images.to(config['device']), labels.to(config['device'])
            outputs = logsoft(model(images))

            _, predicted = torch.max(outputs.data,1)

            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_pred += pred.eq(labels.view_as(pred)).sum().item()
            total_pred += labels.size(0)

        return correct_pred/total_pred
