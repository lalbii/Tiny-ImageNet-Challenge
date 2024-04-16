import torch

def training(dataloader,model,loss_fn,optimizer,batch_size):

    """
    Train the model using the given data loader, loss function, and optimizer.

    Args:
        dataloader (DataLoader): DataLoader containing the training data.
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn: The loss function used for optimization.
        optimizer: The optimization algorithm.
        batch_size (int): The size of each mini-batch.

    Returns:
        float: The average training loss for the epoch.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    train_loss = 0

    model.train() #take the model train mode
    for no_batch, (X,y) in enumerate(dataloader):

        X,y = X.to("cuda"), y.to("cuda") ##To load the data into cuda

        prediction = model(X)  
        loss = loss_fn(prediction,y)   #calcuate the loss respect to the loss fn.
        
        train_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
        if no_batch % 100 == 0:
            loss = loss.item()
            current_data_ =  no_batch * batch_size + len(X) ##currently passed data
            print(f"Current loss: {loss:>7f}  [{current_data_:>5d}/{size:>5d}]")
        
    train_loss /= num_batches
    print(f"Average training loss for epoch: {train_loss:>7f}") 
    return train_loss 

def validation_loss(dataloader,model,loss_fn):
    """
    Calculate validation loss and accuracy for the given data loader.

    Args:
        dataloader (DataLoader): DataLoader containing the validation data.
        model (torch.nn.Module): The neural network model to evaluate.
        loss_fn: The loss function used for evaluation.

    Returns:
        tuple: A tuple containing the validation loss and accuracy in percentage.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    val_loss = 0
    correct_classified = 0
    model.eval()

    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to("cuda"), y.to("cuda")
            prediction = model(X)
            val_loss += loss_fn(prediction, y).item()

            ## to sum the correctly classified ones adding up
            correct_classified_float = (prediction.argmax(1) == y).type(torch.float) #covert from boolen to float to add them.

            correct_classified += correct_classified_float.sum().item()

    val_loss /= num_batches
    correct_classified /= size

    print(f"Val Error: \n Accuracy: {(100*correct_classified):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss,(100*correct_classified)


def training_loop(epochs,train_dataloader,valid_dataloader,model,loss_fn,optimizer,batch_size,patience=5):
    """
    Train the model with early stopping based on validation loss.

    Args:
        epochs (int): Number of epochs for training.
        train_dataloader (DataLoader): DataLoader containing the training data.
        valid_dataloader (DataLoader): DataLoader containing the validation data.
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn: The loss function used for optimization.
        optimizer: The optimization algorithm.
        batch_size (int): The size of each mini-batch.
        patience (int): Number of epochs to wait for improvement in validation loss.

    Returns:
        tuple: A tuple containing arrays of training loss, validation loss, validation accuracy, and the trained model.
    """

    best_val_loss = float('inf')
    num_epochs_without_improvement = 0

    training_loss_array = []
    val_loss_array = []
    val_accuracy = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------")
        training_loss = training(train_dataloader, model, loss_fn, optimizer,batch_size=batch_size)
        training_loss_array.append(training_loss)
        val_loss,accuracy = validation_loss(valid_dataloader, model, loss_fn)
        val_loss_array.append(val_loss)
        val_accuracy.append(accuracy)

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1

        # Check if to early stop
        if num_epochs_without_improvement >= patience:
            print(f'Early stopping: No improvement in validation loss for {patience} epochs.')
            break

    print("Training Finished...")
    return training_loss_array,val_loss_array,val_accuracy,model