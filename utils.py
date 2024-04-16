
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import datasets, transforms
import os



def imshow(img):
    """
    Display a single image.

    Args:
        img (Tensor): The image tensor to display.
    """

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

def show_image(dataloader):
    """
    Display a randomly selected image from a data loader batch.

    Args:
        dataloader (DataLoader): The DataLoader containing images and labels.

    Returns:
        tuple: A tuple containing the label and shape of the displayed image.
    """

   

    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    ## random selected image from batch
    random_number = randint(0, len(images)-1)
    
    imshow(images[random_number])
    label = labels[random_number]
    
    print(f'Label: {label}')
    return label ,images[random_number].shape
    

def dataloader_create(data, name, transform, gpu_enable, batch_size):
    """
    Create a DataLoader for the given dataset.

    Args:
        data (str): Path to the dataset directory.
        name (str): Name of the dataset (e.g., 'train', 'valid', 'test').
        transform (callable): Transformations to apply to the dataset.
        gpu_enable (bool): Flag indicating whether GPU acceleration is enabled.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    
    #loading iamges
    dataset = datasets.ImageFolder(data, transform=transform)
        
    # Set options for device
    if gpu_enable:
        pin_memory = True
        num_workers = 2
    else:
        pin_memory = False
        num_workers = 1
    
    # Creating the dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train" or name=="valid" or name=="test"),pin_memory=pin_memory
                        ,num_workers=num_workers)
    
    return dataloader

def check_folder_existence(folder_path):
    return os.path.exists(folder_path) 


def plot_training_and_validation(training_loss_array,val_loss_array,val_accuracy):

    """
    Plot training and validation loss, and validation accuracy.

    Args:
        training_loss_array (list): List of training loss values.
        val_loss_array (list): List of validation loss values.
        val_accuracy (list): List of validation accuracy values.

    Returns:
        None
    """

    epochs = list(range(1, len(training_loss_array) + 1))
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss_array, 'b', label='Training loss')
    plt.plot(epochs, val_loss_array, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(1, len(epochs))
    plt.show()
    
    # Plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracy, 'g', label='Validation accuracy')
    plt.title('Validation Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    pass
