import argparse

import numpy as np
import torch
import os
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA  
from src.methods.dummy_methods import DummyClassifier
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

import matplotlib.pyplot as plt
import time

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE
        N = xtrain.shape[0]  
        val_ratio = 0.2

        val_size = int(N * val_ratio)  
        indices = np.arange(N)
        np.random.shuffle(indices)

        # Split index into training set and verification set
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        xtest = xtrain[val_indices]
        ytest = ytrain[val_indices]

        xtrain = xtrain[train_indices]
        ytrain = ytrain[train_indices]

        pass

    ### WRITE YOUR CODE HERE to do any other data processing
    mean_xtrain = np.mean(xtrain, axis = 0, keepdims = True)
    std_xtrain = np.std(xtrain, axis = 0, keepdims = True)

    mean_xtest = np.mean(xtest, axis = 0, keepdims = True)
    std_xtest = np.std(xtest, axis = 0, keepdims = True)

    # normalization
    xtrain = normalize_fn(xtrain, mean_xtrain, std_xtrain)
    xtest = normalize_fn(xtest, mean_xtest, std_xtest)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    s1 = time.time()
    n_classes = get_n_classes(ytrain)
    print("xtrain.shape:", xtrain.shape)
    if args.nn_type == "mlp":
        input_size = xtrain.shape[1]
        model = MLP(input_size, n_classes) 
    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(-1, 28, 28)
        xtest = xtest.reshape(-1, 28, 28)
        input_size = xtrain.shape
        print(input_size)
        model = CNN(input_size, n_classes)
    elif args.nn_type == "transformer":
        xtrain = xtrain.reshape(-1, 28, 28)
        xtest = xtest.reshape(-1, 28, 28)
        input_size = xtrain.shape        
        print("输入:" , input_size)
        model = MyViT(chw = [1,28,28], out_d = n_classes)
    else:
        pass

    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    if args.load_path:
        method_obj.load_model("E:\CS-233 milestone\CS233-Milestone-2\CS233-Milestone-2\src\models\model.pth")
    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    s2 = time.time()

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    if not args.test:
    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        pass

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if args.nn_type == "mlp":
        if args.use_pca:
            print("MLP takes", s2-s1, "seconds.")
        else:
            print("MLP with PCA takes", s2-s1, "seconds.")
    elif args.nn_type == "cnn":
        print("CNN takes", s2-s1, "seconds.")
    elif args.nn_type == "transformer":
        print("Transformer takes", s2-s1, "seconds.")
    else:
        pass

    epochs_range = range(1, args.max_iters + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, method_obj.train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2) 
    plt.plot(epochs_range, method_obj.train_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.show()


    model_save_path = "E:\CS-233 milestone\CS233-Milestone-2\CS233-Milestone-2\src\models\model.pth" 
    model_directory = os.path.dirname(model_save_path)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(f"Created directory {model_directory}")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=5, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--load_path', action='store_true', help="Load the model parameters from the specified file path.")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)