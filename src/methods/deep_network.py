import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        self.fc1 = nn.Linear(in_features=input_size, out_features=256, bias=True)
        self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc4 = nn.Linear(in_features=64, out_features=n_classes, bias=True)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        preds = self.fc4(x)      
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        x = x.unsqueeze(1) # add the channel
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        preds = self.fc2(x)
        return preds


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                # Map seq to q, k, v.
                q = q_mapping(seq)
                k = k_mapping(seq)
                v = v_mapping(seq) ### WRITE YOUR CODE HERE
                
                ### WRITE YOUR CODE HERE
                attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
                attention = self.softmax(attention_scores)

                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads) 
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential( 
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # Write code for MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x)) #
        # Write code for MLP(Norm(out)) + residual connection
        out = out + self.mlp(self.norm2(out))
        return out
class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw , n_patches = 7, n_blocks = 6, hidden_d = 64, n_heads = 16, out_d= 10):
        """
        Initialize the network.
        
        """
        super().__init__()
        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.out_d = out_d
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)
    
        assert chw[1] % self.patch_size[0] == 0 and chw[2] % self.patch_size[0] == 0, "Image dimensions must be divisible by the patch size."
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        n_tokens = self.n_patches * self.n_patches + 1
        self.positional_embeddings = nn.Parameter(torch.rand(1, n_tokens, self.hidden_d))

        self.blocks_1 = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )
    def patchify(self, images, n_patches):
        # n, c, h, w = images.shape
        # assert h == w 
        print(images.shape)
        n = images.shape[0] 
        c = images.shape[1]  
        h = images.shape[2]
        w = images.shape[3]

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches
        
        for i in range(n_patches):
            for j in range(n_patches):
                patch = images[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patches[:, i * n_patches + j] = patch.reshape(n, -1)
        
        return patches
        # patch_dim = chw[1] * self.patch_size[0] * self.patch_size[1]
        
        
        
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(chw[1], hidden_d, kernel_size=self.patch_size, stride=self.patch_size),
        #     nn.Flatten(2),
        #     nn.Linear(patch_dim, hidden_d),
        # )
        
        # self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, hidden_d))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_d))
        
        # self.blocks_2 = nn.Transformer(
        #     d_model=self.hidden_d,
        #     nhead=self.n_heads,
        #     num_encoder_layers=self.n_blocks,
        #     num_decoder_layers=0,
        #     dim_feedforward=hidden_d * 4,
        #     activation='gelu'
        # )
        
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(hidden_d),
        #     nn.Linear(hidden_d, out_d)
        # )

    def forward(self, images):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        # n, chw = images.shape
        images = images.unsqueeze(1) # add the channel
        n = images.shape[0] 

        # Divide images into patches.
        patches = self.patchify(images, self.n_patches)
        print("Shape of patches:", patches.shape) 
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        positional_embeddings = self.positional_embeddings.repeat(n, 1, 1)
        out = tokens + positional_embeddings### WRITE YOUR CODE HERE

        # Transformer Blocks
        for block in self.blocks_1:
            out = block(out)
        
        # for block in self.blocks_2:
        #     out = block(out)

        
        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp_head(out)

        return out

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr) ### WRITE YOUR CODE HERE

        self.train_losses = []
        self.train_accuracies = []

    def load_model(self, save_path):
        """
        Load the model parameters from the specified file path.
        """
        self.save_path = save_path
        self.model.load_state_dict(torch.load(self.save_path))
        print(f"Loaded model parameters from {self.save_path}")

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            train_loss, train_accuracy = self.train_one_epoch(dataloader, ep)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.train()
        total_loss = 0
        correct = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {ep+1}/{self.epochs}')
        for batch, (x, y) in enumerate(progress_bar):
            # Compute prediction and loss
            pred = self.model(x)
            loss = self.criterion(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataloader.dataset)
        return avg_loss, accuracy

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        pred = []
        self.model.eval()
        with torch.no_grad():
            for x in dataloader:
                _, label = torch.max(self.model(x[0]), 1)
                pred.extend(label.tolist())
        pred_labels = torch.tensor(pred)
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()