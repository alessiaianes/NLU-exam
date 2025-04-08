# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pandas as pd  # for DataFrame
import os

if __name__ == "__main__":
    DEVICE = 'cuda:0' # it can be changed with 'cpu' if you do not have a gpu
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])


    # Create the datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Define the collate function
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))

    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 200
    emb_size = 300
    vocab_len = len(lang.word2id)
    clip = 5 # Clip the gradient
    lr_values = [0.0001, 0.01, 0.1, 1, 1.5]

    # Create a directory to save the results, if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
        if not os.path.exists('RNN'):
            os.makedirs('results/RNN')

    for lr in lr_values:
        print(f"Training with learning rate: {lr}")

        # Initialize the model
        model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    
    
        n_epochs = 100
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1,n_epochs))
        
        ppl_values = []
    
    #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                ppl_values.append(ppl_dev) # add PPL to list
                pbar.set_description("PPL: %f" % ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

        best_model.to(DEVICE)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
        print(f'Test ppl for lr {lr}: {final_ppl}')


   

        # Save the results in a CSV file
        results_df = pd.DataFrame({
            'Epoch': sampled_epochs,
            'PPL': ppl_values
        })
        csv_filename = f'results/RNN/RNN_ppl_results_lr_{lr}.csv'
        results_df.to_csv(csv_filename, index=False)
        print('CSV file successfully saved in {csv_filename}')
        
    
    # To save the model
    # path = 'model_bin/model_name.pt'
    # torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
