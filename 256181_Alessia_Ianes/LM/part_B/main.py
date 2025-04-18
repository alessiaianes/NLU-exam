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
import math
import seaborn as sns
import time


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


    # Experiment also with a smaller or bigger model by changing hid and emb sizes 
    # A large model tends to overfit
    hid_size = 300 # Hidden size to test
    emb_size = 300 # Embedding size to test
    vocab_len = len(lang.word2id)
    clip = 5 # Clip the gradient
    lr_values = [0.5, 1.0, 1.5, 2, 2.5, 3.0] # Learning rates to test
    batch_sizeT = [32, 64, 128]
    emb_dout = [0.1]
    out_dout = [0.4]

    # Create a directory to save the results, if it doesn't exist
    os.makedirs('results/LSTM_wt_vd/plots', exist_ok=True)

    all_results = []
    total_configurations = len(batch_sizeT) * len(lr_values) * len(emb_dout) * len(out_dout)  # Numero totale di configurazioni
    n_epochs = 100
    total_epochs = total_configurations * n_epochs
    current_configuration = 0  # Contatore per la configurazione corrente
    current_epoch = 0  # Contatore per la configurazione corrente
    script_start_time = time.time()  # Tempo di inizio dell'esecuzione dello script
    initial_eta_printed = False
    



    # Train with differen batch size, emb size, hid size and learning rate
    for bs in batch_sizeT:
        for lr in lr_values:
            for ed in emb_dout:
                for od in out_dout:
                    config_start_time = time.time()  # Tempo di inizio dell'esecuzione
                    print(f"Starting run #{current_configuration + 1}/{total_configurations}")

                    print(f"Training with batch size: {bs}, lr {lr}, emb_dropout {ed}, out_dropout {od}")
                    # Define the collate function
                    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),  shuffle=True)
                    dev_loader = DataLoader(dev_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
                    test_loader = DataLoader(test_dataset, batch_size=bs*2, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
                    
                    # Initialize the model
                    model = LM_LSTM_wt_vd(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"], n_layers=1, emb_dropout=ed, out_dropout=od).to(DEVICE)
                    model.apply(init_weights)

                    optimizer = optim.SGD(model.parameters(), lr=lr)
                    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
                    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')


                    
                    x_min, x_max = 0, n_epochs  # Limiti per l'asse x (epoche)
                    ppl_min, ppl_max = 0, 500  # Limiti per l'asse y (PPL)
                    loss_min, loss_max = 0, 10  # Limiti per l'asse y (Loss)
                    patience = 3
                    losses_train = []
                    losses_dev = []
                    sampled_epochs = []
                    best_ppl = math.inf
                    best_model = None
                    pbar = tqdm(range(1,n_epochs))
                    
                    ppl_values = []


                    elapsed_time = time.time() - config_start_time

                    # Calculate the average time per configuration
                    avg_config_time = elapsed_time / (current_configuration + 1) if current_configuration > 0 else 0

                    # Calculate the remaining configurations
                    remaining_configurations = total_configurations - (current_configuration + 1)
                    remaining_time_total = avg_config_time * remaining_configurations

                    # Convert the remaining time to hours, minutes, and seconds
                    remaining_hours = int(remaining_time_total // 3600)
                    remaining_minutes = int((remaining_time_total % 3600) // 60)
                    remaining_seconds = int(remaining_time_total % 60)

                    print(f"ETA: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s")

                    # elapsed_time = time.time() - start_time

                    # # Calculate the average time per configuration
                    # avg_config_time = elapsed_time / (current_configuration + 1) if current_configuration > 0 else 0

                    # # Calculate the remaining configurations
                    # remaining_configurations = total_configurations - (current_configuration + 1)
                    # remaining_time_total = avg_config_time * remaining_configurations

                    # # Convert the remaining time to hours, minutes, and seconds
                    # remaining_hours = int(remaining_time_total // 3600)
                    # remaining_minutes = int((remaining_time_total % 3600) // 60)
                    # remaining_seconds = int(remaining_time_total % 60)

                    # print(f"ETA: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s")

                
                    #If the PPL is too high try to change the learning rate
                    for epoch in pbar:
                        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)    
                        if epoch % 1 == 0:
                            epoch_start_time = time.time()
                            sampled_epochs.append(epoch)
                            losses_train.append(np.asarray(loss).mean())
                            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                            losses_dev.append(np.asarray(loss_dev).mean())
                            ppl_values.append(ppl_dev) # add PPL to list

                            # current_epoch += 1  # Incrementa il contatore delle epoche
                            # elapsed_time = time.time() - start_time

                            # # Calcola il tempo medio per epoca
                            # avg_epoch_time = elapsed_time / current_epoch if current_epoch > 0 else 0

                            # # Calcola il tempo rimanente per tutte le epoche
                            # remaining_epochs = total_epochs - current_epoch
                            # remaining_time_total = avg_epoch_time * remaining_epochs

                            # # Converti il tempo rimanente in ore, minuti e secondi
                            # remaining_hours = int(remaining_time_total // 3600)
                            # remaining_minutes = int((remaining_time_total % 3600) // 60)
                            # remaining_seconds = int(remaining_time_total % 60)

                            # epoch_duration = time.time() - epoch_start_time
                            current_epoch += 1  # Incrementa il contatore delle epoche
                            elapsed_time = time.time() - config_start_time

                            # Calcola il tempo medio per epoca
                            avg_epoch_time = elapsed_time / current_epoch if current_epoch > 0 else 0

                            # Calcola il tempo rimanente per tutte le epoche
                            remaining_epochs = total_epochs - current_epoch
                            remaining_time_total = avg_epoch_time * remaining_epochs

                            # Converti il tempo rimanente in ore, minuti e secondi
                            remaining_hours = int(remaining_time_total // 3600)
                            remaining_minutes = int((remaining_time_total % 3600) // 60)
                            remaining_seconds = int(remaining_time_total % 60)

                            # Aggiorna la progress bar con il tempo stimato


                            # Aggiorna la progress bar con il tempo stimato
                            pbar.set_description(
                                f"Epoch: {epoch} PPL: {ppl_dev:.2f} ETA: {remaining_hours}h {remaining_minutes}m {remaining_seconds}s"
                            )

                            if  ppl_dev < best_ppl: # the lower, the better
                                best_ppl = ppl_dev
                                best_model = copy.deepcopy(model).to('cpu')
                                patience = 3
                            else:
                                patience -= 1
                            
                        if patience <= 0: # Early stopping triggered at epoch {epoch} for lr={lr}, bs={bs}, emb_dout={ed}, out_dout={od}")
                            break # Not nice but it keeps the code clean

                    best_model.to(DEVICE)
                    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
                    # Inside the loop, append results:
                    all_results.append({
                        'Batch Size': bs,
                        'Learning Rate': lr,
                        'Test PPL': final_ppl
                    })    
                    print(f'Test ppl for batch size {bs}, learning rate {lr}, embedding dropout {ed}, output dropout {od}: {final_ppl}')


                    # Save the results in a CSV file
                    results_df = pd.DataFrame({
                        'Epoch': sampled_epochs,
                        'PPL': ppl_values,
                        'Test PPL': [final_ppl] * len(sampled_epochs)
                    })
                    csv_filename = f'results/LSTM_wt_vd/LSTM_ppl_results_lr_{lr}_bs_{bs}_ed_{ed}_od_{od}.csv'
                    results_df.to_csv(csv_filename, index=False)
                    print(f'CSV file successfully saved in {csv_filename}')
                    

                    # Create ppl_dev plot
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    ax1.plot(sampled_epochs, ppl_values, label='PPL Dev', color='red')
                    ax1.set_title(f'PPL Dev for lr={lr}, bs={bs}, ed={ed}, od={od}')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('PPL')
                    ax1.set_xlim(x_min, x_max)
                    ax1.set_ylim(ppl_min, ppl_max)
                    ax1.legend()
                    ax1.grid()

                    # Save ppl_dev plot
                    ppl_plot_filename = f'results/LSTM_wt_vd/plots/LSTM_ppl_plot_lr_{lr}_bs_{bs}_ed_{ed}_od_{od}.png'
                    plt.savefig(ppl_plot_filename)
                    plt.close(fig)
                    print(f"PPL plot saved: '{ppl_plot_filename}'")

                    # Create the loss plot
                    fig, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(sampled_epochs, losses_train, label='Train Loss', color='orange')
                    ax2.plot(sampled_epochs, losses_dev, label='Dev Loss', color='blue')
                    ax2.set_title(f'Train and Dev Loss for lr={lr}, bs={bs}, ed={ed}, od={od}')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.set_xlim(x_min, x_max)
                    ax2.set_ylim(loss_min, loss_max)
                    ax2.legend()
                    ax2.grid()

                    # Save loss plot
                    loss_plot_filename = f'results/LSTM_wt_vd/plots/LSTM_loss_plot_lr_{lr}_bs_{bs}_ed_{ed}_od_{od}.png'
                    plt.savefig(loss_plot_filename)
                    plt.close(fig)

                    current_configuration += 1  # Incrementa il contatore delle configurazioni
                    print(f"Ending run #{current_configuration}/{total_configurations}")



    pivot_table = pd.DataFrame(all_results).pivot_table(
        values='Test PPL',
        index='Batch Size',  # Rows: Batch Size
        columns='Learning Rate'  # Columns: Learning Rate
    )

    # Visualize the results with a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Test PPL'})
    plt.title("Heatmap of Final PPL for Different Configurations")
    plt.xlabel("Learning Rate")
    plt.ylabel("Batch Size")
    plt.tight_layout()

    # Save the heatmap
    heatmap_filename = 'results/LSTM_wt_vd/plots/heatmap_final_ppl.png'
    plt.savefig(heatmap_filename)
    plt.close()
    print(f"Heatmap saved: '{heatmap_filename}'")


    pd.DataFrame(all_results).to_csv('results/LSTM_wt_vd/all_results.csv', index=False)
    print(f'All results successfully saved in results/LSTM_wt_vd/all_results.csv')

    # After the loops, find the best configuration:
    best_result = min(all_results, key=lambda x: x['Test PPL'])
    print(f"Best configuration: {best_result}")
    best_result_df = pd.DataFrame([best_result])
    best_result_df.to_csv('results/LSTM_wt_vd/best_configuration.csv', index=False)
    print(f'Best configuration successfully saved in results/LSTM_wt_vd/best_configuration.csv')


    
    # To save the model
    # path = 'model_bin/model_name.pt'
    # torch.save(model.state_dict(), path)
    # To load the model you need to initialize it
    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
    # Then you load it
    # model.load_state_dict(torch.load(path))
