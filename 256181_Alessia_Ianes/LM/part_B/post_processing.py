import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Directory contenente i file CSV
results_dir = 'results/LSTM_weight_tying/'

# Funzione per estrarre i parametri dal nome del file
def extract_params_from_filename(filename):
    # Esempio di nome file: LSTM_ppl_results_lr_0.1_bs_64_emb_300_hid_300.csv
    match = re.search(r'lr_([\d.]+)_bs_(\d+)', filename)
    if not match:
        raise ValueError(f"Nome file non valido: {filename}. Assicurati che contenga 'lr_X_bs_Y'.")
    lr = float(match.group(1))  # Learning Rate
    bs = int(match.group(2))    # Batch Size
    return lr, bs

# Leggi tutti i file CSV nella directory
csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

# Estrai i dati da ogni file e aggiungi i parametri estratti
all_results = []
for csv_file in csv_files:
    file_path = os.path.join(results_dir, csv_file)
    
    # Carica solo la colonna "Final PPL"
    df = pd.read_csv(file_path, usecols=['Test PPL'])
    
    # Estrai Batch Size e Learning Rate dal nome del file
    lr, bs = extract_params_from_filename(csv_file)
    
    # Aggiungi le colonne "Batch Size" e "Learning Rate" al DataFrame
    df['Batch Size'] = bs
    df['Learning Rate'] = lr
    
    all_results.append(df)

# Combina tutti i DataFrame in uno solo
results_df = pd.concat(all_results, ignore_index=True)

# Assicurati che i dati abbiano le colonne necessarie
required_columns = {'Batch Size', 'Learning Rate', 'Test PPL'}
if not required_columns.issubset(results_df.columns):
    raise ValueError(f"Il DataFrame deve contenere le seguenti colonne: {required_columns}")

# Aggrega i risultati per configurazione
# Qui usiamo il valore minimo di Final PPL per ogni configurazione
aggregated_results = results_df.groupby(['Batch Size', 'Learning Rate'], as_index=False).agg({
    'Test PPL': 'min'  # Puoi cambiare 'min' con 'mean' o 'max' se preferisci
})

# Creiamo una pivot table per la heatmap
pivot_table = aggregated_results.pivot_table(
    values='Test PPL',
    index='Batch Size',  # Righe: Batch Size
    columns='Learning Rate'  # Colonne: Learning Rate
)

# Visualizziamo la heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Final PPL'})
plt.title("Heatmap of Final PPL for Different Configurations")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.tight_layout()

# Salviamo la heatmap come immagine
heatmap_filename = os.path.join(results_dir, 'plots/heatmap_final_ppl.png')
os.makedirs(os.path.dirname(heatmap_filename), exist_ok=True)  # Crea la cartella "plots" se non esiste
plt.savefig(heatmap_filename)
plt.close()
print(f"Heatmap saved: '{heatmap_filename}'")

# Troviamo la migliore configurazione
best_result = aggregated_results.loc[aggregated_results['Test PPL'].idxmin()]
print(f"Best configuration: {best_result.to_dict()}")

# Salviamo la migliore configurazione in un file CSV
best_result_df = pd.DataFrame([best_result])
best_config_filename = os.path.join(results_dir, 'best_configuration.csv')
best_result_df.to_csv(best_config_filename, index=False)
print(f'Best configuration successfully saved in {best_config_filename}')