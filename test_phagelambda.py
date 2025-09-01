import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn.functional as F
from einops import rearrange
from src.megaDNA import MEGADNA
from src.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from src.dataset.FastaDataset import FastaDataset
import tempfile

# Funzione per salvare i valori su file
def save_values_to_file(data, file_name, normalization_factor=None):
    df = pd.DataFrame(data, columns=["gene_name", "value", "is_essential", "normalized_value"])
    if normalization_factor is not None:
        df["normalized_value"] = df["value"] / normalization_factor
    df.to_csv(file_name, index=False)

# Carica il modello MEGADNA pre-addestrato
def load_model(model_path, tokenizer):
    device = torch.device('cpu')
    model = MEGADNA(
        num_tokens=8,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=tokenizer.pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Usa map_location per caricare su CPU
    return model, device

# Funzione per calcolare la loss gestendo il padding
def calculate_loss(model, sequence, device):
    model.eval()
    # Controlla se sequence è una lista e convertila in un tensore se necessario
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    sequence = sequence.to(device)
    with torch.no_grad():
        loss = model(sequence, return_value='loss')
    return loss.item()

def calculate_likelihood(model, sequence, device):
    model.eval()
    
    # Se `sequence` è una lista, converti in tensore
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    
    sequence = sequence.to(device)
    
    with torch.no_grad():
        # Ottenere logits dal modello
        logits, logits_trunk, start_tokens, ids = model(sequence, return_value="logits")

        # Concatenare i token iniziali con i logits
        logits = torch.cat((start_tokens, logits), dim=-2)

        # Applicare softmax per ottenere le likelihood
        likelihoods = torch.nn.functional.softmax(logits, dim=-1)

        likelihoods = likelihoods[:, :sequence.shape[1], :]
        # Crea una maschera per identificare le posizioni non di padding
        mask = (sequence != 0).int()

        # Ottieni le likelihood dei nucleotidi corretti nella sequenza
        likelihoods_of_true_nucleotides = likelihoods[
            torch.arange(likelihoods.size(0)).unsqueeze(1),  # batch size range
            torch.arange(likelihoods.size(1)),               # sequence length range
            sequence                                          # valori target della sequenza
        ]
        
        # Maschera le likelihood per escludere i valori di padding
        masked_likelihoods = likelihoods_of_true_nucleotides * mask
        
        # Somma delle likelihoods mascherate per ciascuna sequenza nel batch
        sum_likelihoods = masked_likelihoods.sum(dim=1)
        
        # Lunghezza reale della sequenza senza padding
        true_seq_len = mask.sum(dim=1)  # Conta i non-pad token per ciascuna sequenza

        # Calcola la likelihood media per ogni sequenza
        avg_likelihoods = sum_likelihoods / true_seq_len
        
        # Restituisci la likelihood media per ciascuna sequenza
        return avg_likelihoods.mean().item()

# Funzione per calcolare la likelihood
# Funzione per calcolare la likelihood
def calculate_log_likelihood(model, sequence, device):
    model.eval()
    
    # Se `sequence` è una lista, converti in tensore
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    
    sequence = sequence.to(device)
    
    with torch.no_grad():
        # Ottenere logits dal modello
        logits, logits_trunk, start_tokens, ids = model(sequence, return_value="logits")
        logits = torch.cat((start_tokens, logits), dim=-2)
        likelihoods = torch.nn.functional.softmax(logits, dim=-1)
        likelihoods = likelihoods[:, :sequence.shape[1], :]
        mask = (sequence != 0).int()
        likelihoods_of_true_nucleotides = likelihoods[
            torch.arange(likelihoods.size(0)).unsqueeze(1),  # batch size range
            torch.arange(likelihoods.size(1)),               # sequence length range
            sequence                                          # valori target della sequenza
        ]
        log_likelihoods = torch.log(likelihoods_of_true_nucleotides + 1e-10)  # Evita log(0)
        masked_likelihoods = log_likelihoods * mask
        sum_likelihoods = masked_likelihoods.sum(dim=1)
        true_seq_len = mask.sum(dim=1)  # Conta i non-pad token per ciascuna sequenza
        avg_likelihoods = sum_likelihoods / true_seq_len
        return avg_likelihoods.mean().item()

# Funzione per creare il dataset di sequenze
def create_dataset(sequence, tokenizer, mask_prob=0.15):
    # Usa un file temporaneo per salvare la sequenza knock-out in formato FASTA
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as tmp_fasta:
        # Crea un oggetto SeqRecord
        record = SeqRecord(sequence, id="knocked_out_sequence", description="Knock-out gene sequence")
        SeqIO.write(record, tmp_fasta, "fasta")
        fasta_path = tmp_fasta.name
    
    # Crea il dataset FastaDataset utilizzando il file temporaneo
    fasta_data = FastaDataset(fasta_path, tokenizer, mask_prob=mask_prob)
    dataloader = DataLoader(fasta_data, batch_size=1, shuffle=False)
    
    # Rimuovi il file temporaneo dopo averlo caricato nel DataLoader
    os.remove(fasta_path)
    
    return dataloader

# Funzione per knockout del gene
def knockout_gene(sequence, gene_start, gene_end):
    return sequence[:gene_start] + sequence[gene_end:]

# Funzione per caricare le annotazioni dei geni dal file CSV
def load_annotations(csv_file):
    annotations = pd.read_csv(csv_file)
    # Converti la colonna 'essential' in un valore booleano
    annotations['essential'] = annotations['essential'].apply(lambda x: True if x == 'essential' else False)
    return annotations

# Funzione per caricare il genoma dal file FASTA
def load_genome(fasta_file):
    record = SeqIO.read(fasta_file, "fasta")
    return record

# Funzione per creare la cartella di output se non esiste
def create_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def plot_distribution(data, metric_name, file_name):
    essential = [x[1] for x in data if x[2]]
    non_essential = [x[1] for x in data if not x[2]]
    print(f'Number of essentials: {len(essential)}')
    print(f'Number of non essentials: {len(non_essential)}')

    # Crea un DataFrame per l'uso con Seaborn boxplot
    df = pd.DataFrame({
        metric_name: essential + non_essential,
        'Category': ['Essential'] * len(essential) + ['Non-essential'] * len(non_essential)
    })

    # Boxplot delle distribuzioni
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Category', y=metric_name, data=df)
    plt.ylabel(metric_name)
    plt.title(f"Distribution of {metric_name}")
    plt.savefig(file_name)
    plt.clf()

def plot_roc(data, metric_name, file_name, threshold_file):
    y_true = [1 if x[2] else 0 for x in data]
    #y_scores = [-x[3] if metric_name == "Normalized Likelihood" else x[3] for x in data]
    y_scores = [x[3] for x in data]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    optimal_threshold = thresholds[optimal_idx]
    with open(threshold_file, 'w') as f:
        f.write(f"Optimal threshold for {metric_name}: {optimal_threshold}\n")
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label='Optimal Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {metric_name}')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.clf()


from scipy.stats import mannwhitneyu

def mann_whitney_test(data):
    essential = [x[1] for x in data if x[2]]  # Valori per i geni essenziali
    non_essential = [x[1] for x in data if not x[2]]  # Valori per i geni non essenziali
    stat, p_value = mannwhitneyu(essential, non_essential)
    return stat, p_value

def plot_boxplot(data, metric_name, file_name):
    essential = [x[3] for x in data if x[2]]
    non_essential = [x[3] for x in data if not x[2]]
    df = pd.DataFrame({
        metric_name: essential + non_essential,
        'Category': ['Essential'] * len(essential) + ['Non-essential'] * len(non_essential)
    })
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Category', y=metric_name, data=df)
    plt.ylabel(metric_name)
    plt.title(f"Distribution of {metric_name}")
    plt.savefig(file_name)
    plt.clf()

def mann_whitney_test(data, metric_name, file_name):
    essential = [x[3] for x in data if x[2]]
    non_essential = [x[3] for x in data if not x[2]]
    stat, p_value = mannwhitneyu(essential, non_essential, alternative='two-sided')
    with open(file_name, 'w') as f:
        f.write(f"Mann-Whitney U test ({metric_name}): Stat = {stat}, p-value = {p_value}\n")


# Main function
def main():
    # Inserisci qui l'id_prova
    id_prova = '16'  # Cambia l'id della prova quando necessario
    
    # Definisci il percorso del modello addestrato e della cartella per salvare i risultati
    model_path = f'./Trained_models/trained_model_prova{id_prova}.pth'
    output_folder = f'./Figures/Prova{id_prova}/test_lambda_loglikelihood/'

    # Crea la cartella di output se non esiste
    create_output_folder(output_folder)

    # Definisci i file di input e output
    genome_file = 'Data/phage_lambda.fasta'
    annotations_file = 'Data/phage_lambda_genes_with_essential.csv'

    tokenizer = Tokenizer()
    
    model, device = load_model(model_path, tokenizer)
    annotations = load_annotations(annotations_file)
    genome = load_genome(genome_file)

    dataloader_wt = create_dataset(genome.seq, tokenizer)
    for sequence_batch in dataloader_wt:
        sequence_batch = sequence_batch[0]
        wild_type_loss = calculate_loss(model, sequence_batch, device)
        wild_type_likelihood = calculate_log_likelihood(model, sequence_batch, device)

    # Salva i valori di riferimento del wild-type
    with open(os.path.join(output_folder, "WTloss.txt"), 'w') as loss_file:
        loss_file.write(f"Wild-type loss: {wild_type_loss}\n")
    with open(os.path.join(output_folder, "WTlikelihood.txt"), 'w') as likelihood_file:
        likelihood_file.write(f"Wild-type likelihood: {wild_type_likelihood}\n")
    
    # Calcolo della loss e likelihood per il wild-type
    losses, likelihoods = [], []
    for _, gene in annotations.iterrows():
        gene_start = int(gene['start'])
        gene_end = int(gene['end'])
        is_essential = gene['essential']
        
        knocked_out_sequence = knockout_gene(genome.seq, gene_start, gene_end)
        
        # Crea un dataset per il modello
        dataloader = create_dataset(knocked_out_sequence, tokenizer)
        for sequence_batch in dataloader:
            # sequence_batch è una lista, quindi lo trasformiamo in tensore
            sequence_batch = sequence_batch[0]  # Prendi il tensore dalla lista
            loss = calculate_loss(model, sequence_batch, device)
            likelihood = calculate_log_likelihood(model, sequence_batch, device)
            losses.append((gene['name'], loss, is_essential, loss / wild_type_loss))
            likelihoods.append((gene['name'], likelihood, is_essential, likelihood / wild_type_likelihood))
    
    # Salva i valori delle loss e delle likelihood
    save_values_to_file(losses, os.path.join(output_folder, "losses.csv"))
    save_values_to_file(likelihoods, os.path.join(output_folder, "likelihoods.csv"))

    # Boxplot per loss e likelihood normalizzati
    plot_boxplot(losses, "Normalized Loss", os.path.join(output_folder, "normalized_loss_boxplot.png"))
    plot_boxplot(likelihoods, "Normalized Likelihood", os.path.join(output_folder, "normalized_likelihood_boxplot.png"))

    # ROC e AUROC per loss e likelihood normalizzati
    plot_roc(losses, "Normalized Loss", os.path.join(output_folder, "roc_normalized_loss.png"), os.path.join(output_folder, "optimal_threshold_loss.txt"))
    plot_roc(likelihoods, "Normalized Likelihood", os.path.join(output_folder, "roc_normalized_likelihood.png"), os.path.join(output_folder, "optimal_threshold_likelihood.txt"))

    # Test di Mann-Whitney U
    # Mann-Whitney U test per distribuzioni di loss e likelihood normalizzati
    mann_whitney_test(losses, "Normalized Loss", os.path.join(output_folder, "mannwhitney_loss.txt"))
    mann_whitney_test(likelihoods, "Normalized Likelihood", os.path.join(output_folder, "mannwhitney_likelihood.txt"))

    
if __name__ == "__main__":
    main()