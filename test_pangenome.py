import os
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
from torch.utils.data import DataLoader
from src.megaDNA import MEGADNA
from src.tokenizer import Tokenizer
from src.dataset.FastaDataset import FastaDataset
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt

# Annotate Genomes with Prokka
def annotate_sequences_with_prokka(fasta_file, output_folder, sequence_id):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    command = ["prokka", fasta_file, "--outdir", output_folder, "--prefix", f"annotated_{sequence_id}", "--force"]
    subprocess.run(command, check=True)

import re

def parse_clustered_proteins(clustered_proteins_file):
    """Parse the clustered_proteins file to get group frequencies and gene-to-group mappings."""
    group_frequencies = {}
    gene_to_group = {}

    with open(clustered_proteins_file, 'r') as f:
        for line in f:
            # Usa regex per gestire sia tabulazioni che spazi come separatori
            parts = re.split(r'[\t\s]+', line.strip())  # Split on tabs or spaces
            group = parts[0].rstrip(':')  # Rimuove i due punti dal gruppo
            genes = [gene.strip() for gene in parts[1:] if gene]  # Assicura che non ci siano spazi vuoti

            # Incrementa il conteggio del gruppo
            group_frequencies[group] = len(genes)

            # Associa ogni gene al gruppo
            for gene in genes:
                gene_to_group[gene] = group

    return group_frequencies, gene_to_group



# Run Roary to get the pangenome
def run_roary(gff_folder, output_folder):
    """Run Roary on GFF files to generate the pangenome."""
    gff_files = [os.path.join(gff_folder, f'{f}', f'annotated_{f.replace("prokka_output_", "")}.gff') for f in os.listdir(gff_folder) if f.startswith('prokka_output_')]
    command = ["roary", "-p", "8", "-i", "60", "-f", output_folder] + gff_files  # Added --force to make sure the output is directed to the specified folder
    print(f"output fold: {output_folder}")
    print(f"command: {command}")
    subprocess.run(command, check=True)

    return output_folder


def find_most_and_least_represented_genes(clustered_proteins_file, gff_file):
    """Find the most and least represented genes for a given genome."""
    # Ottieni le frequenze dei gruppi e la mappatura gene-gruppo
    group_frequencies, gene_to_group = parse_clustered_proteins(clustered_proteins_file)

    # Estrai i geni dal file GFF
    genes_in_genome = parse_gff_for_genes(gff_file)

    # Trova i gruppi per ciascun gene e calcola le frequenze
    gene_frequencies = {}
    for gene in genes_in_genome:
        if gene in gene_to_group:
            group = gene_to_group[gene]
            gene_frequencies[gene] = group_frequencies.get(group, 0)
        else:
            print(f"Warning: Gene {gene} not found in clustered_proteins")

    # Verifica che ci siano geni con corrispondenza nel file clustered_proteins
    if gene_frequencies:
        most_represented_gene = max(gene_frequencies, key=gene_frequencies.get)
        print(f"Most representative gene is: {most_represented_gene} with its group having {gene_frequencies[most_represented_gene]} elements")
        least_represented_gene = min(gene_frequencies, key=gene_frequencies.get)
        print(f"Least representative gene is: {least_represented_gene} with its group having {gene_frequencies[least_represented_gene]} elements")
    else:
        raise ValueError("No genes from the GFF file were found in the clustered_proteins file.")

    return most_represented_gene, least_represented_gene, gene_frequencies



# Knockout Genes
def knockout_gene(sequence, gene_start, gene_end):
    return sequence[:gene_start] + sequence[gene_end:]

# Get gene position (Placeholder function)
def get_gene_position(gene_name, genome_index, gff_folder, sequence_id):
    """Get the start and end positions of a gene from the GFF file."""
    gff_file = os.path.join(
        gff_folder, f'prokka_output_{sequence_id}', f'annotated_{sequence_id}.gff'
    )
    start, end = None, None

    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            columns = line.strip().split('\t')
            if len(columns) < 9:
                continue
            if columns[2] == 'CDS':  # If it's a coding sequence
                info = columns[8]
                # Parse the attributes
                attributes = dict(
                    attr.split('=') for attr in info.split(';') if '=' in attr
                )
                # Check if gene_name matches ID or locus_tag
                if (
                    attributes.get('ID') == gene_name
                    or attributes.get('locus_tag') == gene_name
                ):
                    start = int(columns[3]) - 1  # Convert to zero-based index
                    end = int(columns[4])        # GFF end positions are inclusive
                    break  # Stop once the gene is found

    if start is None or end is None:
        print(f"Warning: Gene {gene_name} not found in genome {genome_index}.")

    return start, end


# Create Dataset from Sequence
def create_dataset(sequence, tokenizer, mask_prob=0.15):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as tmp_fasta:
        record = SeqRecord(Seq(sequence), id="knocked_out_sequence", description="Knock-out gene sequence")
        SeqIO.write(record, tmp_fasta, "fasta")
        fasta_path = tmp_fasta.name
        print(f"fasta path: {fasta_path}")
    fasta_data = FastaDataset(fasta_path, tokenizer, mask_prob=mask_prob)
    dataloader = DataLoader(fasta_data, batch_size=1, shuffle=False)
    os.remove(fasta_path)
    return dataloader

# Load MEGADNA Model
def load_model(model_path, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    model = MEGADNA(
        num_tokens=8,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=tokenizer.pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    return model, device

# Calculate Loss
def calculate_loss(model, sequence, device):
    model.eval()
    '''
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    '''
    sequence = sequence.to(device)
    with torch.no_grad():
        loss = model(sequence, return_value='loss')
    return loss.item()

# Calculate Likelihood
def calculate_likelihood(model, sequence, device):
    model.eval()
    '''
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    '''
    sequence = sequence.to(device)
    with torch.no_grad():
        logits, logits_trunk, start_tokens, ids = model(sequence, return_value="logits")
        logits = torch.cat((start_tokens, logits), dim=-2)
        likelihoods = torch.nn.functional.softmax(logits, dim=-1)
        likelihoods = likelihoods[:, :sequence.shape[1], :]
        mask = (sequence != 0).int()
        likelihoods_of_true_nucleotides = likelihoods[torch.arange(likelihoods.size(0)).unsqueeze(1), torch.arange(likelihoods.size(1)), sequence]
        masked_likelihoods = likelihoods_of_true_nucleotides * mask
        sum_likelihoods = masked_likelihoods.sum(dim=1)
        true_seq_len = mask.sum(dim=1)
        avg_likelihoods = sum_likelihoods / true_seq_len
        return avg_likelihoods.mean().item()

# Sample 2000 sequences from test_loader
def sample_sequences_from_test_loader(test_loader, n=2000):
    sampled_sequences = []
    for i, (sequences, sequence_ids) in enumerate(test_loader):
        if len(sampled_sequences) < n:
            sampled_sequences.extend(zip(sequences, sequence_ids))
        else:
            break
    return sampled_sequences[:n]

# Plot Distribution
def plot_distribution(data, labels, file_name):
    for label, values in data.items():
        sns.histplot(values, label=label, kde=True)
    plt.xlabel('Likelihood/ Loss')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(file_name)
    plt.clf()

def get_most_and_least_representative_genes(genes_in_genome, gene_frequencies):
    """Returns the most and least representative genes based on presence frequency in other genomes."""
    most_representative_gene = max(genes_in_genome, key=lambda gene: gene_frequencies.get(gene, 0))
    least_representative_gene = min(genes_in_genome, key=lambda gene: gene_frequencies.get(gene, float('inf')))
    return most_representative_gene, least_representative_gene

def parse_gff_for_genes(gff_file):
    """Extracts gene IDs from a GFF file."""
    genes = []
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            columns = line.split('\t')
            if len(columns) < 9:
                continue
            if columns[2] == 'CDS':  # Considera solo le CDS
                info = columns[8]
                gene_id = ''
                if 'ID=' in info:
                    gene_id = info.split('ID=')[1].split(';')[0].strip()  # Estrarre l'ID
                genes.append(gene_id)
    return genes


def read_roary_gene_presence(roary_output_subfolder):
    """Reads the gene presence/absence matrix generated by Roary."""
    # Il file gene_presence_absence.csv è situato nella sottocartella generata
    gene_presence_file = os.path.join(roary_output_subfolder, 'gene_presence_absence.csv')
    
    # Legge il file CSV
    gene_presence_df = pd.read_csv(gene_presence_file, sep=',', index_col=0)
    
    # Restituisce la matrice di presenza dei geni
    return gene_presence_df

def check_sequence_length(sampled_sequences, max_length=131000):
    """Check if any sequence exceeds the maximum length."""
    for sequence_tensor, sequence_id in sampled_sequences:
        sequence_length = sequence_tensor.shape[1]  # Controlla la lunghezza della sequenza
        if sequence_length > max_length:
            raise ValueError(f"Sequence ID {sequence_id} exceeds the maximum length of {max_length} nucleotides (Length: {sequence_length})")
    print("All sequences are within the allowed length.")

def check_sequences_in_test_loader(test_loader, max_length=131000):
    """Itera sul test_loader e verifica che nessuna sequenza superi la lunghezza specificata."""
    for i, (sequence_tensor, sequence_id) in enumerate(test_loader):
        sequence_length = sequence_tensor.shape[1]  # Controlla la lunghezza della sequenza lungo l'asse 1
        if sequence_length > max_length:
            raise ValueError(f"Sequence ID {sequence_id} exceeds the maximum length of {max_length} nucleotides. Length: {sequence_length}")
        print(f"Sequence ID {sequence_id} is valid with length: {sequence_length}")
    print("All sequences in the test loader are within the allowed length.")

# Knockout Genes on tensor (not sequence string)
def knockout_gene_tensor(sequence_tensor, gene_start, gene_end):
    """Performs knockout on the tensor representation of the gene."""
    print(f"Removing most representative that goes from {gene_start} to {gene_end}")
    knockout_sequence = torch.cat((sequence_tensor[:, :gene_start], sequence_tensor[:, gene_end:]), dim=1)
    print(f"Knocked out sequence length: {knockout_sequence.shape}")
    return knockout_sequence

def check_sequence_lengths_after_knockout(datasets, max_length=131000):
    """Check the lengths of sequences after the knockout process."""
    for label, sequence_batch in datasets.items():
        sequence_length = sequence_batch.shape[1]
        if sequence_length > max_length:
            raise ValueError(f"{label} sequence exceeds the maximum length of {max_length} nucleotides. Length: {sequence_length}")
        print(f"{label} sequence length: {sequence_length}")

def main():
    id_prova = 16
    model_path = f'./Trained_models/trained_model_prova{id_prova}.pth'
    test_dataset_path = f'./Trained_models/test_dataset_prova{id_prova}.pth'
    output_folder = f'./Figures/Prova{id_prova}/pangenome_knockout/'
    prokka_output_folder = f'./Figures/Prova{id_prova}/prokka_annotations_roary/'
    roary_output_folder = f'./Figures/Prova{id_prova}/roary_output/'
    gff_folder = prokka_output_folder
    tokenizer = Tokenizer()

    # Load the model
    model, device = load_model(model_path, tokenizer)
    
    # Annotate genomes with Prokka
    test_dataset_completo = torch.load(test_dataset_path)

    total_samples = test_dataset_completo.__len__()
    test_size = 1000
    inutil_size = total_samples - test_size

    test_dataset, inutil_dataset = torch.utils.data.random_split(test_dataset_completo, [test_size, inutil_size])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    check_sequences_in_test_loader(test_loader, max_length=131000)
    
    # Annotate and prepare GFF files for Roary
    for i, (sequence_tensor, sequence_id) in enumerate(test_loader):
        sequence_str = tokenizer.decode(sequence_tensor[0])
        sequence_id = str(sequence_id[0])
        fasta_file = f'temp_sequence_{sequence_id}_roary.fasta'
        with open(fasta_file, 'w') as f:
            SeqIO.write(SeqRecord(Seq(sequence_str), id=str(sequence_id), description='Sampled sequence'), f, 'fasta')
        
        prokka_output = os.path.join(prokka_output_folder, f'prokka_output_{sequence_id}')
        annotate_sequences_with_prokka(fasta_file, prokka_output, sequence_id)
        os.remove(fasta_file)
    
    # Run Roary to calculate the pangenome
    roary_csv = run_roary(gff_folder, roary_output_folder)

    #gene_frequencies = read_roary_gene_presence(roary_csv)

    # Process each genome
    results = {'losses': {}, 'likelihoods': {}}
    csv_results = []
    
    for i, (sequence_tensor, sequence_id) in enumerate(test_loader):
        sequence_str = tokenizer.decode(sequence_tensor[0])
        sequence_id = str(sequence_id[0])
        gff_file = os.path.join(prokka_output_folder, f'prokka_output_{sequence_id}', f'annotated_{sequence_id}.gff')
        
        # Trova il gene più e meno rappresentato per questo genoma usando il file clustered_proteins
        clustered_proteins_file = os.path.join(roary_csv, 'clustered_proteins')

        print(f"Let's find most essential gene and least of sequence {sequence_id}")

        most_represented_gene, least_represented_gene, gene_frequencies = find_most_and_least_represented_genes(clustered_proteins_file, gff_file)
        
        # Verifica se i cluster del gene più rappresentativo e meno rappresentativo hanno la stessa dimensione
        if gene_frequencies[most_represented_gene] == gene_frequencies[least_represented_gene]:
            print(f"Skipping sequence {sequence_id} as the cluster sizes for most and least represented genes are equal.")
            continue  # Salta al ciclo successivo

        # Trova le posizioni dei geni da eliminare
        most_rep_start, most_rep_end = get_gene_position(most_represented_gene, i, gff_folder, sequence_id)
        least_rep_start, least_rep_end = get_gene_position(least_represented_gene, i, gff_folder, sequence_id)
        
        print(f"Most rep start and end: {most_rep_start} - {most_rep_end}")
        print(f"Least rep start and end: {least_rep_start} - {least_rep_end}")
        # Esegui il knockout sui tensori codificati
        knocked_out_most_rep = knockout_gene_tensor(sequence_tensor, most_rep_start, most_rep_end)
        knocked_out_least_rep = knockout_gene_tensor(sequence_tensor, least_rep_start, least_rep_end)
        
        # Prepare datasets for wild-type, most representative knockout, and least representative knockout
        datasets = {
            'wild_type': sequence_tensor,  # Mantenere la forma originale del tensore
            'most_rep_knockout': knocked_out_most_rep,
            'least_rep_knockout': knocked_out_least_rep
        }
        
        print(f"Original sequence length: {sequence_tensor.shape}")
        check_sequence_lengths_after_knockout(datasets, max_length=131000)
        
        # Initialize row data for CSV
        row_data = {'sequence_id': sequence_id, 'wild_type_likelihood': None, 'wild_type_loss': None,
                    'most_rep_knockout_likelihood': None, 'most_rep_knockout_loss': None,
                    'least_rep_knockout_likelihood': None, 'least_rep_knockout_loss': None}

        # Calculate losses and likelihoods for each version
        for label, sequence_batch in datasets.items():
            print(f"sequence: {sequence_batch}")
            print(f"sequence_len {len(sequence_batch)}")
            print(f"shape {sequence_batch.shape}")
            loss = calculate_loss(model, sequence_batch, device)
            likelihood = calculate_likelihood(model, sequence_batch, device)

            row_data[f'{label}_loss'] = loss
            row_data[f'{label}_likelihood'] = likelihood
            results['losses'].setdefault(label, []).append(loss)
            results['likelihoods'].setdefault(label, []).append(likelihood)


        # Append row data to CSV results
        csv_results.append(row_data)

    # Save results to CSV
    df_results = pd.DataFrame(csv_results)
    # Save results to CSV
    os.makedirs(output_folder, exist_ok=True)  # Crea la cartella se non esiste
    df_results.to_csv(os.path.join(output_folder, 'results.csv'), index=False)

    # Plot distributions
    plot_distribution(results['losses'], datasets.keys(), os.path.join(output_folder, 'loss_distribution.png'))
    plot_distribution(results['likelihoods'], datasets.keys(), os.path.join(output_folder, 'likelihood_distribution.png'))

    print(f"Knocked out sequences: {len(csv_results)}")

if __name__ == "__main__":
    main()