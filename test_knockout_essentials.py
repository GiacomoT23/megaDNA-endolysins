import os
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import torch
from torch.utils.data import DataLoader
from src.megaDNA import MEGADNA
from src.tokenizer import Tokenizer
from src.dataset.FastaDataset import FastaDataset
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt

# Annotate Genomes with Prokka
def annotate_sequences_with_prokka(fasta_file, output_folder):
    """Annotates a genome using Prokka."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    command = ["prokka", fasta_file, "--outdir", output_folder, "--prefix", "annotated", "--force"]


    subprocess.run(command, check=True)

# Parse Prokka GFF3 Output
def parse_gff_for_genes(gff_file):
    """Extracts gene locations and names from a GFF file."""
    genes = []
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            columns = line.split('\t')
            if len(columns) < 9:
                continue
            if columns[2] == 'CDS':  # Coding sequences
                info = columns[8]
                gene_name = ''
                if 'gene=' in info:
                    gene_name = info.split('gene=')[1].split(';')[0]
                elif 'product=' in info and 'hypothetical protein' in info:
                    gene_name = 'hypothetical protein'
                protein_id = ''
                if 'ID=' in info:
                    protein_id = info.split('ID=')[1].split(';')[0]
                start = int(columns[3])
                end = int(columns[4])
                genes.append({'name': gene_name, 'start': start, 'end': end, 'protein_id': protein_id})
    return genes

# Run blastp to identify essential proteins
def run_blastp(query_proteins_fasta, essentials_db, blast_output_file):
    """Run blastp on Prokka-annotated proteins against the essential proteins DB."""
    command = [
        "blastp",
        "-query", query_proteins_fasta,
        "-db", essentials_db,
        "-out", blast_output_file,
        "-outfmt", "6 qseqid sseqid pident length qcovs evalue",
        "-evalue", "1e-3",  # Filter out low-quality hits
        "-num_threads", "4"
    ]
    subprocess.run(command, check=True)

# Parse blastp results to find the best hit based on identity × coverage
def parse_blast_results(blast_output_file, query_fasta_file, identity_threshold=70.0, coverage_threshold=60.0, evalue_threshold=1e-3):
    """Parse blastp output and return the best hit based on identity × coverage, using correct coverage calculation."""
    
    # Step 1: Load query lengths from the FASTA file
    query_lengths = {}
    for record in SeqIO.parse(query_fasta_file, "fasta"):
        query_lengths[record.id] = len(record.seq)
    
    best_hit = None
    best_score = 0  # To store the maximum identity × coverage score
    
    # Step 2: Parse the BLAST output and calculate the best score
    with open(blast_output_file, 'r') as f:
        for line in f:
            columns = line.strip().split('\t')
            query_id = columns[0]
            identity = float(columns[2])
            alignment_length = int(columns[3])
            coverage = float(columns[4])
            evalue = float(columns[5])
            
            # Get the length of the query sequence
            query_length = query_lengths.get(query_id, None)
            
            if query_length is None:
                print(f"Warning: Length of query {query_id} not found.")
                continue
            
            # Calculate the coverage percentage
            coverage = (alignment_length / query_length) * 100
            
            # Calculate the product of identity and coverage
            score = identity * coverage
            
            # Apply thresholds for identity, coverage, and e-value
            if identity >= identity_threshold and coverage >= coverage_threshold and evalue <= evalue_threshold:
                if score > best_score:
                    best_score = score
                    best_hit = {'query_id': query_id, 'score': score, 'identity': identity, 'coverage': coverage, 'evalue': evalue}
    
    return best_hit

# Knockout Genes
def knockout_gene(sequence, gene_start, gene_end):
    return sequence[:gene_start] + sequence[gene_end:]

# Create Dataset from Sequence
def create_dataset(sequence, tokenizer, mask_prob=0.15):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.fasta') as tmp_fasta:
        record = SeqRecord(Seq(sequence), id="knocked_out_sequence", description="Knock-out gene sequence")
        SeqIO.write(record, tmp_fasta, "fasta")
        fasta_path = tmp_fasta.name
    fasta_data = FastaDataset(fasta_path, tokenizer, mask_prob=mask_prob)
    dataloader = DataLoader(fasta_data, batch_size=1, shuffle=False)
    os.remove(fasta_path)
    return dataloader

# Load MEGADNA Model
def load_model(model_path, tokenizer):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = MEGADNA(
        num_tokens=8,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=tokenizer.pad_id
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, device

# Calculate Loss
def calculate_loss(model, sequence, device):
    model.eval()
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    sequence = sequence.to(device)
    with torch.no_grad():
        loss = model(sequence, return_value='loss')
    return loss.item()

# Calculate Likelihood
def calculate_likelihood(model, sequence, device):
    model.eval()
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
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

def calculate_log_likelihood(model, sequence, device):
    model.eval()
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)
    sequence = sequence.to(device)
    with torch.no_grad():
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
    
# Plot Distribution
def plot_distribution(data, labels, file_name):
    for label, values in data.items():
        sns.histplot(values, label=label, kde=True)
    plt.xlabel('Likelihood/ Loss')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(file_name)
    plt.clf()

# Sample 100 sequences from saved test dataset
def sample_sequences_from_test_loader(test_loader, n=100):
    sampled_sequences = []
    for i, (sequences, sequence_ids) in enumerate(test_loader):
        if len(sampled_sequences) < n:
            sampled_sequences.extend(zip(sequences, sequence_ids))
        else:
            break
    return sampled_sequences[:n]

def process_all_sequences_from_test_loader(test_loader):
    sequences = []
    for sequences_batch, sequence_ids in test_loader:
        sequences.extend(zip(sequences_batch, sequence_ids))
    return sequences

# Main Function
def main():
    id_prova = 16
    model_path = f'./Trained_models/trained_model_prova{id_prova}.pth'
    test_dataset_path = f'./Trained_models/test_dataset_prova{id_prova}.pth'
    output_folder = f'./Figures/Prova{id_prova}/test_portal_terminase_major_loglikelihood/'
    csv_file_path = os.path.join(output_folder, 'results.csv')
    essentials_db = 'Data/essentialproteins/essentials_portal_terminase_major_db'
    tokenizer = Tokenizer()

    # Load the model
    model, device = load_model(model_path, tokenizer)

    # Load test dataset
    test_dataset = torch.load(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Sample 1000 sequences from the test dataset
    sampled_sequences = sample_sequences_from_test_loader(test_loader, 1000)

    #sampled_sequences = process_all_sequences_from_test_loader(test_loader)

    # Initialize results
    results = {'losses': {}, 'likelihoods': {}, 'loglikelihoods': {}}
    csv_results = []

    for i, (sequence_tensor, sequence_id) in enumerate(sampled_sequences):
        sequence_str = tokenizer.decode(sequence_tensor)
        fasta_file = f'temp_sequence_{sequence_id}.fasta'
        with open(fasta_file, 'w') as f:
            SeqIO.write(SeqRecord(Seq(sequence_str), id=str(sequence_id), description='Sampled sequence'), f, 'fasta')
        
        prokka_output_folder = os.path.join(output_folder, f'prokka_output_{sequence_id}')
        annotate_sequences_with_prokka(fasta_file, prokka_output_folder)

        # Run blastp on Prokka-annotated proteins
        prokka_proteins_fasta = os.path.join(prokka_output_folder, 'annotated.faa')
        blast_output_file = os.path.join(prokka_output_folder, 'blastp_output.txt')
        run_blastp(prokka_proteins_fasta, essentials_db, blast_output_file)

        # Parse blastp results and find best essential protein based on identity × coverage
        best_essential_hit = parse_blast_results(blast_output_file, prokka_proteins_fasta)
        gff_file = os.path.join(prokka_output_folder, 'annotated.gff')
        genes = parse_gff_for_genes(gff_file)
        
        # Annotate essential proteins
        essential_protein = None
        if best_essential_hit:
            essential_protein = next((g for g in genes if g['protein_id'] == best_essential_hit['query_id']), None)
        
        # Find hypothetical proteins
        hypothetical_protein = next((g for g in genes if 'hypothetical' in g['name'].lower()), None)

        # Only perform essential knockout if an essential protein is found
        if essential_protein:
            # Perform knockout for the essential protein
            knocked_out_essential = knockout_gene(sequence_str, essential_protein['start'], essential_protein['end'])
            
            # Perform knockout for hypothetical protein if found
            knocked_out_hypothetical = knockout_gene(sequence_str, hypothetical_protein['start'], hypothetical_protein['end']) if hypothetical_protein else sequence_str

            # Prepare datasets for wild-type, essential knockout, and hypothetical knockout
            datasets = {
                'wild_type': create_dataset(sequence_str, tokenizer),
                'essential_knockout': create_dataset(knocked_out_essential, tokenizer),
                'hypothetical_knockout': create_dataset(knocked_out_hypothetical, tokenizer) if hypothetical_protein else None
            }

            # Initialize row data for CSV
            row_data = {'sequence_id': sequence_id, 'wild_type_likelihood': None, 'wild_type_loss': None,
                        'essential_knockout_likelihood': None, 'essential_knockout_loss': None,
                        'hypothetical_knockout_likelihood': None, 'hypothetical_knockout_loss': None}

            # Calculate losses and likelihoods for each version
            for label, dataset in datasets.items():
                if dataset is None:
                    continue
                for sequence_batch in dataset:
                    sequence_batch = sequence_batch[0]
                    loss = calculate_loss(model, sequence_batch, device)
                    likelihood = calculate_likelihood(model, sequence_batch, device)
                    loglikelihood = calculate_log_likelihood(model, sequence_batch, device)
                    row_data[f'{label}_loss'] = loss
                    row_data[f'{label}_likelihood'] = likelihood
                    row_data[f'{label}_loglikelihood'] = loglikelihood
                    results['losses'].setdefault(label, []).append(loss)
                    results['likelihoods'].setdefault(label, []).append(likelihood)
                    results['loglikelihoods'].setdefault(label, []).append(loglikelihood)

            # Append row data to CSV results
            csv_results.append(row_data)
        os.remove(fasta_file)

    # Save results to CSV
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv(csv_file_path, index=False)

    # Plot distributions of losses and likelihoods
    plot_distribution(results['losses'], datasets.keys(), os.path.join(output_folder, 'loss_distribution.png'))
    plot_distribution(results['likelihoods'], datasets.keys(), os.path.join(output_folder, 'likelihood_distribution.png'))
    plot_distribution(results['loglikelihoods'], datasets.keys(), os.path.join(output_folder, 'loglikelihood_distribution.png'))
    print(f"Numero di essential proteins trovate e di cui è stato effettuato il knockout: {len(csv_results)}")

if __name__ == "__main__":
    main()