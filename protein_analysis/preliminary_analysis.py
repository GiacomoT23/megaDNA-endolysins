#!/usr/bin/env python3

"""
analysis_endolysins.py

Script per analizzare i risultati di generazione di endolisine in vari scenari:
- Carica uno o più file FASTA nucleotidici generati.
- (Opz.) Un file FASTA di sequenze di training (nucleotidico).
- (Opz.) Un file FASTA di training proteico corrispondente.
- (Opz.) Un file FASTA con proteine di endolisine prese da UniProt.
- Un certo numero di cartelle con i PDB già foldati corrispondenti a ciascun file FASTA,
  oppure zero cartelle (in tal caso, se --do_esmfold è attivo, esegue ESMFold).
- (Opz.) Cartelle con i PDB del training e di UniProt.

Output:
- Crea nella cartella di output un file info.txt con una riga per ogni file FASTA analizzato.
- Crea un’immagine tabella colorata (table.png) con le stesse metriche. 
- (Se necessario) genera i PDB foldati con ESMFold in sottocartelle "structures_<nome_fasta>".

Requisiti:
- blast+ installato (makeblastdb, blastn, blastp)
- foldseek installato
- transformers (per ESMFold)
- biopython
- matplotlib, numpy
"""
import torch
import os
import sys
import argparse
import uuid
import gc
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
from Bio.Blast import NCBIXML

from transformers import EsmForProteinFolding

###############################################################################
# Dizionario di codon usage (mean, std) - placeholder con alcune entry
###############################################################################
codon_usage_ref = {
    "AAA": {"mean": 0.0356, "std": 0.0290},
    "GAT": {"mean": 0.0335, "std": 0.0229},
    "AAG": {"mean": 0.0328, "std": 0.0194},
    "GAC": {"mean": 0.0307, "std": 0.0183},
    "GAA": {"mean": 0.0301, "std": 0.0207},
    "GGC": {"mean": 0.0290, "std": 0.0242},
    "GGT": {"mean": 0.0263, "std": 0.0173},
    "AAC": {"mean": 0.0259, "std": 0.0144},
    "GAG": {"mean": 0.0250, "std": 0.0154},
    "GCT": {"mean": 0.0245, "std": 0.0154},
    "AAT": {"mean": 0.0240, "std": 0.0193},
    "ATG": {"mean": 0.0222, "std": 0.0105},
    "GCC": {"mean": 0.0220, "std": 0.0217},
    "ATC": {"mean": 0.0213, "std": 0.0157},
    "TGG": {"mean": 0.0212, "std": 0.0111},
    "GCA": {"mean": 0.0207, "std": 0.0138},
    "ATT": {"mean": 0.0206, "std": 0.0158},
    "CAG": {"mean": 0.0197, "std": 0.0145},
    "TAC": {"mean": 0.0196, "std": 0.0133},
    "GTT": {"mean": 0.0194, "std": 0.0137},
    "TAT": {"mean": 0.0186, "std": 0.0153},
    "TTT": {"mean": 0.0184, "std": 0.0173},
    "GCG": {"mean": 0.0183, "std": 0.0185},
    "TTC": {"mean": 0.0179, "std": 0.0121},
    "CTG": {"mean": 0.0175, "std": 0.0180},
    "GGA": {"mean": 0.0175, "std": 0.0136},
    "CAA": {"mean": 0.0171, "std": 0.0140},
    "GTG": {"mean": 0.0162, "std": 0.0139},
    "GTC": {"mean": 0.0156, "std": 0.0147},
    "CGC": {"mean": 0.0152, "std": 0.0157},
    "GTA": {"mean": 0.0149, "std": 0.0125},
    "ACC": {"mean": 0.0148, "std": 0.0138},
    "TTA": {"mean": 0.0140, "std": 0.0161},
    "ACT": {"mean": 0.0138, "std": 0.0113},
    "ACA": {"mean": 0.0137, "std": 0.0129},
    "CCG": {"mean": 0.0131, "std": 0.0144},
    "CTT": {"mean": 0.0129, "std": 0.0111},
    "CCT": {"mean": 0.0120, "std": 0.0100},
    "GGG": {"mean": 0.0119, "std": 0.0099},
    "CGT": {"mean": 0.0118, "std": 0.0104},
    "CAC": {"mean": 0.0118, "std": 0.0105},
    "AGA": {"mean": 0.0114, "std": 0.0139},
    "CTC": {"mean": 0.0114, "std": 0.0129},
    "TCT": {"mean": 0.0110, "std": 0.0103},
    "CAT": {"mean": 0.0106, "std": 0.0101},
    "AGC": {"mean": 0.0104, "std": 0.0088},
    "ACG": {"mean": 0.0103, "std": 0.0101},
    "CCA": {"mean": 0.0100, "std": 0.0095},
    "TCA": {"mean": 0.0098, "std": 0.0097},
    "TTG": {"mean": 0.0096, "std": 0.0093},
    "ATA": {"mean": 0.0096, "std": 0.0130},
    "AGT": {"mean": 0.0096, "std": 0.0095},
    "TCG": {"mean": 0.0080, "std": 0.0092},
    "CCC": {"mean": 0.0077, "std": 0.0098},
    "CTA": {"mean": 0.0073, "std": 0.0084},
    "TCC": {"mean": 0.0073, "std": 0.0081},
    "CGG": {"mean": 0.0068, "std": 0.0088},
    "TGC": {"mean": 0.0065, "std": 0.0073},
    "AGG": {"mean": 0.0056, "std": 0.0080},
    "TGT": {"mean": 0.0055, "std": 0.0068},
    "CGA": {"mean": 0.0052, "std": 0.0061},
    "TAA": {"mean": 0.0024, "std": 0.0031},
    "TGA": {"mean": 0.0019, "std": 0.0029},
    "TAG": {"mean": 0.0008, "std": 0.0023}
}

# Stop codon veri e propri (batteriofagi? Adattare se serve)
REAL_STOP_CODONS = {'TAA', 'TAG', 'TGA'}

###############################################################################
# Funzioni helper di shell
###############################################################################
def run_command(cmd_list):
    """ Utility: esegue un comando in shell e stoppa se c'è errore. """
    print(f"Eseguo comando:\n  {' '.join(cmd_list)}\n")
    subprocess.run(cmd_list, check=True)

###############################################################################
# BLAST LOCALE NUCL (blastn) / PROT (blastp)
###############################################################################
def ensure_blast_db_nucl(db_fasta, db_prefix):
    """
    Crea un DB locale nucleotidico se non già esistente (makeblastdb -dbtype nucl).
    """
    if (not os.path.exists(db_prefix+".nhr") and
        not os.path.exists(db_prefix+".nsq") and
        not os.path.exists(db_prefix+".nin")):
        cmd = [
            "makeblastdb",
            "-in", db_fasta,
            "-dbtype", "nucl",
            "-out", db_prefix
        ]
        run_command(cmd)

def ensure_blast_db_prot(db_fasta, db_prefix):
    """
    Crea un DB locale proteico se non già esistente (makeblastdb -dbtype prot).
    """
    if (not os.path.exists(db_prefix+".phr") and
        not os.path.exists(db_prefix+".psq") and
        not os.path.exists(db_prefix+".pin")):
        cmd = [
            "makeblastdb",
            "-in", db_fasta,
            "-dbtype", "prot",
            "-out", db_prefix
        ]
        run_command(cmd)

def _create_temp_fasta(seq_str, tmp_dir, is_prot=False):
    """
    Crea un file FASTA temporaneo con la singola sequenza seq_str.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    import uuid
    q_id = str(uuid.uuid4())
    query_path = os.path.join(tmp_dir, f"query_{q_id}.fasta")
    with open(query_path, "w") as f:
        if is_prot:
            f.write(f">prot_query\n{seq_str}\n")
        else:
            f.write(f">nucl_query\n{seq_str}\n")
    return query_path

import os
import uuid
import math
import subprocess
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline

###########################################
# Utility per unire gli intervalli HSP
###########################################
def merge_intervals(intervals):
    """
    Unisce gli intervalli (start, end) eventualmente sovrapposti
    e ritorna la lista di intervalli disgiunti, ordinati.
    Attenzione: se i numeri sono 1-based e 'end' è inclusivo, 
    potresti dover aggiustare un +1 in coverage.
    """
    if not intervals:
        return []
    # Ordiniamo per inizio
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]

    for (s, e) in intervals[1:]:
        if s <= current_end+1:  
            # Sovrapposizione o adiacenza
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged

def run_blast_nucl(seq, training_nucl_fasta, db_prefix=None, tmp_dir="tmp_blast_nucl"):
    """
    Esegue BLASTN locale di 'seq' (nucleotidica) contro training_nucl_fasta.
    - Usa max_target_seqs=1 => prendi 1 soggetto (supposto come best).
    - Parso l'XML e prendo alignment[0].
    - Unisco TUTTI gli HSP => coverage = (unione dei segmenti di query_start..query_end) / len(seq).
    - identity = somma(identita' su ogni HSP) / somma(lunghezze HSP).
      (NB: se ci sono overlapp, quest'ultime verranno contate due volte per identity.)

    Ritorna (coverage, identity).
    Se nessun hit => (0.0, 0.0).
    """
    from Bio import SeqIO
    if (not training_nucl_fasta) or (not os.path.isfile(training_nucl_fasta)):
        return (0.0, 0.0)

    # Creazione (o riuso) del DB
    if db_prefix is None:
        db_prefix = os.path.join(tmp_dir, "train_nucl_db")
    ensure_blast_db_nucl(training_nucl_fasta, db_prefix)

    os.makedirs(tmp_dir, exist_ok=True)
    query_id = str(uuid.uuid4())
    query_path = os.path.join(tmp_dir, f"query_{query_id}.fasta")
    with open(query_path, "w") as fq:
        fq.write(f">temp_nucl\n{seq}\n")

    out_xml = os.path.join(tmp_dir, f"blastn_{uuid.uuid4()}.xml")

    cmd_blast = NcbiblastnCommandline(
        cmd="blastn",
        query=query_path,
        db=db_prefix,
        outfmt=5,  # XML
        out=out_xml,
        max_target_seqs=1
    )
    print(f"\n[BLASTN] {cmd_blast}")
    stdout, stderr = cmd_blast()

    coverage, identity = 0.0, 0.0
    from Bio.Blast import NCBIXML
    with open(out_xml) as fx:
        blast_record = NCBIXML.read(fx)
    if blast_record.alignments:
        # Prendi la best alignment
        alignment = blast_record.alignments[0]
        # Somma i parametri su tutti gli HSP
        intervals = []
        sum_id = 0  # somma identità
        sum_len = 0  # somma lunghezze HSP

        for hsp in alignment.hsps:
            # HSP in BLAST nucleotidico, tipicamente 1-based e inclusive
            qstart = min(hsp.query_start, hsp.query_end)  # Sicuro in caso invertito
            qend = max(hsp.query_start, hsp.query_end)
            intervals.append((qstart, qend))

            sum_id += hsp.identities
            sum_len += hsp.align_length

        # Merge intervals per coverage
        merged = merge_intervals(intervals)
        # coverage => unione intervalli su len(seq)
        # Se query_start e query_end sono inclusivi e 1-based,
        # la lunghezza di un intervallo (s, e) e' e - s + 1
        total_bases_covered = 0
        for (s, e) in merged:
            total_bases_covered += (e - s + 1)
        if len(seq) > 0:
            coverage = total_bases_covered / len(seq)

        # identity => somma(identita') / somma(length HSP)
        # Overlap e' contato due volte nella somma, occhio.
        if sum_len>0:
            identity = sum_id / float(sum_len)

    return (coverage, identity)

def run_blast_prot(prot_seq, training_prot_fasta, db_prefix=None, tmp_dir="tmp_blast_prot"):
    """
    Esegue BLASTP locale di 'prot_seq' (amminoacidica) contro training_prot_fasta.
    - Usa max_target_seqs=1 => prendi 1 soggetto (best).
    - coverage = unione di TUTTI gli HSP su query / len(prot_seq)
    - identity = somma(identities) / somma(allineamenti HSP)
      (NB: overlap contato due volte x identity, e' la "media" su hsp totali)

    Ritorna (coverage, identity).
    Se nessun hit => (0.0, 0.0).
    """
    if (not training_prot_fasta) or (not os.path.isfile(training_prot_fasta)):
        return (0.0, 0.0)

    if db_prefix is None:
        base_name = os.path.basename(training_prot_fasta)
        base_noext = os.path.splitext(base_name)[0]
        db_prefix = os.path.join(tmp_dir, f"{base_noext}_db")
    ensure_blast_db_prot(training_prot_fasta, db_prefix)

    os.makedirs(tmp_dir, exist_ok=True)
    query_id = str(uuid.uuid4())
    query_path = os.path.join(tmp_dir, f"query_{query_id}.fasta")
    with open(query_path, "w") as fq:
        fq.write(f">temp_prot\n{prot_seq}\n")

    out_xml = os.path.join(tmp_dir, f"blastp_{uuid.uuid4()}.xml")

    cmd_blast = NcbiblastpCommandline(
        cmd="blastp",
        query=query_path,
        db=db_prefix,
        outfmt=5,
        out=out_xml,
        max_target_seqs=1
    )
    print(f"\n[BLASTP] {cmd_blast}")
    stdout, stderr = cmd_blast()

    coverage, identity = 0.0, 0.0
    with open(out_xml) as fx:
        blast_record = NCBIXML.read(fx)
    if blast_record.alignments:
        alignment = blast_record.alignments[0]

        intervals = []
        sum_id = 0
        sum_len = 0
        for hsp in alignment.hsps:
            # BLASTP hsp => query_start, query_end
            qstart = min(hsp.query_start, hsp.query_end)
            qend = max(hsp.query_start, hsp.query_end)
            intervals.append((qstart, qend))

            sum_id += hsp.identities
            sum_len += hsp.align_length

        merged = merge_intervals(intervals)
        total_cov = 0
        for (s,e) in merged:
            total_cov += (e - s + 1)
        if len(prot_seq)>0:
            coverage = total_cov / len(prot_seq)

        if sum_len>0:
            identity = sum_id / float(sum_len)

    return (coverage, identity)


###############################################################################
# FOLDSEEK
###############################################################################
def parse_foldseek_tsv(tsv_file):
    """
    Ritorna un dict:
      {
         queryID: [
            { 'target':..., 'fident':..., 'alntmscore':..., 'qtmscore':..., 'ttmscore':..., 'lddt':... },
            ...
         ],
         ...
      }
    """
    data = {}
    if not os.path.isfile(tsv_file):
        return data
    with open(tsv_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            qId, tId, fident, alntmscore, qtmscore, ttmscore, lddt = parts[:7]
            row = {
                "target": tId,
                "fident": float(fident),
                "alntmscore": float(alntmscore),
                "qtmscore": float(qtmscore),
                "ttmscore": float(ttmscore),
                "lddt": float(lddt)
            }
            data.setdefault(qId, []).append(row)
    return data

def run_foldseek_easysearch(query_dir, ref_dir, out_basename="foldseek_results", threads=32, num_iterations=3):
    """
    Esegue foldseek easy-search query_dir vs ref_dir => out_basename.tsv
    Ritorna un dict { queryID -> [hits] }.
    """
    out_tsv = out_basename + ".tsv"
    out_tmp = out_basename + "_tmp"

    cmd = [
        "foldseek", "easy-search",
        query_dir,
        ref_dir,
        out_tsv,
        out_tmp,
        "--threads", str(threads),
        "--num-iterations", str(num_iterations),
        "--format-output", "query,target,fident,alntmscore,qtmscore,ttmscore,lddt"
    ]
    run_command(cmd)

    fs_results = parse_foldseek_tsv(out_tsv)
    return fs_results

###############################################################################
# ESMFOLD
###############################################################################
def load_esmfold_model():
    """
    Carica e restituisce ESMFold v1 su GPU.
    """
    print("🔄 Caricamento ESMFold (facebook/esmfold_v1)...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().cuda()
    model.trunk.set_chunk_size(128)
    return model

def run_esmfold(protein_sequence, output_pdb_path, model):
    """
    Predice struttura con ESMFold e salva in output_pdb_path.
    Ritorna pLDDT medio.
    """
    seq_len = len(protein_sequence)
    if seq_len > 1650:
        print(f"⚠️ Sequenza {seq_len}aa, troppo lunga per ESMFold. Skipping.")
        return None
    
    try:
        with torch.no_grad():
            pdb_str = model.infer_pdb(protein_sequence)
        with open(output_pdb_path, "w") as f:
            f.write(pdb_str)
        plddt_val = compute_plddt(output_pdb_path)
        print(f"✅ ESMFold -> {output_pdb_path}, pLDDT={plddt_val:.2f}")
        return plddt_val
    except RuntimeError as e:
        print(f"❌ Errore ESMFold: {e}")
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def compute_plddt(pdb_file):
    """
    Calcola media pLDDT dai campi B-factor (col 61-66) nel PDB ESMFold/AlphaFold.
    """
    plddt_vals = []
    with open(pdb_file, "r") as fp:
        for line in fp:
            if line.startswith("ATOM"):
                val_str = line[60:66].strip()
                if val_str:
                    try:
                        val = float(val_str)
                        plddt_vals.append(val)
                    except ValueError:
                        pass
    if plddt_vals:
        return sum(plddt_vals)/len(plddt_vals)
    return 0.0

###############################################################################
# 1) Funzioni di analisi base
###############################################################################

def ends_with_stop(seq):
    if len(seq) < 3:
        return False
    last_codon = seq[-3:].upper()
    return last_codon in REAL_STOP_CODONS

def translate_nuc(seq):
    """Traduce in amminoacidi (no stop finale)."""
    return str(Seq(seq).translate(to_stop=True, table=11))

def codon_distribution_penalty(seq):
    """
    Calcola quante volte (count) i codoni escono dal range [mean - std, mean + std].
    """
    from collections import Counter
    seq = seq.upper()
    length = len(seq)
    if length<3: 
        return 0

    total_codons = length//3
    local_count = Counter(seq[i:i+3] for i in range(0, total_codons*3, 3))
    # freq
    local_freq = {}
    for c, ncount in local_count.items():
        local_freq[c] = ncount/total_codons

    count_violations = 0
    for codon, stats in codon_usage_ref.items():
        freq = local_freq.get(codon, 0.0)
        low = stats['mean'] - stats['std']
        high = stats['mean'] + stats['std']
        if freq<low or freq>high:
            count_violations +=1
    return count_violations

###############################################################################
# 2) Funzione analyze_fasta
###############################################################################

def analyze_fasta(nucl_fasta_path, 
                  training_nucl_fasta=None,
                  training_prot_fasta=None,
                  #uniprot_prot_fasta=None,
                  output_dir=None,
                  existing_pdb_dir=None,
                  do_esmfold=False,
                  training_pdb_dir=None,
                  #uniprot_pdb_dir=None,
                  esmfold_model=None):
    """
    Analizza un singolo file FASTA nucleotidico generato.
    Ritorna un dizionario con i campi richiesti.
    """
    from collections import defaultdict

    records = list(SeqIO.parse(nucl_fasta_path, "fasta"))
    if not records:
        # Se non ci sono sequenze
        return {
            'fasta': str(nucl_fasta_path),
            'mean_length': 0,
            'std_length': 0,
            'multiple_of3': 0,
            'pct_stop_end': 0.0,
            'plddt_mean': None,
            'num_codons_out_of_range': 0,
            'coverage_nucl_mean': 0.0,
            'identity_nucl_mean': 0.0,
            'pct_coverage_nucl_gt_0_5': 0.0,
            'coverage_prot_mean': 0.0,
            'identity_prot_mean': 0.0,
            'pct_coverage_prot_gt_0_5': 0.0,
            #'coverage_prot_uniprot_mean': 0.0,
            #'identity_prot_uniprot_mean': 0.0,
            #'pct_coverage_prot_uniprot_gt_0_5': 0.0,
            'pct_foldseek_training_gt_03': 0.0,
            #'pct_foldseek_uniprot_gt_03': 0.0
        }

    lengths = []
    multiple3_count = 0
    stop_end_count = 0
    codon_violations = []

    coverage_nucl_list = []
    identity_nucl_list = []
    cov_nucl_05_count = 0

    coverage_prot_list = []
    identity_prot_list = []
    cov_prot_05_count = 0

    '''
    coverage_up_list = []
    identity_up_list = []
    cov_up_05_count = 0
    '''

    plddt_vals = []

    base_name = os.path.splitext(os.path.basename(nucl_fasta_path))[0]
    # Dir strutture
    if existing_pdb_dir is not None:
        structures_dir = existing_pdb_dir
    elif do_esmfold:
        # Crea cartella structures_<base_name>
        parent_dir = os.path.dirname(nucl_fasta_path)
        structures_dir = os.path.join(parent_dir, f"structures_{base_name}")
        os.makedirs(structures_dir, exist_ok=True)
    else:
        structures_dir = None  # nessuna analisi strutturale

    # analisi sequenza x sequenza
    for rec in records:
        nucseq = str(rec.seq).upper()
        if len(nucseq)<3:
            continue
        if len(nucseq) % 3 == 0:
            multiple3_count += 1
        else:
            to_remove = len(nucseq) % 3
            nucseq = nucseq[: len(nucseq) - to_remove]

        # controlla stop
        if ends_with_stop(nucseq):
            stop_end_count +=1

        # traduci
        prot_seq = translate_nuc(nucseq)
        lengths.append(len(prot_seq))

        # codon usage
        viol = codon_distribution_penalty(nucseq)
        codon_violations.append(viol)

        # blast nucleotidico
        cov_nucl, id_nucl = run_blast_nucl(nucseq, training_nucl_fasta)
        coverage_nucl_list.append(cov_nucl)
        identity_nucl_list.append(id_nucl)
        if cov_nucl>0.5:
            cov_nucl_05_count +=1

        # blast proteico vs training
        cov_prot, id_prot = run_blast_prot(prot_seq, training_prot_fasta)
        coverage_prot_list.append(cov_prot)
        identity_prot_list.append(id_prot)
        if cov_prot>0.5:
            cov_prot_05_count+=1

        '''
        # blast proteico vs uniprot
        cov_up, id_up = run_blast_prot(prot_seq, uniprot_prot_fasta)
        coverage_up_list.append(cov_up)
        identity_up_list.append(id_up)
        if cov_up>0.5:
            cov_up_05_count+=1
        '''

        # fold se serve
        if structures_dir:
            pdb_path = os.path.join(structures_dir, f"{rec.id}.pdb")
            if not os.path.exists(pdb_path):
                # genero con ESMFold se ho model e do_esmfold
                if do_esmfold and (esmfold_model is not None) and (existing_pdb_dir is None):
                    run_esmfold(prot_seq, pdb_path, esmfold_model)
            if os.path.exists(pdb_path):
                plv = compute_plddt(pdb_path)
                plddt_vals.append(plv)

    if not lengths:
        # se nessuna seq è passata i filtri
        return {
            'fasta': str(nucl_fasta_path),
            'mean_length': 0,
            'std_length': 0,
            'multiple_of3': 0,
            'pct_stop_end': 0.0,
            'plddt_mean': None,
            'num_codons_out_of_range': 0,
            'coverage_nucl_mean': 0.0,
            'identity_nucl_mean': 0.0,
            'pct_coverage_nucl_gt_0_5': 0.0,
            'coverage_prot_mean': 0.0,
            'identity_prot_mean': 0.0,
            'pct_coverage_prot_gt_0_5': 0.0,
            #'coverage_prot_uniprot_mean': 0.0,
            #'identity_prot_uniprot_mean': 0.0,
            #'pct_coverage_prot_uniprot_gt_0_5': 0.0,
            'pct_foldseek_training_gt_03': 0.0,
            #'pct_foldseek_uniprot_gt_03': 0.0
        }

    arr_len = np.array(lengths)
    mean_len = float(arr_len.mean())
    std_len = float(arr_len.std())
    pct_multiple3 = float(multiple3_count/len(lengths))
    pct_stop = float(stop_end_count/len(lengths))

    plddt_mean = None
    if plddt_vals:
        plddt_mean = float(np.mean(plddt_vals))

    mean_cod_viol = float(np.mean(codon_violations))

    arr_cov_nucl = np.array(coverage_nucl_list)
    arr_id_nucl = np.array(identity_nucl_list)
    cov_nucl_mean = float(arr_cov_nucl.mean()) if len(arr_cov_nucl)>0 else 0.0
    id_nucl_mean = float(arr_id_nucl.mean()) if len(arr_id_nucl)>0 else 0.0
    pct_cov_nucl_05 = float(np.sum(arr_cov_nucl>0.5)/len(arr_cov_nucl))

    arr_cov_prot = np.array(coverage_prot_list)
    arr_id_prot = np.array(identity_prot_list)
    cov_prot_mean = float(arr_cov_prot.mean()) if len(arr_cov_prot)>0 else 0.0
    id_prot_mean = float(arr_id_prot.mean()) if len(arr_id_prot)>0 else 0.0
    pct_cov_prot_05 = float(np.sum(arr_cov_prot>0.5)/len(arr_cov_prot))

    '''
    arr_cov_up = np.array(coverage_up_list)
    arr_id_up = np.array(identity_up_list)
    cov_up_mean = float(arr_cov_up.mean()) if len(arr_cov_up)>0 else 0.0
    id_up_mean = float(arr_id_up.mean()) if len(arr_id_up)>0 else 0.0
    pct_cov_up_05 = float(np.sum(arr_cov_up>0.5)/len(arr_cov_up))
    '''

    # foldseek training/uniprot se ho training_pdb_dir e structures_dir
    pct_foldseek_training = 0.0
    pct_foldseek_uniprot = 0.0

    if training_pdb_dir and os.path.isdir(training_pdb_dir) and structures_dir and os.path.isdir(structures_dir):
        fsres = run_foldseek_easysearch(structures_dir, training_pdb_dir, out_basename=os.path.join(output_dir, "foldseek_training"))
        # Qui fsres = { queryID: [ {fident, alntmscore,...}, ... ] }
        count_ok = 0
        total = len(fsres)
        for qId, hits in fsres.items():
            # controlla se almeno uno ha >0.3
            keep = False
            for h in hits:
                if (h["fident"]>0.3 and h["alntmscore"]>0.3 and h["qtmscore"]>0.3 and
                    h["ttmscore"]>0.3 and h["lddt"]>0.3):
                    keep = True
                    break
            if keep:
                count_ok+=1
        if total>0:
            pct_foldseek_training = float(count_ok/total)
    '''
    if uniprot_pdb_dir and os.path.isdir(uniprot_pdb_dir) and structures_dir and os.path.isdir(structures_dir):
        fsres_up = run_foldseek_easysearch(structures_dir, uniprot_pdb_dir, out_basename=os.path.join(output_dir, "foldseek_uniprot"))
        count_ok = 0
        total = len(fsres_up)
        for qId, hits in fsres_up.items():
            keep = False
            for h in hits:
                if (h["fident"]>0.3 and h["alntmscore"]>0.3 and h["qtmscore"]>0.3 and
                    h["ttmscore"]>0.3 and h["lddt"]>0.3):
                    keep = True
                    break
            if keep:
                count_ok+=1
        if total>0:
            pct_foldseek_uniprot = float(count_ok/total)
    '''

    res = {
        'fasta': str(nucl_fasta_path),
        'mean_length': mean_len,
        'std_length': std_len,
        'multiple_of3': pct_multiple3,
        'pct_stop_end': pct_stop,
        'plddt_mean': plddt_mean,
        'num_codons_out_of_range': mean_cod_viol,
        'coverage_nucl_mean': cov_nucl_mean,
        'identity_nucl_mean': id_nucl_mean,
        'pct_coverage_nucl_gt_0_5': pct_cov_nucl_05,
        'coverage_prot_mean': cov_prot_mean,
        'identity_prot_mean': id_prot_mean,
        'pct_coverage_prot_gt_0_5': pct_cov_prot_05,
        #'coverage_prot_uniprot_mean': cov_up_mean,
        #'identity_prot_uniprot_mean': id_up_mean,
        #'pct_coverage_prot_uniprot_gt_0_5': pct_cov_up_05,
        'pct_foldseek_training_gt_03': pct_foldseek_training,
        #'pct_foldseek_uniprot_gt_03': pct_foldseek_uniprot
    }
    return res


###############################################################################
# 3) Creazione della tabella colorata
###############################################################################
def create_colored_table_for_columns(all_results, columns, output_image_path):
    """
    Crea un'immagine heatmap limitata alle 'columns' fornite.
    """
    if not all_results:
        return
    
    row_labels = []
    data_matrix = []
    for rd in all_results:
        row_labels.append(os.path.basename(rd['fasta']))
        row_vals = []
        for col in columns:
            val = rd[col]
            if val is None:
                val = 0.0
            row_vals.append(val)
        data_matrix.append(row_vals)

    data_matrix = np.array(data_matrix, dtype=float)
    
    # Creiamo la figura e la heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(columns)*1.2), max(4, len(all_results)*0.8)))
    im = ax.imshow(data_matrix, aspect='auto', cmap='viridis')

    # Imposta i ticks
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(all_results)))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(row_labels)

    # Mostra il valore all'interno di ogni cella
    for i in range(len(all_results)):
        for j in range(len(columns)):
            textval = f"{data_matrix[i,j]:.2f}"
            ax.text(j, i, textval, ha="center", va="center", color="white", fontsize=7)

    plt.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()


def create_colored_tables_separate(all_results, output_dir):
    """
    Suddivide le colonne in tre gruppi e genera 3 immagini separate,
    ognuna con la propria scala di colori.
    """
    # 1) Colonne con valori grandi (es. lunghezze)
    group1 = ["mean_length", "std_length"]

    # 2) Colonna dei codoni out-of-range
    group2 = ["num_codons_out_of_range"]

    # 3) Tutto il resto (valori tipicamente da 0 a 1)
    group3 = [
        "multiple_of3", "pct_stop_end", "plddt_mean",
        "coverage_nucl_mean", "identity_nucl_mean", "pct_coverage_nucl_gt_0_5",
        "coverage_prot_mean", "identity_prot_mean", "pct_coverage_prot_gt_0_5",
        #"coverage_prot_uniprot_mean", "identity_prot_uniprot_mean", "pct_coverage_prot_uniprot_gt_0_5",
        "pct_foldseek_training_gt_03"#, "pct_foldseek_uniprot_gt_03"
    ]

    os.makedirs(output_dir, exist_ok=True)

    # Creiamo i tre file PNG
    out1 = os.path.join(output_dir, "table_length_stats.png")
    create_colored_table_for_columns(all_results, group1, out1)
    
    out2 = os.path.join(output_dir, "table_codon_out_of_range.png")
    create_colored_table_for_columns(all_results, group2, out2)
    
    out3 = os.path.join(output_dir, "table_01_stats.png")
    create_colored_table_for_columns(all_results, group3, out3)

###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Analisi endolisine generate (nucleotidi).")
    parser.add_argument('--fasta_generated', nargs='+', required=True,
                        help="Uno o più file FASTA nucleotidici generati.")
    parser.add_argument('--training_nucl', type=str, default=None,
                        help="Fasta nucleotidico usato per training (opz.).")
    parser.add_argument('--training_prot', type=str, default=None,
                        help="Fasta proteico corrispondente (opz.).")
    #parser.add_argument('--uniprot_prot', type=str, default=None,
                        #help="Fasta proteico con endolisine uniprot (opz.).")
    parser.add_argument('--pdb_generated_dirs', nargs='*', default=None,
                        help="0 o N cartelle con pdb corrispondenti ai file. Se 0 e --do_esmfold => genero ESMFold.")
    parser.add_argument('--do_esmfold', action='store_true',
                        help="Se attivato e se non ho cartelle, genero i PDB con ESMFold.")
    parser.add_argument('--training_pdb_dir', default=None, help="Cartella con PDB del training (opz.)")
    #parser.add_argument('--uniprot_pdb_dir', default=None, help="Cartella con PDB uniprot (opz.)")
    parser.add_argument('--output_dir', required=True,
                        help="Cartella di output (info.txt + table.png).")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    info_txt = os.path.join(args.output_dir, "info.txt")

    # Carico modello ESMFold se do_esmfold
    model_esmfold = None
    if args.do_esmfold:
        model_esmfold = load_esmfold_model()

    # Abbiamo M cartelle PDB vs N file. Se M=0 => no existing, generiamo se do_esmfold
    existing_pdb_map = {}
    if args.pdb_generated_dirs:
        if len(args.pdb_generated_dirs) not in [0, len(args.fasta_generated)]:
            print("ERRORE: numero cartelle PDB deve essere 0 o uguale al numero di FASTA.")
            sys.exit(1)
        if len(args.pdb_generated_dirs) == len(args.fasta_generated):
            for i, fpath in enumerate(args.fasta_generated):
                existing_pdb_map[fpath] = args.pdb_generated_dirs[i]

    all_results = []
    for fpath in args.fasta_generated:
        print(f"\n=== Analizzo {fpath} ===")
        if fpath in existing_pdb_map:
            existing_dir = existing_pdb_map[fpath]
        else:
            existing_dir = None

        # Richiamiamo
        res = analyze_fasta(
            nucl_fasta_path=fpath,
            training_nucl_fasta=args.training_nucl,
            training_prot_fasta=args.training_prot,
            #uniprot_prot_fasta=args.uniprot_prot,
            output_dir=args.output_dir,
            existing_pdb_dir=existing_dir,
            do_esmfold=args.do_esmfold,
            training_pdb_dir=args.training_pdb_dir,
            #uniprot_pdb_dir=args.uniprot_pdb_dir,
            esmfold_model=model_esmfold
        )
        all_results.append(res)

    # Scriviamo info.txt
    with open(info_txt, "w") as fp:
        colnames = [
            'fasta','mean_length','std_length', 'multiple_of3', 'pct_stop_end','plddt_mean',
            'num_codons_out_of_range',
            'coverage_nucl_mean','identity_nucl_mean','pct_coverage_nucl_gt_0_5',
            'coverage_prot_mean','identity_prot_mean','pct_coverage_prot_gt_0_5',
            #'coverage_prot_uniprot_mean','identity_prot_uniprot_mean','pct_coverage_prot_uniprot_gt_0_5',
            'pct_foldseek_training_gt_03'#,'pct_foldseek_uniprot_gt_03'
        ]
        fp.write("\t".join(colnames)+"\n")
        for rd in all_results:
            rowvals = []
            for c in colnames:
                val = rd[c]
                if val is None:
                    rowvals.append("NA")
                else:
                    rowvals.append(str(val))
            fp.write("\t".join(rowvals)+"\n")

    # Creiamo la tabella immagine
    create_colored_tables_separate(all_results, args.output_dir)

    print(f"\nAnalisi completata. Risultati in:\n - {info_txt}\n - {args.output_dir}\n")


if __name__ == "__main__":
    main()