#!/usr/bin/env python3
import os
import argparse
import subprocess
import math
import matplotlib.pyplot as plt
from Bio import SeqIO

###############################################################################
#                           CD-HIT RELATED FUNCTIONS                          #
###############################################################################

def run_cdhit(input_fasta, output_prefix, threshold=0.95, is_prot=False):
    """
    Esegue CD-HIT o CD-HIT-EST su 'input_fasta' con threshold di identità.
    is_prot=False -> cd-hit-est (nucleotidi), is_prot=True -> cd-hit (proteine).
    Ritorna il path del file .clstr generato.
    """
    if is_prot:
        cmd_base = "cd-hit"
    else:
        cmd_base = "cd-hit-est"
    cmd = [
        cmd_base,
        "-i", input_fasta,
        "-o", output_prefix,
        "-c", str(threshold)
    ]
    print("Esecuzione di", cmd_base, ":", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_prefix + ".clstr"

def count_clusters(clstr_file):
    """
    Conta il numero di cluster presenti nel file .clstr generato da CD-HIT.
    Ogni cluster è indicato da una riga che inizia con '>Cluster'.
    """
    count = 0
    with open(clstr_file, "r") as f:
        for line in f:
            if line.startswith(">Cluster"):
                count += 1
    return count

def run_cdhit_multiple(input_fasta, output_folder, is_prot=False):
    """
    Esegue CD-HIT(-EST) su 'input_fasta' per:
      - sequenze nucleotidiche: threshold [0.80, 0.85, 0.90, 0.95]
      - sequenze proteiche: threshold [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    Restituisce una lista di tuple (threshold, num_clusters, total_sequences).
    Salva i vari output in 'output_folder' con prefissi dedicati.
    """
    if is_prot:
        thresholds = [round(x, 2) for x in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
    else:
        thresholds = [round(x, 2) for x in [0.80, 0.85, 0.90, 0.95]]
    total_sequences = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    results = []
    for th in thresholds:
        prefix = os.path.join(
            output_folder,
            ("cdhit_prot_" if is_prot else "cdhit_nucl_") + str(th)
        )
        clstr_file = run_cdhit(input_fasta, prefix, threshold=th, is_prot=is_prot)
        nc = count_clusters(clstr_file)
        results.append((th, nc, total_sequences))
    return results

def write_cdhit_summary_tsv(results, tsv_file):
    """
    Salva in un TSV i risultati: threshold, num_clusters, total_num_sequences.
    """
    with open(tsv_file, "w") as f:
        f.write("threshold\tnum_clusters\ttotal_num_sequences\n")
        for (th, nc, tot) in results:
            f.write(f"{th}\t{nc}\t{tot}\n")

def plot_cdhit_clusters(results, plot_file, is_prot=False):
    """
    Crea un plot threshold vs num_clusters.
    """
    results_sorted = sorted(results, key=lambda x: x[0])
    x_vals = [r[0] for r in results_sorted]
    y_vals = [r[1] for r in results_sorted]
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Num. clusters")
    plt.title("CD-HIT Clustering " + ("Proteico" if is_prot else "Nucleotidico"))
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot CD-HIT salvato in: {plot_file}")

###############################################################################
#                               BLAST RELATED FUNCTIONS                       #
###############################################################################

def make_blast_db(input_fasta, db_name, is_prot=False):
    """
    Crea un database BLAST a partire dal file di training.
    is_prot=False -> 'nucl', is_prot=True -> 'prot'.
    """
    dbtype = "prot" if is_prot else "nucl"
    cmd = ["makeblastdb", "-in", input_fasta, "-dbtype", dbtype, "-out", db_name]
    print("Creazione database BLAST:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_blast(query_fasta, db_name, output_file, is_prot=False):
    """
    Esegue BLASTn o BLASTp (a seconda di is_prot) su query_fasta contro il database db_name.
    Formato tabulare con i campi:
      qseqid sseqid qstart qend pident length qlen slen nident
    """
    blast_cmd = "blastp" if is_prot else "blastn"
    outfmt = "6 qseqid sseqid qstart qend pident length qlen slen nident"
    cmd = [
        blast_cmd,
        "-query", query_fasta,
        "-db", db_name,
        "-out", output_file,
        "-outfmt", outfmt
    ]
    print("Esecuzione di", blast_cmd, ":", " ".join(cmd))
    subprocess.run(cmd, check=True)

def merge_intervals(intervals):
    """
    Unisce gli intervalli sovrapposti/adiacenti e restituisce una lista di
    intervalli disgiunti ordinati.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= current_end + 1:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))
    return merged

def parse_blast_output(blast_file):
    """
    Legge il file BLAST (formato tabulare) e per ogni coppia (qseqid, sseqid)
    raccoglie tutti gli HSP (qstart, qend, nident, align_len, qlen).
    Restituisce una struttura:
        data[(qseqid, sseqid)] = {
            'qlen': lunghezza query (int),
            'hsps': [ { 'qstart':..., 'qend':..., 'nident':..., 'align_len':... }, ... ]
        }
    """
    data = {}
    with open(blast_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            qseqid, sseqid, qstart, qend, pident, length, qlen, slen, nident = parts
            try:
                qstart = int(qstart)
                qend = int(qend)
                length = int(length)
                qlen = int(qlen)
                nident = int(nident)
            except ValueError:
                continue
            key = (qseqid, sseqid)
            if key not in data:
                data[key] = {'qlen': qlen, 'hsps': []}
            data[key]['hsps'].append({
                'qstart': qstart,
                'qend': qend,
                'nident': nident,
                'align_len': length
            })
    return data

def compute_coverage_identity_scores(data):
    """
    Per ogni (query, target):
      - coverage = (unione degli HSP sulla query) / qlen
      - identity = sum(nident)/sum(align_len)
      - final_score = coverage * identity
    Restituisce un dizionario con chiave (qseqid, sseqid) -> final_score.
    """
    scores = {}
    for (qseqid, sseqid), info in data.items():
        qlen = info['qlen']
        hsps = info['hsps']
        intervals = [(h['qstart'], h['qend']) for h in hsps]
        merged = merge_intervals(intervals)
        covered_bases = sum((e - s + 1) for (s, e) in merged)
        coverage = covered_bases / qlen if qlen > 0 else 0.0
        sum_nident = sum(h['nident'] for h in hsps)
        sum_align_len = sum(h['align_len'] for h in hsps)
        identity = sum_nident / sum_align_len if sum_align_len > 0 else 0.0
        final_score = coverage * identity
        scores[(qseqid, sseqid)] = final_score
    return scores

def write_blast_summary(scores, queries, targets, out_file):
    """
    Scrive un TSV con le colonne:
      query, target, complessive_identity
    Scrive solo le righe con complessive_identity > 0.
    """
    with open(out_file, "w") as f:
        f.write("query\ttarget\tcomplessive_identity\n")
        for t in targets:
            for q in queries:
                val = scores.get((q, t), 0.0)
                if val > 0:
                    f.write(f"{q}\t{t}\t{val:.4f}\n")
    print(f"File di riepilogo BLAST scritto in: {out_file}")

def compute_mean_identity_scores(scores, query_ids, target_ids):
    """
    Per ogni target, calcola la media del final_score (coverage*identity)
    su tutte le query. Se una coppia non è presente, considera 0.
    Ritorna un dict target -> mean_score.
    """
    results = {t: [] for t in target_ids}
    for t in target_ids:
        for q in query_ids:
            results[t].append(scores.get((q, t), 0.0))
    mean_scores = {}
    for t, score_list in results.items():
        mean_scores[t] = sum(score_list) / len(score_list) if score_list else 0.0
    return mean_scores

def plot_mean_identity_scores(mean_scores, plot_file, title="Mean Identity Scores"):
    """
    Genera un plot a barre con x = ID target e y = mean final_score.
    """
    sorted_items = sorted(mean_scores.items(), key=lambda x: x[0])
    x_labels = [x for x, _ in sorted_items]
    y_values = [y for _, y in sorted_items]
    plt.figure(figsize=(10, 5))
    plt.bar(x_labels, y_values)
    plt.xlabel("Target ID")
    plt.ylabel("Mean Final Score (coverage * identity)")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot mean identity scores salvato in: {plot_file}")

def count_unique_targets(summary_file):
    """
    Legge un file TSV (generato da write_blast_summary) e restituisce
    il numero di target unici (colonna 2) presenti.
    """
    unique_targets = set()
    with open(summary_file, "r") as f:
        header = f.readline()  # salta header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                unique_targets.add(parts[1])
    return len(unique_targets)

def write_unique_target_count(summary_file, out_txt):
    """
    Scrive in un file di testo il numero di target unici presenti in summary_file.
    """
    count = count_unique_targets(summary_file)
    with open(out_txt, "w") as f:
        f.write(f"Numero di target unici in {os.path.basename(summary_file)}: {count}\n")
    print(f"File di conteggio target unici scritto in: {out_txt}")

###############################################################################
#                               PROCESSING PER FILE                           #
###############################################################################

def process_generated_file(generated_file, training_file, training_prot_file, base_output):
    """
    Esegue l'intero workflow (CD-HIT, BLAST, plotting, riepiloghi, conteggio target unici)
    per un singolo file FASTA di sequenze generate.
    
    Crea una sottocartella in 'base_output' con il nome del file (senza estensione).
    """
    file_base = os.path.splitext(os.path.basename(generated_file))[0]
    out_folder = os.path.join(base_output, file_base)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(f"\n***** Elaborazione del file {generated_file} *****")
    
    # Costruzione del path del file proteico generato (prefisso "translated_")
    gen_dir = os.path.dirname(generated_file)
    gen_base = os.path.basename(generated_file)
    translated_generated = os.path.join(gen_dir, "TRANSLATED_" + gen_base)
    
    ##############################
    #  1) ESECUZIONE MULTIPLA CD-HIT
    ##############################
    print("==> Esecuzione multipla di CD-HIT-EST su sequenze nucleotidiche")
    cdhit_nucl_results = run_cdhit_multiple(
        input_fasta=generated_file,
        output_folder=out_folder,
        is_prot=False
    )
    nucl_tsv = os.path.join(out_folder, "cdhit_nucl_summary.tsv")
    write_cdhit_summary_tsv(cdhit_nucl_results, nucl_tsv)
    nucl_plot = os.path.join(out_folder, "cdhit_nucl_plot.png")
    plot_cdhit_clusters(cdhit_nucl_results, nucl_plot, is_prot=False)

    print("==> Esecuzione multipla di CD-HIT su sequenze proteiche (translated_)")
    cdhit_prot_results = run_cdhit_multiple(
        input_fasta=translated_generated,
        output_folder=out_folder,
        is_prot=True
    )
    prot_tsv = os.path.join(out_folder, "cdhit_prot_summary.tsv")
    write_cdhit_summary_tsv(cdhit_prot_results, prot_tsv)
    prot_plot = os.path.join(out_folder, "cdhit_prot_plot.png")
    plot_cdhit_clusters(cdhit_prot_results, prot_plot, is_prot=True)

    ##############################
    #  2) BLAST su nucleotidi
    ##############################
    print("\n==> BLASTn delle sequenze generate contro il training set (nucleotidi)")
    blast_db_nucl = os.path.join(out_folder, "training_db_nucl")
    make_blast_db(training_file, blast_db_nucl, is_prot=False)
    blast_out_nucl = os.path.join(out_folder, "blast_results_nucl.tsv")
    run_blast(generated_file, blast_db_nucl, blast_out_nucl, is_prot=False)
    data_nucl = parse_blast_output(blast_out_nucl)
    scores_nucl = compute_coverage_identity_scores(data_nucl)
    query_ids_nucl = [rec.id for rec in SeqIO.parse(generated_file, "fasta")]
    target_ids_nucl = [rec.id for rec in SeqIO.parse(training_file, "fasta")]
    blast_summary_nucl = os.path.join(out_folder, "blast_summary_nucl.tsv")
    write_blast_summary(scores_nucl, query_ids_nucl, target_ids_nucl, blast_summary_nucl)
    mean_scores_nucl = compute_mean_identity_scores(scores_nucl, query_ids_nucl, target_ids_nucl)
    mean_plot_nucl = os.path.join(out_folder, "mean_identity_scores_nucl.png")
    plot_mean_identity_scores(mean_scores_nucl, mean_plot_nucl, title="Mean Identity Scores - Nucleotidi")

    ##############################
    #  3) BLAST su proteine
    ##############################
    print("\n==> BLASTp delle sequenze generate contro il training set (proteine)")
    blast_db_prot = os.path.join(out_folder, "training_db_prot")
    make_blast_db(training_prot_file, blast_db_prot, is_prot=True)
    blast_out_prot = os.path.join(out_folder, "blast_results_prot.tsv")
    run_blast(translated_generated, blast_db_prot, blast_out_prot, is_prot=True)
    data_prot = parse_blast_output(blast_out_prot)
    scores_prot = compute_coverage_identity_scores(data_prot)
    query_ids_prot = [rec.id for rec in SeqIO.parse(translated_generated, "fasta")]
    target_ids_prot = [rec.id for rec in SeqIO.parse(training_prot_file, "fasta")]
    blast_summary_prot = os.path.join(out_folder, "blast_summary_prot.tsv")
    write_blast_summary(scores_prot, query_ids_prot, target_ids_prot, blast_summary_prot)
    mean_scores_prot = compute_mean_identity_scores(scores_prot, query_ids_prot, target_ids_prot)
    mean_plot_prot = os.path.join(out_folder, "mean_identity_scores_prot.png")
    plot_mean_identity_scores(mean_scores_prot, mean_plot_prot, title="Mean Identity Scores - Proteine")

    ##############################
    #  4) Conteggio target unici dai file BLAST summary
    ##############################
    nucl_target_count_file = os.path.join(out_folder, "blast_nucl_unique_targets.txt")
    prot_target_count_file = os.path.join(out_folder, "blast_prot_unique_targets.txt")
    write_unique_target_count(blast_summary_nucl, nucl_target_count_file)
    write_unique_target_count(blast_summary_prot, prot_target_count_file)
    
    print(f"\nElaborazione del file {generated_file} completata.\n")

###############################################################################
#                                     MAIN                                     #
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Script per clustering (CD-HIT) e analisi BLAST di sequenze nucleotidiche e proteiche."
    )
    parser.add_argument("--generated", required=True, nargs='+',
                        help="Uno o più file FASTA con le sequenze nucleotidiche generate (es. myseqs1.fasta myseqs2.fasta ...).")
    parser.add_argument("--training", required=True,
                        help="File FASTA con le sequenze nucleotidiche di training.")
    parser.add_argument("--training_prot", required=True,
                        help="File FASTA con le sequenze proteiche di training.")
    parser.add_argument("--output", required=True,
                        help="Cartella di output dove salvare i risultati (sarà creata una sottocartella per ogni file FASTA di input).")
    args = parser.parse_args()

    # Crea la cartella di output base se non esiste
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Processa ogni file FASTA di sequenze generate
    for generated_file in args.generated:
        process_generated_file(generated_file, args.training, args.training_prot, args.output)

    print("\nTutte le elaborazioni sono state completate correttamente.")

if __name__ == "__main__":
    main()