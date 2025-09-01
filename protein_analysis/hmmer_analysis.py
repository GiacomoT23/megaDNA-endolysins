#!/usr/bin/env python3

import argparse
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Conta quanti domini Pfam (chiave) compaiono in uno o più file FASTA (endolisine generate)."
    )
    parser.add_argument("--fasta", required=True, nargs='+',
                        help="Uno o più file FASTA con proteine generate (accetta wildcard).")
    parser.add_argument("--pfam_db", default="Data/Genomes/Pfam-A.hmm",
                        help="Database Pfam HMM (default: Pfam-A.hmm). Assicurati di aver eseguito hmmpress.")
    parser.add_argument("--output", default="results",
                        help="Cartella di output in cui salvare TSV e immagine (default: 'results').")
    parser.add_argument("--evalue_threshold", type=float, default=1e-3,
                        help="Soglia di E-value per considerare un dominio valido (default: 1e-3).")
    args = parser.parse_args()

    # Crea cartella di output, se non esiste
    os.makedirs(args.output, exist_ok=True)

    # Lista dei domini di interesse
    target_domains = [
        "Phage_lysozyme",
        "PG_binding_1",
        "PG_binding_3",
        "Peptidase_M15_4",
        "CHAP",
        "Glyco_hydro_25",
        "LysM",
        "LGFP",
        "Glyco_hydro_19",
        "VanY",
        "dGTP_diPhyd_N",
        "Peptidase_M23",
        "SH3_5",
        "Glyco_hydro_108",
        "Amidase_3",
        "CW_7",
        "LysM3_LYK4_5",
        "SH3_3",
        "Peptidase_C39_2",
        "Hydrolase_2",
        "LysM_RLK",
        "Peptidase_M15_3",
        "Choline_bind_1",
        "Muramidase",
        "SLT",
        "Glucosaminidase",
        "Choline_bind_3",
        "Choline_bind_2",
        "ZoocinA_TRD",
        "Cutinase",
        "Peptidase_M15_2",
        "Amidase_5",
    ]

    # Per comodità, aggiungiamo una colonna "total_count" in fondo
    col_labels = target_domains + ["total_count"]

    # Prepara una struttura dati per salvare i risultati per ogni FASTA
    # Ogni riga => conterrà i conteggi per i target_domains e la somma
    results_matrix = []
    row_labels = []

    # 1) Per ogni file FASTA, esegui hmmscan e conta i domini
    for fasta_file in args.fasta:
        # Inizializza i conteggi a 0
        domain_counts = {dom: 0 for dom in target_domains}

        # Esegui hmmscan => salviamo in un file temporaneo .domtbl
        domtbl_file = "temp_pfam.domtbl"
        cmd = [
            "hmmscan",
            "--cpu", "32",                   # puoi regolare i thread
            "--domtblout", domtbl_file,
            args.pfam_db,
            fasta_file
        ]
        print(f"\nEseguo: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Parso l'output .domtbl
        with open(domtbl_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split()
                if len(fields) < 7:
                    continue

                domain_name = fields[0]  # Pfam ID
                evalue = float(fields[6])  # colonna E-value (full seq)

                if domain_name in domain_counts and evalue <= args.evalue_threshold:
                    domain_counts[domain_name] += 1

        # Rimuovi il file temporaneo
        if os.path.exists(domtbl_file):
            os.remove(domtbl_file)

        # Calcola la somma dei conteggi
        total_hits = sum(domain_counts.values())

        # Prepara la riga per la matrice e la riga per il TSV
        row_data = []
        for dom in target_domains:
            row_data.append(domain_counts[dom])
        row_data.append(total_hits)

        # Salva nella struttura principale
        results_matrix.append(row_data)
        # Nome riga
        row_labels.append(os.path.basename(fasta_file))

    # 2) Salviamo i risultati in un file TSV unico
    tsv_path = os.path.join(args.output, "domain_counts.tsv")
    with open(tsv_path, "w") as out_tsv:
        # Header
        out_tsv.write("fasta_name\t" + "\t".join(col_labels) + "\n")

        # Righe
        for i, fasta_file in enumerate(args.fasta):
            row_vals = [os.path.basename(fasta_file)]
            # Aggiungi i conteggi
            for val in results_matrix[i]:
                row_vals.append(str(val))
            out_tsv.write("\t".join(row_vals) + "\n")

    print(f"\nRisultati salvati in: {tsv_path}")

    # 3) Creiamo un'immagine di tabella colorata (heatmap)
    data_matrix = np.array(results_matrix, dtype=float)
    # data_matrix.shape => (#fasta, #domini+1)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*0.6),
                                   max(4, len(row_labels)*0.4)))
    im = ax.imshow(data_matrix, aspect='auto', cmap='viridis')

    # Impostiamo le label su assi
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90)
    ax.set_yticklabels(row_labels)

    # Mettiamo i valori numerici dentro le celle
    n_rows = data_matrix.shape[0]
    n_cols = data_matrix.shape[1]
    for i in range(n_rows):
        for j in range(n_cols):
            val_str = str(int(data_matrix[i, j]))
            ax.text(j, i, val_str,
                    ha="center", va="center",
                    color="white", fontsize=7)

    # Aggiungiamo la colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=90)

    plt.tight_layout()

    png_path = os.path.join(args.output, "domain_counts.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Heatmap salvata in: {png_path}")

if __name__ == "__main__":
    main()