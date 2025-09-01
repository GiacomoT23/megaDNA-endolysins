#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Eseguiamo il comando foldseek easy-search
###############################################################################
def run_foldseek_easysearch(query_dir, ref_dir, out_tsv, tmp_dir, threads=8):
    """
    Esegue `foldseek easy-search` per confrontare le strutture in query_dir
    con quelle in ref_dir. Salva i risultati in formato TSV (out_tsv).
    Ritorna il path del file TSV generato.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        query_dir,
        ref_dir,
        out_tsv,
        os.path.join(tmp_dir, "foldseek_tmp"),  # DB temporaneo
        "--alignment-type", "1",
        "--threads", str(threads),
        # Aggiungi campi: query,target,fident,alntmscore,qtmscore,ttmscore,lddt,prob
        "--format-output", "query,target,bits,fident,alntmscore,qtmscore,ttmscore,lddt,prob"
    ]

    print("[Foldseek] " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)

    return out_tsv

###############################################################################
# Parsing del TSV prodotto da foldseek
###############################################################################
def parse_foldseek_tsv(tsv_file):
    """
    Legge il TSV di Foldseek e ritorna un dizionario:
      {
        "query_filename_1": [ { 'target':..., 'alntmscore':..., 'qtmscore':..., 'ttmscore':..., 'lddt':..., 'prob':... }, ... ],
        "query_filename_2": [ ... ],
        ...
      }
    Ogni query può avere una o più hit (cioè target), con i relativi valori.
    """
    results = {}
    if not os.path.isfile(tsv_file):
        print(f"File TSV {tsv_file} non trovato!")
        return results

    with open(tsv_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            # Campi: query, target, fident, alntmscore, qtmscore, ttmscore, lddt, prob
            if len(parts) < 9:
                continue

            qId, tId, bits, fident, alntmscore, qtmscore, ttmscore, lddt, prob = parts[:9]

            row = {
                "target": tId,
                "bits": float(bits),
                "fident": float(fident),
                "alntmscore": float(alntmscore),
                "qtmscore": float(qtmscore),
                "ttmscore": float(ttmscore),
                "lddt": float(lddt),
                "prob": float(prob)
            }
            results.setdefault(qId, []).append(row)

    return results

###############################################################################
# Funzione per creare una tabella colorata (heatmap) con matplotlib
###############################################################################
def create_colored_table(rows_data, columns, output_image):
    """
    Crea un'immagine con una tabella colorata (tipo heatmap) data da:
      - `rows_data`: lista di dizionari con { 'label': nome_riga, 'col1': valore, 'col2': valore, ... }
      - `columns`: lista con i nomi delle colonne da visualizzare, in ordine.
      - `output_image`: path del file immagine (.png) da salvare.

    Esempio di rows_data:
      [
        { 'label': 'cartella1', 'alntmscore': 0.9, 'qtmscore': 0.8, ...},
        { 'label': 'cartella2', 'alntmscore': 0.7, 'qtmscore': 0.6, ...},
        ...
      ]
    """
    row_labels = [rd['label'] for rd in rows_data]
    
    # Costruiamo la matrice dei valori (num_rows x num_cols)
    data_matrix = []
    for rd in rows_data:
        row_vals = []
        for c in columns:
            val = rd.get(c, 0.0)
            row_vals.append(float(val))
        data_matrix.append(row_vals)

    data_matrix = np.array(data_matrix, dtype=float)

    # Creiamo la figura
    fig, ax = plt.subplots(figsize=(max(6, len(columns)*1.4), 
                                   max(4, len(rows_data)*0.8)))
    im = ax.imshow(data_matrix, aspect='auto', cmap='viridis')

    # Impostiamo i ticks per righe e colonne
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(rows_data)))
    ax.set_xticklabels(columns, rotation=90)
    ax.set_yticklabels(row_labels)

    # Mostra il valore numerico in ogni cella
    for i in range(len(rows_data)):
        for j in range(len(columns)):
            cell_val = data_matrix[i, j]
            ax.text(j, i, f"{cell_val:.4f}", 
                    ha="center", va="center", color="white", fontsize=7)

    plt.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Tabella colorata salvata in: {output_image}")

###############################################################################
# Funzione principale
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Analisi best-hit Foldseek per endolisine generate vs training set, su più cartelle di query."
    )
    parser.add_argument("-q", "--query_dirs", nargs='+', required=True,
                        help="Una o più cartelle con PDB delle endolisine (es. 112 .pdb).")
    parser.add_argument("-r", "--ref_dir", required=True,
                        help="Cartella con i PDB del training set.")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="Cartella di output per TSV e immagine.")
    parser.add_argument("--threads", type=int, default=32,
                        help="Numero di threads per foldseek (default: 32).")
    parser.add_argument("--tmp_dir", default="foldseek_tmp",
                        help="Cartella temporanea per i DB foldseek.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Salviamo i risultati medi in una lista di dizionari
    results_summary = []

    try:
        # Cicliamo su ogni cartella di query
        for q_dir in args.query_dirs:
            q_basename = os.path.basename(os.path.normpath(q_dir)) 
            # Esempio: "cartella1", "cartella2", etc.

            # File TSV di output *temporaneo* (uno per ogni cartella query)
            out_tsv = os.path.join(args.tmp_dir, f"foldseek_{q_basename}.tsv")

            # Eseguiamo foldseek
            run_foldseek_easysearch(
                query_dir=q_dir,
                ref_dir=args.ref_dir,
                out_tsv=out_tsv,
                tmp_dir=args.tmp_dir,
                threads=args.threads
            )

            # Parso i risultati
            foldseek_data = parse_foldseek_tsv(out_tsv)
            if not foldseek_data:
                print(f"\n[ATTENZIONE] Nessun risultato da foldseek per {q_dir}!")
                # Inseriamo comunque un record "vuoto" nel summary
                results_summary.append({
                    'label': q_basename,
                    'alntmscore': 0.0,
                    'qtmscore': 0.0,
                    'ttmscore': 0.0,
                    'lddt': 0.0,
                    'prob': 0.0
                })
                continue

            # Per ogni query, trovo la best hit come max(alntmscore)
            best_hits = []
            for qId, hitlist in foldseek_data.items():
                if hitlist:
                    best = hitlist[0]  # prima riga = bit score massimo
                    best_hits.append(best)

            if not best_hits:
                print(f"\n[ATTENZIONE] Nessuna best hit trovata per {q_dir}")
                # Inseriamo comunque un record "vuoto"
                results_summary.append({
                    'label': q_basename,
                    'alntmscore': 0.0,
                    'qtmscore': 0.0,
                    'ttmscore': 0.0,
                    'lddt': 0.0,
                    'prob': 0.0
                })
                continue

            # Calcolo la media di alntmscore, qtmscore, ttmscore, lddt, prob
            alntmscore_list = [bh["alntmscore"] for bh in best_hits]
            qtmscore_list   = [bh["qtmscore"]   for bh in best_hits]
            ttmscore_list   = [bh["ttmscore"]   for bh in best_hits]
            lddt_list       = [bh["lddt"]       for bh in best_hits]
            prob_list       = [bh["prob"]       for bh in best_hits]

            mean_alntmscore = sum(alntmscore_list) / len(alntmscore_list)
            mean_qtmscore   = sum(qtmscore_list)   / len(qtmscore_list)
            mean_ttmscore   = sum(ttmscore_list)   / len(ttmscore_list)
            mean_lddt       = sum(lddt_list)       / len(lddt_list)
            mean_prob       = sum(prob_list)       / len(prob_list)

            print(f"\n[OK] {q_dir}")
            print(f"  # di endolisine (query) = {len(best_hits)}")
            print(f"  media alntmscore = {mean_alntmscore:.4f}")
            print(f"  media qtmscore   = {mean_qtmscore:.4f}")
            print(f"  media ttmscore   = {mean_ttmscore:.4f}")
            print(f"  media lddt       = {mean_lddt:.4f}")
            print(f"  media prob       = {mean_prob:.4f}\n")

            results_summary.append({
                'label': q_basename,
                'alntmscore': mean_alntmscore,
                'qtmscore': mean_qtmscore,
                'ttmscore': mean_ttmscore,
                'lddt': mean_lddt,
                'prob': mean_prob
            })

    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        raise  
    finally:
        # Rimuovi la cartella temporanea, se esiste
        if os.path.isdir(args.tmp_dir):
            shutil.rmtree(args.tmp_dir, ignore_errors=True)
            print(f"Cartella temporanea rimossa: {args.tmp_dir}")

    # == A questo punto abbiamo i risultati in results_summary ==
    # Salviamo in un file TSV globale
    final_tsv = os.path.join(args.output_dir, "foldseek_multiquery_summary.tsv")
    with open(final_tsv, "w") as f:
        header = ["folder", "mean_alntmscore", "mean_qtmscore", "mean_ttmscore", "mean_lddt", "mean_prob"]
        f.write("\t".join(header) + "\n")
        for rd in results_summary:
            row_vals = [
                rd['label'],
                f"{rd['alntmscore']:.4f}",
                f"{rd['qtmscore']:.4f}",
                f"{rd['ttmscore']:.4f}",
                f"{rd['lddt']:.4f}",
                f"{rd['prob']:.4f}"
            ]
            f.write("\t".join(row_vals) + "\n")

    print(f"File di riepilogo creato: {final_tsv}")

    # Creiamo una tabella colorata
    # Per creare la heatmap, passiamo i dati e le colonne desiderate
    columns = ["alntmscore", "qtmscore", "ttmscore", "lddt", "prob"]
    # Convertiamo results_summary in una lista di dict ad hoc:
    # {
    #   'label': 'nome_cartella',
    #   'alntmscore': ...,
    #   ...
    # }
    table_image = os.path.join(args.output_dir, "foldseek_multiquery_summary.png")
    create_colored_table(results_summary, columns, table_image)

    print("\nAnalisi completata con successo!")

if __name__ == "__main__":
    main()