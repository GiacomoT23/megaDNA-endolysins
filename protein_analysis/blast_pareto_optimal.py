#!/usr/bin/env python3

import os
import sys
import argparse
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = []
    curr_start, curr_end = intervals[0]
    for (s, e) in intervals[1:]:
        if s <= curr_end + 1:
            curr_end = max(curr_end, e)
        else:
            merged.append((curr_start, curr_end))
            curr_start, curr_end = s, e
    merged.append((curr_start, curr_end))
    return merged

def main():
    parser = argparse.ArgumentParser(description="""
    Esegue un BLASTP remoto su NCBI per ciascuna proteina in un file FASTA,
    estrae la best hit e calcola coverage e identity (sommando tutti gli HSP).
    Risultati salvati in TSV.
    """)
    parser.add_argument("--proteins_fasta", required=True,
                        help="File FASTA con le sequenze proteiche da inviare a NCBI.")
    parser.add_argument("--out_dir", default="results",
                        help="Directory di output per salvare i risultati (default: results)")
    parser.add_argument("--database", default="nr",
                        help="Database remoto NCBI su cui lanciare BLASTP (default: nr).")
    parser.add_argument("--evalue", default="0.001",
                        help="E-value threshold (default: 0.001).")
    parser.add_argument("--hitlist_size", type=int, default=1,
                        help="Max numero di hit da recuperare (default: 1 => best hit).")
    parser.add_argument("--show_xml", action="store_true",
                        help="Se attivo, salva un file .xml per ogni query (debug).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_tsv = os.path.join(args.out_dir, "remote_blastp_results.tsv")

    proteins = list(SeqIO.parse(args.proteins_fasta, "fasta"))
    if not proteins:
        print(f"Nessuna sequenza trovata in {args.proteins_fasta}")
        sys.exit(1)

    with open(out_tsv, "w") as fw:
        fw.write("query_id\tbest_hit_accession\tbest_hit_title\tcoverage\tidentity\n")

        for i, record in enumerate(proteins, start=1):
            query_id = record.id
            seq_str = str(record.seq)
            qlen = len(seq_str)

            print(f"[{i}/{len(proteins)}] Eseguo BLASTP per {query_id}, length={qlen}...")

            try:
                result_handle = NCBIWWW.qblast(
                    program="blastp",
                    database=args.database,
                    sequence=seq_str,
                    hitlist_size=args.hitlist_size,
                    expect=args.evalue,
                )
            except Exception as e:
                print(f"Errore nel BLASTP di {query_id}: {e}")
                fw.write(f"{query_id}\terrore_blast\terrore_blast\t0.0\t0.0\n")
                continue

            if args.show_xml:
                xml_path = os.path.join(args.out_dir, f"{query_id}.xml")
                with open(xml_path, "w") as fx:
                    fx.write(result_handle.read())
                result_handle.close()
                with open(xml_path, "r") as fx2:
                    blast_record = NCBIXML.read(fx2)
            else:
                blast_record = NCBIXML.read(result_handle)

            result_handle.close()

            if not blast_record.alignments:
                print(f"Nessun hit per {query_id}")
                fw.write(f"{query_id}\tno_hit\tno_hit\t0.0\t0.0\n")
                continue

            alignment = blast_record.alignments[0]
            intervals = []
            sum_ident = 0
            sum_hsp_len = 0

            for hsp in alignment.hsps:
                qs = min(hsp.query_start, hsp.query_end)
                qe = max(hsp.query_start, hsp.query_end)
                intervals.append((qs, qe))
                sum_ident += hsp.identities
                sum_hsp_len += hsp.align_length

            merged = merge_intervals(intervals)
            total_cov_len = sum((end - start + 1) for (start, end) in merged)

            coverage_val = (total_cov_len / qlen) if qlen > 0 else 0.0
            identity_val = (sum_ident / float(sum_hsp_len)) if sum_hsp_len > 0 else 0.0

            accession = alignment.accession
            best_title = alignment.title.replace("\t", " ").strip()

            fw.write(f"{query_id}\t{accession}\t{best_title}\t{coverage_val:.4f}\t{identity_val:.4f}\n")

    print(f"\n[OK] Risultati salvati in: {out_tsv}")

if __name__ == "__main__":
    main()