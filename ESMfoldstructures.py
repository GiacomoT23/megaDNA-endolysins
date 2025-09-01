import os
import torch
import argparse
import gc
from transformers import EsmForProteinFolding
from Bio import SeqIO

# 📌 Funzione per calcolare il pLDDT medio da un file PDB
def calculate_plddt(pdb_path):
    plddt_values = []
    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):  # Filtra solo le righe ATOM
                plddt = float(line[60:66])  # pLDDT è nei caratteri 61-66
                plddt_values.append(plddt)
    return sum(plddt_values) / len(plddt_values) if plddt_values else 0

# 🔄 Caricamento globale del modello
print("🔄 Caricamento del modello ESMFold...")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().cuda()
model.trunk.set_chunk_size(128)

def predict_structure_and_embeddings(fasta_file, output_folder, reload_every=50):
    global model  # Dichiarazione globale, così viene ricaricato correttamente
    
    if not os.path.exists(fasta_file):
        print(f"⚠️ Il file {fasta_file} non esiste. Skipping...")
        return

    os.makedirs(output_folder, exist_ok=True)

    counter = 0  # Conta il numero di proteine processate
    predicted = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id
        sequence = str(record.seq)

        print(f"\n🛠️ Elaborazione della proteina: {protein_id} (lunghezza: {len(sequence)})")

        # 🔴 Skippa proteine troppo lunghe
        if len(sequence) > 1650:
            print(f"⚠️ Proteina troppo lunga ({len(sequence)} aa). Skipping PDB...")
        else:
            try:
                with torch.no_grad():
                    output_pdb = model.infer_pdb(sequence)

                pdb_path = os.path.join(output_folder, f"{protein_id}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(output_pdb)

                print(f"✅ Struttura salvata in: {pdb_path}")
                predicted +=1

            except RuntimeError as e:
                print(f"❌ Errore nella predizione della struttura per {protein_id}: {e}")
                continue

        # 🔄 Libera memoria dopo ogni predizione
        torch.cuda.empty_cache()
        gc.collect()

        counter += 1
        if counter % reload_every == 0:
            print("🔁 Ricarico il modello per liberare memoria...")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().cuda()
            model.trunk.set_chunk_size(128)  # IMPORTANTE: Rimettere il chunk size!
    
    print(f"Successfully predicted structures: {predicted}/{counter}\n")
        # 📌 Calcola la media dei pLDDT delle proteine generate
    pdb_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".pdb")]
    if pdb_files:
        plddt_values = [calculate_plddt(pdb) for pdb in pdb_files]
        mean_plddt = sum(plddt_values) / len(plddt_values)
        
        # 📂 Salva la media dei pLDDT nella cartella upstream
        output_summary = os.path.join(os.path.dirname(output_folder), "plddt_summary.txt")
        with open(output_summary, "w") as f:
            f.write(f"Numero di proteine: {len(plddt_values)}\n")
            f.write(f"Media pLDDT: {mean_plddt:.2f}\n")

        print(f"📊 Media pLDDT salvata in {output_summary}: {mean_plddt:.2f}")


# 📌 Blocco per eseguire il codice come script da riga di comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice strutture e embeddings per proteine generate e UniProt usando ESMFold.")
    parser.add_argument("-g", "--generated", required=True, type=str, help="File FASTA delle proteine generate")
    parser.add_argument("-u", "--uniprot", type=str, default=None, help="File FASTA delle proteine UniProt (opzionale)")
    parser.add_argument("-o", "--output", required=True, type=str, help="Cartella in cui salvare i risultati")

    args = parser.parse_args()

    # Processa i due file
    predict_structure_and_embeddings(args.generated, os.path.join(args.output, "generated"))

    if args.uniprot:
        predict_structure_and_embeddings(args.uniprot, os.path.join(args.output, "uniprot"))