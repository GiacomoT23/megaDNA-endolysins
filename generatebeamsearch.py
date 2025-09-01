import os
import torch
import argparse
import math
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# IMPORTA i tuoi moduli custom (assicurati che i percorsi siano corretti)
from src.megaDNA import MEGADNA
from src.endolysinsTokenizer import endolysinsTokenizer

def load_pretrained_checkpoint(path, model):
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    print('Loaded pretrained model.')

################################################################################
# ESEMPIO di tabella di codon usage (fake). Dovrai sostituire con la tua tabella
################################################################################
codon_usage_ref = codon_stats = {
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

import math

def distribution_diff_rmsd_interval(sequence):
    """
    Calcola quanto la distribuzione di codoni di 'sequence' è fuori
    dall'intervallo [mean-std, mean+std] definito in 'codon_usage_ref',
    usando come metrica l'RMSD: root-mean-square di tali "distanze".
    """

    seq_len = len(sequence)
    total_codons = seq_len // 3

    # Conta i codoni presenti
    codon_count = {c: 0 for c in codon_usage_ref.keys()}
    for i in range(0, total_codons * 3, 3):
        cod = sequence[i:i+3]
        if cod in codon_count:
            codon_count[cod] += 1

    # Frequenze nella sequenza
    freq_map = {}
    for c in codon_count:
        freq_map[c] = (codon_count[c] / total_codons) if total_codons > 0 else 0.0

    # Calcolo la distanza quadratica dalle soglie e poi faccio la radice della media
    squared_diffs = []
    for codon, stats in codon_usage_ref.items():
        mean_val = stats["mean"]
        std_val = stats["std"]
        lower_bound = mean_val - std_val
        upper_bound = mean_val + std_val

        freq = freq_map.get(codon, 0.0)

        # Se è nel range, distanza=0, altrimenti calcolo la differenza
        if freq < lower_bound:
            diff = lower_bound - freq
        elif freq > upper_bound:
            diff = freq - upper_bound
        else:
            diff = 0.0

        squared_diffs.append(diff ** 2)

    n_codon_types = len(codon_usage_ref)
    mean_squared = sum(squared_diffs) / n_codon_types
    rmsd = math.sqrt(mean_squared)

    return rmsd


def calculate_repetitions(nuc_seq):
    """
    Penalizza le coppie contigue di amminoacidi identici nel blocco appena generato.
    Eleva penalty_base^(numero_coppie)
    """
    aa_seq = str(Seq(nuc_seq).translate(to_stop=False))
    count_pairs = 0
    for i in range(len(aa_seq) - 1):
        if aa_seq[i] == aa_seq[i+1]:
            count_pairs += 1
    return count_pairs

def score_sequence(sequence_to_score,
                   rep_term,
                   stop_in_frame,
                   w_rmsd=1.0,
                   w_stop=1.0,
                   w_rep=1.0):
    """
    Calcola lo score della sequenza combinando:
      - RMSD sui codoni (termine dist)
      - penalizzazione di ripetizioni (rep_factor)
      - penalità per lo stop codon, proporzionale alla lunghezza
        e pesata dal coefficiente w_stop

    Parametri:
      w_rmsd: peso del termine RMSD
      w_stop: peso del termine di penalizzazione stop
      w_rep : peso del termine di penalizzazione ripetizioni (se serve bilanciare)

    Ritorna un valore che, più è alto, meglio è (se vogliamo massimizzarlo),
    oppure più è basso, meglio è (se vogliamo minimizzare). 
    Qui diamo un esempio con punteggio “più basso è meglio” => costruiamo una perdita.
    """

    # RMSD con la distribuzione di riferimento
    dist = distribution_diff_rmsd_interval(sequence_to_score)
    
    # Penalizzazione ripetizioni
    #exponent = calculate_repetitions(sequence_to_score)
    #rep_factor = (rep_term ** exponent)

    '''
    # Penalizzazione stop legata alla lunghezza
    activate_len_penalization = 
    stop_penalty = stop_in_frame * total_codons
    '''

    # Costruiamo la funzione di costo (loss).
    # Se vogliamo che "più basso è meglio", allora sommiano tutti i termini di penalità.
    # Applichiamo i vari pesi per controllare l'impatto relativo.
    loss = (w_rmsd * dist)

    return loss

def expand_sequence_blockwise(model, seq_tensor, block_size, temperature, filter_thres):
    """
    Genera 'block_size' token aggiuntivi a partire da seq_tensor (shape (1, len_seq)).
    Ritorna:
      - new_seq_tensor (la sequenza estesa)
      - new_block_tokens (solo gli ultimi block_size token generati)
    """
    device = seq_tensor.device
    # Usiamo generate() per generare block_size token. 
    # Oppure, se preferisci un sampling custom, puoi farlo manualmente.
    with torch.no_grad():
        out = model.generate(
            seq_tensor,
            seq_len= len(seq_tensor[0]) + block_size,
            temperature=temperature,
            filter_thres=filter_thres
        )
    # out include la sequenza originale + i nuovi block_size token
    new_seq_tensor = out
    # estrai solo gli ultimi block_size
    new_block_tokens = out[0, -block_size:].tolist()
    return new_seq_tensor, new_block_tokens

def beam_search_blockwise(model, tokenizer, primer_tensor, num_context, new_tokenizer, max_total_tokens, 
                          beam_width, beams_to_save, threshold, temp, bsize, repfactor):
    """
    - beam_width = 3
    - Ogni step genera 3 varianti per ogni sequenza nel beam
    - Ogni variante aggiunge 9 nucleotidi
    - Valutiamo le 9 totali e ne prendiamo 3
    - Ripetiamo fino a STOP
    """
    # Ogni elemento del beam è una tupla: (seq_tensor, nuc_string, score)
    # dove 'nuc_string' è la sequenza nucleotidica decodificata finora.
    # Iniziamo con beam di dimensione 1 = primer
    decoded_primer = tokenizer.decode(primer_tensor[0].tolist())
    beam = [(primer_tensor, 100)]

    while True:
        new_candidates = []
        for (seq_tens, sc) in beam:
            # Se abbiamo raggiunto token speciali di STOP oppure max lunghezza, saltiamo
            if len(seq_tens[0]) >= max_total_tokens:
                # Non espandere oltre
                new_candidates.append((seq_tens, sc))
                continue
            # Controllo se la sequenza contiene ID di stop (prot_end_3_id, end_id)
            # Basta controllare l'ultimo token
            last_token = seq_tens[0][-1].item()
            if last_token in {tokenizer.prot_end_3_id, tokenizer.end_id}:
                # Non espandere oltre
                new_candidates.append((seq_tens, sc))
                continue

            # Altrimenti, generiamo 3 varianti
            expansions = []
            for _ in range(beam_width):
                new_seq_tens, new_block_tokens = expand_sequence_blockwise(
                    model, seq_tens, block_size=bsize, temperature=temp, filter_thres=threshold
                )
                
                # Controlla se i nuovi token contengono un token di stop (prot_end_3_id, end_id)
                # Se sì, troncalo
                valid_nucleotides = {1, 2, 3, 4}
                stop_token_set = {tokenizer.prot_end_3_id, tokenizer.end_id}

                # Codice da inserire subito dopo new_block_tokens = expand_sequence_blockwise(...)
                # Filtra i token in chunk di 3
                filtered_block_tokens = []
                i = 0
                while i < len(new_block_tokens):
                    # Se non ci sono almeno 3 token da prendere, esci dal loop
                    if i + 3 > len(new_block_tokens):
                        break
                    
                    codon = new_block_tokens[i : i + 3]
                    # Controlla se dentro questo codon compare un token "invalid"
                    # (non è né nucleotidico, né stop)
                    if any((t not in valid_nucleotides) and (t not in stop_token_set) for t in codon):
                        # Salta completamente il codone
                        i += 3
                        continue
                    
                    # Se vuoi, puoi anche verificare se trovi uno stop in mezzo:
                    # - Se lo trovi, decidi se troncare lì oppure aggiungerlo e fermarti.
                    #   Esempio semplice: aggiungo il codone finché non trovo stop
                    chunk = []
                    for t in codon:
                        chunk.append(t)
                        if t in stop_token_set:
                            break
                    filtered_block_tokens.extend(chunk)
                    
                    # Se l'ultimo token aggiunto è uno stop, ci fermiamo (non ha senso
                    # proseguire dentro questo blocco di 3 token)
                    if chunk[-1] in stop_token_set:
                        break
                    
                    i += 3

                # Ora new_block_tokens diventa la versione filtrata
                new_block_tokens = filtered_block_tokens

                truncated_block_tokens = []
                found_stop = False

                for i, tok in enumerate(new_block_tokens):
                    if tok in stop_token_set:
                        # tronca
                        truncated_block_tokens = new_block_tokens[: i + 1]
                        found_stop = True
                        break
                if not found_stop:
                    truncated_block_tokens = new_block_tokens
                # Aggiorna l'output tensore in caso di stop parziale
                # Tagliamo new_seq_tens all'esatto punto dello stop
                # serve conoscere la posizione relativa
                final_size = len(seq_tens[0]) + len(truncated_block_tokens)
                new_seq_tens = new_seq_tens[:, :final_size]

                # Calcola punteggio
                if new_tokenizer:
                    tokens_to_remove = 2 + num_context
                else:
                    tokens_to_remove = 1 + num_context
                if found_stop:
                    sequence_no_stop = new_seq_tens[:, :-1]  # rimuove l’ultimo token dalla dimensione della sequenza
                else:
                    sequence_no_stop = new_seq_tens
                sequence_no_stop = sequence_no_stop[:, tokens_to_remove:]
                sequence_to_score = tokenizer.decode(sequence_no_stop[0].tolist())
                if found_stop and len(sequence_to_score)%3==0:
                    stop_in_frame = True
                else:
                    stop_in_frame = False
                new_score = score_sequence(sequence_to_score, rep_term=repfactor, stop_in_frame=stop_in_frame)
                
                expansions.append((new_seq_tens, new_score))

            # Ora expansions contiene 3 versioni nuove, già tronche se necessario.
            new_candidates.extend(expansions)

        # Se non abbiamo generato nulla in new_candidates, usciamo
        if (not new_candidates):
            break

        # Ordiniamo i candidati per punteggio decrescente (dipende se score è più alto = meglio)
        # Ricorda: abbiamo definito score = come mi pare, quindi più piccolo è => meglio è
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=False)

        # Prendiamo i primi tot
        beam = new_candidates[:beams_to_save]

        # Controlla se tutti hanno finito
        # ovvero se hanno un token di stop o la lunghezza massima
        all_finished = True
        for (seq_tens, sc) in beam:
            if len(seq_tens[0]) < max_total_tokens:
                last_token = seq_tens[0][-1].item()
                if last_token not in {tokenizer.prot_end_3_id, tokenizer.end_id}:
                    all_finished = False
                    break

        if all_finished:
            break

    return beam


def translate_sequence(nucleotide_seq, translation_table=11):
    """Traduce una sequenza nucleotidica in una sequenza proteica usando il codice genetico 11."""
    try:
        return Seq(nucleotide_seq).translate(table=translation_table, to_stop=True)
    except Exception as e:
        print(f"Errore nella traduzione: {e}")
        return None

def run_generation(input_dir, output_dir, device_name, new_tokenizer, checkpoint_number,
                   output_file_name, primer_fasta, coding_primer_length, context, total_prepended_context, tot_gen, threshold, temp, bsize, repfactor):
    """
    1. Legge sequenze nucleotidiche da primer_fasta.
    2. Costruisce il primer (contesto non codificante + token speciali + primer codificante).
    3. Lancia la beam search 'a blocchi' per generare i nucleotidi mancanti.
    4. Salva la sequenza nucleotidica e la sua traduzione proteica in due file FASTA.
    """

    # File di output
    output_fasta = os.path.join(output_dir, output_file_name)             # Sequenze nucleotidiche
    translated_fasta = os.path.join(output_dir, f"translated_{output_file_name}")  # Sequenze proteiche tradotte

    # Inizializza il tokenizer
    tokenizer = endolysinsTokenizer()
    num_tokens = 12 if new_tokenizer else 8

    # Inizializza il modello
    model = MEGADNA(
        num_tokens=num_tokens,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=tokenizer.pad_id
    ).to(device_name)

    # Carica il checkpoint
    checkpoint_pretrained_model = os.path.join(input_dir, f"checkpoint_{checkpoint_number}.pth")
    load_pretrained_checkpoint(checkpoint_pretrained_model, model)
    model.eval()

    # Legge le sequenze nucleotidiche dal primer_fasta
    primer_records = list(SeqIO.parse(primer_fasta, "fasta"))
    print(f"Trovate {len(primer_records)} sequenze nel file dei primer: {primer_fasta}")

    # Mapping char->token
    char2token = {'A': 1,'T': 2,'C': 3,'G': 4}

    # Cicla su ogni sequenza e genera 1 output
    for rec_index, rec in enumerate(primer_records, start=1):
        # Prepara contesto
        total_genomic_context = rec.seq[:total_prepended_context]  # Primi 1000 nucleotidi
        coding_region = rec.seq[total_prepended_context:]
        used_context = total_genomic_context[-context:] if context > 0 else ''
        primer_coding = coding_region[:coding_primer_length]

        # Converti in token
        used_context_token_list = [char2token[nt] for nt in used_context]
        primer_coding_tokens_list = [char2token[nt] for nt in primer_coding]

        # Costruisci il primer tensor (includendo i token speciali)
        if new_tokenizer:
            final_primer = ([tokenizer.start_id]
                            + used_context_token_list
                            + [tokenizer.prot_start_5_id]
                            + primer_coding_tokens_list)
        else:
            final_primer = ([tokenizer.start_id]
                            + used_context_token_list
                            + primer_coding_tokens_list)

        primer_tensor = torch.tensor([final_primer], device=device_name)  # shape (1, len_primer)

        # Usa la beam search a blocchi
        # tot_gen = numero totale di token da generare (es: 4952)
        # ma in questa implementazione, 'max_total_tokens' = lunghezza massima di tokens
        max_total_tokens = len(final_primer) + tot_gen
        final_beam = beam_search_blockwise(
            model=model,
            tokenizer=tokenizer,
            primer_tensor=primer_tensor,
            num_context=context,
            new_tokenizer=new_tokenizer,
            max_total_tokens=max_total_tokens,
            beam_width=3,
            beams_to_save=3,
            threshold=threshold,
            temp=temp,
            bsize=bsize,
            repfactor=repfactor
        )

        output_path = Path(output_fasta)
        translated_output_path = Path(translated_fasta)
        if output_path.exists():
            with open(output_path, "r") as fasta_file:
                records = list(SeqIO.parse(fasta_file, "fasta"))
                last_id = int(records[-1].id) if records else 0
        else:
            last_id = 0

        for i, (beam_seq_tens, beam_score) in enumerate(final_beam, start=1):
            # Identificatore univoco, ad esempio last_id + i
            new_id = last_id + i
            # Tronca per i token speciali di stop
            truncated_seq = []
            stop_tokens_set = {tokenizer.prot_end_3_id, tokenizer.end_id}
            for t in beam_seq_tens[0]:
                if t.item() in stop_tokens_set:
                    break
                truncated_seq.append(t.item())

            # Togli i token di contesto e special
            if new_tokenizer:
                tokens_to_remove = 2 + context
            else:
                tokens_to_remove = 1 + context
            truncated_seq = truncated_seq[tokens_to_remove:] if len(truncated_seq) > tokens_to_remove else []

            # Verifica validità
            valid_values = {1, 2, 3, 4}
            if not set(truncated_seq).issubset(valid_values):
                print(f"[{rec.id}] Sequenza beam {i} non valida. Skipping.")
                continue

            if (len(truncated_seq) % 3 != 0):
                print(f"[{rec.id}] Sequenza beam {i} non multipla di 3. Skipping.")
                continue

            # Decodifica
            decoded_sequence = tokenizer.decode(truncated_seq)

            # Salva in FASTA
            seq_record = SeqRecord(Seq(decoded_sequence), id=str(new_id), description=f"Generated from {rec.id} (beam {i}, score={beam_score:.6f})")
            with open(output_path, "a") as fasta_file:
                SeqIO.write(seq_record, fasta_file, "fasta")

            # Traduce e salva la proteina
            translated_seq = translate_sequence(decoded_sequence)
            if translated_seq:
                translated_record = SeqRecord(
                    translated_seq,
                    id=str(new_id),
                    description=f"Translated from {rec.id} (beam {i}, score={beam_score:.6f})"
                )
                with open(translated_output_path, "a") as prot_file:
                    SeqIO.write(translated_record, prot_file, "fasta")

        torch.cuda.empty_cache()

    print("\nGenerazione completata! Risultati in:")
    print(f" - {output_fasta}")
    print(f" - {translated_fasta}")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help="input folder containing checkpoints", required=True, dest='input')
    argparser.add_argument('-o', '--output', help="output fasta file name", required=True, dest='output_file_name')
    argparser.add_argument('-n', '--checkpointnumber', help='checkpoint to use for generation', dest='checknumber', type=int, required=True)
    argparser.add_argument('-p', '--platform', help='platform', dest='platform', type=str, required=False, default='CPU', choices=['CUDA', 'CPU'])
    argparser.add_argument('-x', '--gpu_index', help='gpu device index', dest='gpu_index', type=str, required=False, default=None)
    argparser.add_argument('-t', '--tokenizer', help='which tokenizer was used', dest='tokenizer', type=str, required=False, default='old', choices=['old', 'new'])
    argparser.add_argument('--primer_fasta', type=str, required=True, help="FASTA con le sequenze di endolisine (nucleotidiche) da cui estrarre il primer.")
    argparser.add_argument('--primer_length', dest='primer_length', type=int, required=True, help="Numero di nucleotidi endolisina da usare come primer.")
    argparser.add_argument('--context', dest='context', type=int, required=True, help="Numero di nucleotidi contesto.")
    argparser.add_argument('--tpc', dest='total_prepended_context', type=int, required=True, help="Numero totale di nucleotidi contesto nel file.")
    argparser.add_argument('--tot', dest='total_to_generate', type=int, required=True, help="Numero totale di token da generare")
    argparser.add_argument('--topk', dest='topk', type=float, required=True, help="threshold topk")
    argparser.add_argument('--temp', dest='temp', type=float, required=True, help="temperature")
    argparser.add_argument('--bsize', dest='bsize', type=int, required=True, help="Block size beam search")
    argparser.add_argument('--repfactor', dest='repfactor', type=float, required=True, help="repetition factor")

    args = argparser.parse_args()
    
    device_name = f'cuda:{args.gpu_index}' if (args.platform == 'CUDA' and args.gpu_index is not None) else args.platform.lower()
    output_dir = os.path.join(args.input, f"generated_from_checkpoint_{args.checknumber}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_tokenizer = (args.tokenizer == 'new')
    
    run_generation(
        input_dir=args.input,
        output_dir=output_dir,
        device_name=device_name,
        new_tokenizer=new_tokenizer,
        checkpoint_number=args.checknumber,
        output_file_name=args.output_file_name,
        primer_fasta=args.primer_fasta,
        coding_primer_length=args.primer_length,
        context=args.context,
        total_prepended_context=args.total_prepended_context,
        tot_gen=args.total_to_generate,
        threshold=args.topk,
        temp=args.temp,
        bsize=args.bsize,
        repfactor=args.repfactor
    )