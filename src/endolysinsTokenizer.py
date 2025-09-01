import torch

class endolysinsTokenizer:
    def __init__(self, vocab='ATCG'):
        self.vocab = vocab
        self.token2id = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.id2token = {idx + 1: char for idx, char in enumerate(vocab)}

        # Token speciali
        self.pad_token = '[PAD]'
        self.pad_id = 0  # Padding token ID
        self.unknown_token = 'N'
        self.unknown_id = len(vocab) + 1  # ID per nucleotidi sconosciuti
        self.start_token = '[START]'
        self.start_id = len(vocab) + 2
        self.end_token = '[END]'
        self.end_id = len(vocab) + 3

        # Token per le CDS (delimitatori gene)
        self.prot_start_5 = '[5’-PROT_START]'
        self.prot_start_5_id = len(vocab) + 4
        self.prot_end_3 = '[3’-PROT_END]'
        self.prot_end_3_id = len(vocab) + 5
        self.prot_start_3 = '[3’-PROT_START]'  # Se mai lo userai
        self.prot_start_3_id = len(vocab) + 6
        self.prot_end_5 = '[5’-PROT_END]'      # Se mai lo userai
        self.prot_end_5_id = len(vocab) + 7

        # Aggiunta dei token speciali ai dizionari
        self.token2id.update({
            self.unknown_token: self.unknown_id,
            self.start_token: self.start_id,
            self.end_token: self.end_id,
            self.pad_token: self.pad_id,
            self.prot_start_5: self.prot_start_5_id,
            self.prot_end_3: self.prot_end_3_id,
            self.prot_start_3: self.prot_start_3_id,
            self.prot_end_5: self.prot_end_5_id
        })

        self.id2token.update({
            self.unknown_id: self.unknown_token,
            self.start_id: self.start_token,
            self.end_id: self.end_token,
            self.pad_id: self.pad_token,
            self.prot_start_5_id: self.prot_start_5,
            self.prot_end_3_id: self.prot_end_3,
            self.prot_start_3_id: self.prot_start_3,
            self.prot_end_5_id: self.prot_end_5
        })

    def tokenize(self, sequence):
        """
        Converte una sequenza in una lista di token (caratteri).
        Se un carattere non è in vocab, usa self.unknown_token.
        """
        return [char if char in self.vocab else self.unknown_token for char in sequence]

    def encode(self, sequence, add_gene_delimiters=False, gene_start=None, gene_end=None):
        """
        Converte una sequenza in ID numerici, inserendo [START] e [END].
        Se add_gene_delimiters=True, inserisce [5’-PROT_START] e [3’-PROT_END] intorno al gene.

        Args:
            sequence (str): La sequenza genomica (upstream + gene).
            add_gene_delimiters (bool): Se True, aggiunge i token gene-specifici.
            gene_start (int): Indice in 'sequence' dove inizia il gene.
            gene_end (int): Indice in 'sequence' dove finisce il gene.
        Returns:
            torch.Tensor: Tensore con gli ID dei token.
        """
        # Tokenizzo base per base (A,T,C,G o N)
        base_tokens = self.tokenize(sequence)

        # Se NON devi aggiungere i delimitatori di gene, allora
        # [START] + base_tokens + [END] e basta
        if not add_gene_delimiters or gene_start is None or gene_end is None:
            full_tokens = [self.start_token] + base_tokens + [self.end_token]

        else:
            # Dividiamo la sequenza in 3 parti:
            #   upstream_part = base_tokens[:gene_start]
            #   gene_part     = base_tokens[gene_start:gene_end]
            #   downstream_part = base_tokens[gene_end:]
            # Poi inseriamo i token di inizio/fine gene.
            upstream_part = base_tokens[:gene_start]
            gene_part = base_tokens[gene_start:gene_end]
            downstream_part = base_tokens[gene_end:]

            # [START], upstream, [5’-PROT_START], gene, [3’-PROT_END], downstream, [END]
            full_tokens = (
                [self.start_token]
                + upstream_part
                + [self.prot_start_5]
                + gene_part
                + [self.prot_end_3]
                + downstream_part
                + [self.end_token]
            )

        # Convertiamo i token in ID
        token_ids = [self.token2id.get(tok, self.unknown_id) for tok in full_tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        """
        Converte ID numerici in una stringa di token.

        Args:
            token_ids (torch.Tensor o list): Lista di ID da convertire.

        Returns:
            str: Sequenza decodificata (token speciali inclusi).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.id2token.get(tid, self.unknown_token) for tid in token_ids]
        # Rimuoviamo eventuali [PAD] per rendere la stringa più leggibile
        tokens = [t for t in tokens if t != self.pad_token]
        return ''.join(tokens)