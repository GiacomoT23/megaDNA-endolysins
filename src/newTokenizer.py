import torch

class newTokenizer:
    def __init__(self, vocab='ATCG'):
        self.vocab = vocab
        self.token2id = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.id2token = {idx + 1: char for idx, char in enumerate(vocab)}

        # Token speciali
        self.pad_token = '[PAD]'
        self.pad_id = 0
        self.unknown_token = 'N'
        self.unknown_id = len(vocab) + 1
        self.start_token = '[START]'
        self.start_id = len(vocab) + 2
        self.end_token = '[END]'
        self.end_id = len(vocab) + 3

        # Token per le CDS
        self.prot_start_5 = '[5’-PROT_START]'
        self.prot_start_5_id = len(vocab) + 4
        self.prot_end_3 = '[3’-PROT_END]'
        self.prot_end_3_id = len(vocab) + 5
        self.prot_start_3 = '[3’-PROT_START]'
        self.prot_start_3_id = len(vocab) + 6
        self.prot_end_5 = '[5’-PROT_END]'
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
            self.pad_id: '[PAD]',
            self.prot_start_5_id: self.prot_start_5,
            self.prot_end_3_id: self.prot_end_3,
            self.prot_start_3_id: self.prot_start_3,
            self.prot_end_5_id: self.prot_end_5
        })

    def encode(self, sequence, coding_regions=None):
        """
        Converte una sequenza in ID numerici, aggiungendo token speciali nelle coding regions.

        Args:
            sequence (str): La sequenza genomica da tokenizzare.
            coding_regions (list of tuples): Lista di tuple (start, end, strand).
            add_special_tokens (bool): Se includere [START] e [END].

        Returns:
            torch.Tensor: Tensore con gli ID dei token.
        """

        tokens = [char if char in self.vocab else self.unknown_token for char in sequence]

        if coding_regions:
            insertions = {}
            for start, end, strand in coding_regions:
                if strand == '+':
                    prot_start = self.prot_start_5
                    prot_end = self.prot_end_3
                    insertions.setdefault(start, []).append(prot_start)
                    insertions.setdefault(end, []).append(prot_end)
                else:
                    # Se lo strand è negativo, invertiamo il concetto di start e end
                    prot_start = self.prot_start_3  # Il gene parte da end
                    prot_end = self.prot_end_5      # E termina a start
                    insertions.setdefault(end, []).append(prot_start)  # Inizio CDS su end
                    insertions.setdefault(start, []).append(prot_end)  # Fine CDS su start

            new_tokens = []
            for i, token in enumerate(tokens):
                if i in insertions:
                    new_tokens.extend(insertions[i])
                new_tokens.append(token)

            if len(tokens) in insertions:
                new_tokens.extend(insertions[len(tokens)])

        else:
            new_tokens = tokens

        new_tokens = [self.start_token] + new_tokens + [self.end_token]

        # Verifica che `new_tokens` non sia vuoto prima della conversione
        if not new_tokens:
            raise ValueError(f"Errore: `new_tokens` è vuoto dopo la tokenizzazione della sequenza {sequence[:50]}...")

        token_ids = [self.token2id.get(token, self.unknown_id) for token in new_tokens]

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids):
        """
        Converte una lista di ID dei token in una stringa di sequenza.
        
        Args:
            token_ids (list): Lista di ID dei token.
            
        Returns:
            str: Sequenza decodificata.
        """
        sequence = [self.id2token.get(token_id, self.unknown_token) for token_id in token_ids]
        
        # Rimuovi eventuali token padding
        sequence = [token for token in sequence if token not in {'[PAD]'}]

        return ''.join(sequence)