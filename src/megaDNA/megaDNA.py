import torch
import torch.nn.functional as F
from itertools import zip_longest
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
from beartype import beartype
from beartype.typing import Tuple, Union

from src.megaDNA.MEGABYTE_pytorch import MEGABYTE
from src.megaDNA.MEGABYTE_pytorch.megabyte import reduce_mult, remainder_to_mult, default, exists
from src.megaDNA.MEGABYTE_pytorch.megabyte import pack_one, unpack_one, top_k, gumbel_sample
from torch import nn, einsum



class MEGADNA(MEGABYTE):

    @beartype
    def __init__(
        self,
        *,
        num_tokens,
        dim: Union[Tuple, int],
        depth: Tuple,
        max_seq_len: Tuple,
        flash_attn: bool,
        pad_id: int = 0
    ):
        super().__init__(
            num_tokens = num_tokens,
            dim = dim,
            depth = depth,
            max_seq_len = max_seq_len,
            dim_head = 64,
            heads = 8,
            attn_dropout = 0.,
            ff_mult = 4,
            ff_dropout = 0.,
            pad_id = pad_id,
            rel_pos = False,
            pos_emb = False,
            flash_attn=flash_attn
            # flash_attn = True
        )

    def generate(
        self,
        prime=None,
        seq_len=1024,
        filter_thres=0.9,
        temperature=1.0,
        default_batch_size=1,
        stop_tokens=None  # <-- nuovo parametro opzionale
    ):
        """
        Genera fino a 'seq_len' token, ma si interrompe prima se il token generato
        appartiene a 'stop_tokens'.
        
        Args:
            prime: tensore iniziale (batch_size, lunghezza_iniziale) oppure None.
            seq_len: lunghezza massima di token da generare.
            filter_thres: soglia per il top_k.
            temperature: fattore di temperatura per gumbel_sample.
            default_batch_size: batch size usato se prime=None.
            stop_tokens: lista/insieme di token che interrompono la generazione
                        appena uno di essi è generato. Se None, non si interrompe.
        Returns:
            seq (tensor): la sequenza (batch_size, lunghezza_finale) generata.
        """
        from einops import rearrange
        device = next(self.parameters()).device

        # Se non esiste prime, crea un tensore vuoto
        if prime is None:
            seq = torch.empty((default_batch_size, 0), dtype=torch.long, device=device)
        else:
            seq = prime

        batch = seq.shape[0]

        with torch.no_grad():
            # Ciclo per generare i token rimanenti
            for _ in tqdm(range(seq_len - seq.shape[-1])):
                # Ottieni i logit dell'ultimo token
                logits = self.forward(seq, return_value='logits')[:, -1]

                # Applica il top_k filtering
                logits = top_k(logits, thres=filter_thres)

                # Campiona con gumbel_sample
                sampled = gumbel_sample(logits, dim=-1, temperature=temperature)

                # Accoda il token generato
                seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim=-1)

                # Memoria intermedia (opzionale)
                del logits, sampled

                # Se ho definito stop_tokens e il token generato è in stop_tokens => break
                if stop_tokens is not None:
                    # Nel caso batch=1 (comune), basta controllare seq[-1]
                    # Se invece c'è un batch>1, puoi decidere se interrompere appena
                    # uno dei batch genera uno stop-token, o per ciascun batch separatamente.
                    # Qui assumiamo batch=1 per semplicità:
                    last_token = seq[0, -1].item()  # token appena generato
                    if last_token in stop_tokens:
                        break

        return seq.reshape(batch, -1)


    def forward(self, ids, return_value = 'loss'):

        if return_value not in ['logits', 'embedding', 'loss']:
            raise ValueError('return_value must be one of "embedding", "logits", or "loss"')
        
        batch = ids.shape[0]

        assert ids.ndim in {2, self.stages + 1}
        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        assert prec_dims[0] <= self.max_seq_len[0], 'the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)'
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), 'all subsequent dimensions must match exactly'

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0
                
            tokens = token_emb(ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device = device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, '... m n -> ... (m n)')

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None
        hidden_states = []

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions        
        
        for stage_start_tokens, stage_tokens, transformer, proj in zip(self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections):
            stage_tokens, ps = pack_one(stage_tokens, '* n d')
            stage_start_tokens = repeat(stage_start_tokens, 'f -> b 1 f', b = stage_tokens.shape[0])

            # concat start token

            stage_tokens = torch.cat((
                stage_start_tokens,
                stage_tokens,
            ), dim = -2)

            # sum the previous hierarchy's representation
            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value = 0.)
                stage_tokens = stage_tokens + prev_stage_tokens_repr


            attended = transformer(stage_tokens)
            #hidden_states.append(attended)
            attended = unpack_one(attended, ps, '* n d')
            #ho spostato la append per fare in modo che restituisca il tensore già 'spacchettato'
            hidden_states.append(attended)

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])    

        if return_value == 'embedding':
            return hidden_states
            
        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, 'b d -> b 1 d')

        logits = logits[..., 1:, :]

        if return_value == 'logits':

            if flattened_dims:
                logits = rearrange(logits, 'b ... c -> b (...) c')
                logits = logits[:, :seq_len]

            return logits

        logits = rearrange(logits, 'b ... c -> b (...) c')

        logits = torch.cat((start_tokens, logits), dim = -2)

        preds = rearrange(logits, 'b n c -> b c n')

        labels = rearrange(ids, 'b ... -> b (...)')

        loss = F.cross_entropy(
            preds[..., :-1],
            labels,
            ignore_index = self.pad_id
        )

        return loss