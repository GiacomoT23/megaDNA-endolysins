import os
import random
import torch.utils
import torch.utils.data
import yaml

from src.dataset.FastaDataset import FastaDataset
from src.config import Config
from src.megaDNA import MEGADNA
from src.tokenizer import Tokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

import gc
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.nn.utils import clip_grad_norm_
import argparse
import torch.multiprocessing as mp
import re
from src.dataset.newFastaDataset import newFastaDataset
from src.newTokenizer import newTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


file_path = os.path.dirname(os.path.realpath(__file__))

def ddp_setup(rank, world_size, selected_gpus):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    use_cpu = os.environ.get("USE_CPU", "0") == "1"
    if not use_cpu and torch.cuda.is_available():
        #selected_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
        torch.cuda.set_device(selected_gpus[rank])  # Imposta la GPU corretta per il processo
        #selected_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(',')))
        #torch.cuda.set_device(selected_gpus[rank])
        backend = "nccl"
    else:
        backend = "gloo"
    init_process_group(backend=backend, rank=rank, world_size=world_size)


class Trainer:
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 optimizer,
                 checkpoint_path,
                 output_path, 
                 warmup_steps,
                 plateau_steps,
                 decay_steps,
                 schedule_type,
                 steps_avg_loss,
                 gradient_clip_norm,
                 training_accuracy_every,
                 gradient_accumulate_every,
                 validation_accuracy_every,
                 rank
    ):
        self.rank = rank
        # Verifica se l'utente ha scelto di usare la CPU
        use_cpu = os.environ.get("USE_CPU", "0") == "1"
        if not use_cpu and torch.cuda.is_available():
            # Se l'utente NON ha scelto la CPU e ci sono GPU disponibili
            #selected_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.gpu_id = torch.cuda.current_device()  # Seleziona la GPU giusta per il processo
            print(f"GPU: {self.gpu_id}")
            model = model.to(self.gpu_id)
            self.model = DDP(model, device_ids=[self.gpu_id])
        else:
            # Se l'utente ha forzato la CPU o non ci sono GPU disponibili
            self.gpu_id = "cpu"
            model = model.to("cpu")
            self.model = DDP(model)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = get_lr_schedule(optimizer, warmup_steps, plateau_steps, decay_steps, schedule_type)
        self.start_epoch = 0
        self.training_losses_epochs = []
        self.validation_losses_epochs = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.learning_rates = []
        self.training_losses_samples = []
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print("Loading checkpoint")
                self.load_checkpoint(checkpoint_path)
        self.output_path = output_path
        self.steps_avg_loss = steps_avg_loss
        self.gradient_clip_norm = gradient_clip_norm
        self.training_accuracy_every = training_accuracy_every
        self.gradient_accumulate_every = gradient_accumulate_every
        self.validation_accuracy_every = validation_accuracy_every


    def load_checkpoint(self, checkpoint_path):
        device = torch.device(self.gpu_id if torch.cuda.is_available() and self.gpu_id != "cpu" else "cpu")
        # Carica il checkpoint nella posizione corretta
        state = torch.load(checkpoint_path, map_location=device)
        self.model.module.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.start_epoch = state['epoch'] + 1
        self.training_losses_epochs = state['training_losses_epochs']
        self.validation_losses_epochs = state['validation_losses_epochs']
        self.training_accuracies = state['training_accuracies']
        self.validation_accuracies = state['validation_accuracies']
        self.learning_rates = state['learning_rates']
        self.training_losses_samples = state['training_losses_samples']
        print(f'Checkpoint loaded, resuming training from epoch {self.start_epoch + 1}')
        return

    def save_checkpoint(self, epoch, path):
        state = {
        'epoch': epoch,
        'model_state_dict': self.model.module.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'training_losses_epochs': self.training_losses_epochs,
        'validation_losses_epochs': self.validation_losses_epochs,
        'training_accuracies': self.training_accuracies,
        'validation_accuracies': self.validation_accuracies,
        'learning_rates': self.learning_rates,
        'training_losses_samples': self.training_losses_samples,
        }
        torch.save(state, path)
        print(f'Checkpoint saved at epoch {epoch + 1}')

    def run_epoch(self, epoch):
        self.model.train()
        epoch_training_loss = 0
        total_correct_train = 0
        total_train = 0
        training_loss_last_samples = 0
        torch.cuda.empty_cache()
        gc.collect()
        self.optimizer.zero_grad()
        self.train_data.sampler.set_epoch(epoch)
        print(f"GPU: {self.gpu_id}, Epoch: {epoch}, Batch size: {next(iter(self.train_data))[0].shape[0]} Steps: {len(self.train_data)}")
        for i, (token_ids, sequence_id) in enumerate(self.train_data):
            token_ids = token_ids.to(self.gpu_id)
            #Qui ho messo i anziché i+1 per fare in modo che il calcolo dell'accuracy avvenga sempre subito dopo aver
            #aggiornato i pesi e non prima. Non credo sia necessario ma non si sa mai. Ciò è valido ovviamente solo se si imposta
            #TRAINING_ACCURACY_EVERY come un multiplo di GRADIENT_ACCUMULATE_EVERY.
            if(i) % self.training_accuracy_every == 0:
                num_correct, num_non_padding = calculate_accuracy(self.model, token_ids)
                total_correct_train += num_correct
                total_train += num_non_padding
            self.model.train()
            loss = self.model(token_ids, return_value='loss')
            # Backpropagation
            loss.backward()
            # Clipping dei gradienti
            clip_grad_norm_(self.model.module.parameters(), self.gradient_clip_norm)
            if (i + 1) % self.gradient_accumulate_every == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            # Memorizza il valore corrente del learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            epoch_training_loss += loss.item()
            training_loss_last_samples += loss.item()
            if (i+1) % self.steps_avg_loss == 0:
                # Converti training_loss_last_samples e numero di sample in tensori su GPU
                training_loss_last_samples_tensor = torch.tensor(training_loss_last_samples, dtype=torch.float, device=self.gpu_id)
                num_samples_in_avg_tensor = torch.tensor(self.steps_avg_loss, dtype=torch.float, device=self.gpu_id)
                # Sincronizza tra tutti i processi DDP
                torch.distributed.all_reduce(training_loss_last_samples_tensor, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(num_samples_in_avg_tensor, op=torch.distributed.ReduceOp.SUM)
                # Calcola la loss media corretta rispetto al numero totale di sample
                avg_loss_sampled = training_loss_last_samples_tensor.item() / num_samples_in_avg_tensor.item()
                self.training_losses_samples.append(avg_loss_sampled)
                training_loss_last_samples = 0
        # Perform a final optimization step if needed
        if (i + 1) % self.gradient_accumulate_every != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if(i+1) % self.steps_avg_loss != 0:
            # Ultimi sample rimanenti
            n_last_samples = (i+1) % self.steps_avg_loss
            # Converti in tensori su GPU
            training_loss_last_samples_tensor = torch.tensor(training_loss_last_samples, dtype=torch.float, device=self.gpu_id)
            num_samples_last_tensor = torch.tensor(n_last_samples, dtype=torch.float, device=self.gpu_id)
            # Sincronizza tra tutti i processi DDP
            torch.distributed.all_reduce(training_loss_last_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_samples_last_tensor, op=torch.distributed.ReduceOp.SUM)
            # Calcola la media globale
            avg_loss_last_samples = training_loss_last_samples_tensor.item() / num_samples_last_tensor.item()
            self.training_losses_samples.append(avg_loss_last_samples)
            training_loss_last_samples = 0
        # Converti epoch_training_loss e numero di campioni in tensori su GPU
        epoch_training_loss_tensor = torch.tensor(epoch_training_loss, dtype=torch.float, device=self.gpu_id)
        num_samples_tensor = torch.tensor(len(self.train_data), dtype=torch.float, device=self.gpu_id)
        # Sincronizza la somma delle loss e il numero totale di campioni tra tutti i processi
        torch.distributed.all_reduce(epoch_training_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        # Ora la loss totale è condivisa tra tutti i processi, quindi calcoliamo la media corretta
        avg_epoch_training_loss = epoch_training_loss_tensor.item() / num_samples_tensor.item()
        # Solo il processo master (gpu_id == 0) stampa e salva il valore corretto
        if self.rank == 0:
            print(f'Epoch {epoch + 1}, Training Loss: {avg_epoch_training_loss}')
            self.training_losses_epochs.append(avg_epoch_training_loss)
        # Sincronizza i risultati tra tutti i processi DDP
        total_correct_train_tensor = torch.tensor(total_correct_train, dtype=torch.float, device=self.gpu_id)
        total_train_tensor = torch.tensor(total_train, dtype=torch.float, device=self.gpu_id)
        torch.distributed.all_reduce(total_correct_train_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_train_tensor, op=torch.distributed.ReduceOp.SUM)
        # Ora tutti i processi hanno lo stesso valore per total_correct_train e total_train
        if total_train_tensor.item() > 0:
            training_accuracy = total_correct_train_tensor.item() / total_train_tensor.item()
        else:
            training_accuracy = 0.0
        print(f'Epoch {epoch+1}, Training Accuracy: {training_accuracy * 100:.2f}%')
        self.training_accuracies.append(training_accuracy)

        # Validation phase
        self.model.eval()
        epoch_validation_loss = 0
        total_correct_val = 0
        total_val = 0
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            for i, (token_ids, sequence_id) in enumerate(self.val_data):
                token_ids = token_ids.to(self.gpu_id)
                loss_val = self.model(token_ids, return_value='loss')
                if(i + 1) % self.validation_accuracy_every == 0:
                    num_correct, num_non_padding = calculate_accuracy(self.model, token_ids)
                    total_correct_val += num_correct
                    total_val += num_non_padding
                epoch_validation_loss += loss_val.item()
        epoch_validation_loss_tensor = torch.tensor(epoch_validation_loss, dtype=torch.float, device=self.gpu_id)
        num_val_samples_tensor = torch.tensor(len(self.val_data), dtype=torch.float, device=self.gpu_id)
        torch.distributed.all_reduce(epoch_validation_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(num_val_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        avg_epoch_validation_loss = epoch_validation_loss_tensor.item() / num_val_samples_tensor.item()
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_epoch_validation_loss}')
        self.validation_losses_epochs.append(avg_epoch_validation_loss)
        # Sincronizza i risultati di validazione tra tutti i processi DDP
        total_correct_val_tensor = torch.tensor(total_correct_val, dtype=torch.float, device=self.gpu_id)
        total_val_tensor = torch.tensor(total_val, dtype=torch.float, device=self.gpu_id)
        torch.distributed.all_reduce(total_correct_val_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_val_tensor, op=torch.distributed.ReduceOp.SUM)
        if total_val_tensor.item() > 0:
            validation_accuracy = total_correct_val_tensor.item() / total_val_tensor.item()
        else:
            validation_accuracy = 0.0
        print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy * 100:.2f}%')
        self.validation_accuracies.append(validation_accuracy)

    def train(self, max_epochs):
        for epoch in range(self.start_epoch, max_epochs):
            self.run_epoch(epoch)
            #SALVA CHECKPOINT
            if self.rank == 0:    
                new_checkpoint_path = os.path.join(self.output_path, 'checkpoints', f'checkpoint_{epoch + 1}.pth')
                self.save_checkpoint(epoch, new_checkpoint_path)
                #Aggiorna plot
                plot_losses(self.training_losses_epochs, self.validation_losses_epochs, self.output_path)
                plot_accuracies(self.training_accuracies, self.validation_accuracies, self.output_path)
                plot_losses_samples(self.training_losses_samples, self.output_path, self.steps_avg_loss)

def load_train_objs(splits_dir, num_tokens, pad_id, learning_rate):
    training_set, validation_set, test_set = load_datasets(splits_dir)
    model = MEGADNA(
        num_tokens=num_tokens,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=pad_id
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return training_set, validation_set, test_set, model, optimizer

def prepare_dataloader(dataset, batch_size, rank, world_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle = False,
                      sampler= DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True))

def get_lr_schedule(optimizer, warmup_steps, plateau_steps, decay_steps, schedule_type='warmup_only'):
    """
    Crea uno scheduler che supporta:
    1. Warm-up lineare fino al valore massimo.
    2. Plateau con LR costante per plateau_steps passi.
    3. Decadimento cosinusoidale per decay_steps passi fino a LR / 10.
    
    Args:
        optimizer: Ottimizzatore PyTorch
        warmup_steps: Numero di passi per il warm-up
        plateau_steps: Numero di passi per mantenere LR costante
        decay_steps: Numero di passi per il cosine decay
        schedule_type: 'warmup_only' per mantenere costante dopo il warm-up, 'warmup_cosine' per applicare decay
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # Warm-up lineare
        
        elif schedule_type == 'warmup_only':
            return 1.0  # Mantiene il learning rate massimo dopo il warm-up
        
        elif schedule_type == 'warmup_cosine':
            if warmup_steps <= current_step < warmup_steps + plateau_steps:
                return 1.0  # Plateau con LR costante
            
            elif warmup_steps + plateau_steps <= current_step < warmup_steps + plateau_steps + decay_steps:
                # Decay cosinusoidale
                decay_progress = (current_step - warmup_steps - plateau_steps) / float(decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
                return max(0.1, cosine_decay)  # Assicura che non scenda sotto il 10%

            else:
                return 0.1  # Mantiene il valore minimo dopo il decay

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return LambdaLR(optimizer, lr_lambda)

def create_and_save_datasets_if_needed(dataset, split_dir, train_ratio=0.75, val_ratio=0.15, generator = None):
    train_dataset_path = os.path.join(split_dir, 'train_dataset.pth')
    val_dataset_path = os.path.join(split_dir, 'val_dataset.pth')
    test_dataset_path = os.path.join(split_dir, 'test_dataset.pth')
    if not (os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path)):
        print("Creating new datasets...")
        total_samples = len(dataset)
        print(f"totalsamples: {total_samples}")

        # Calcola le dimensioni dei vari set
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # Usa random_split per dividere il dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = generator)

        # Salva i dataset
        torch.save(train_dataset, train_dataset_path)
        torch.save(val_dataset, val_dataset_path)
        if test_size > 0:
            torch.save(test_dataset, test_dataset_path)

    return

# Funzione per creare o caricare i dataset
def load_datasets(split_dir):
    train_dataset_path = os.path.join(split_dir, 'train_dataset.pth')
    val_dataset_path = os.path.join(split_dir, 'val_dataset.pth')
    test_dataset_path = os.path.join(split_dir, 'test_dataset.pth')

    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
        print("Loading existing datasets...")
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        val_dataset = torch.load(val_dataset_path, weights_only=False)
        test_dataset = torch.load(test_dataset_path, weights_only=False) if os.path.exists(test_dataset_path) else None
    else:
        print("Errore")
    return train_dataset, val_dataset, test_dataset

def find_latest_checkpoint(input_path):
    # Regular expression to match file names like 'checkpoint_123.pth'
    pattern = re.compile(r'checkpoint_(\d+)\.pth')
    
    max_num = -1
    latest_checkpoint = None

    # Walk through all files in the folder
    for filename in os.listdir(input_path):
        match = pattern.match(filename)
        if match:
            checkpoint_num = int(match.group(1))
            if checkpoint_num > max_num:
                max_num = checkpoint_num
                latest_checkpoint = filename

    if latest_checkpoint:
        latest_checkpoint = os.path.join(input_path, latest_checkpoint)
                
    return latest_checkpoint

def calculate_accuracy(model, token_ids):
    model.eval()
    with torch.no_grad():
        logits = model(token_ids, return_value='logits')
        preds = rearrange(logits, 'b n c -> b c n')
        # Ottieni le predizioni prendendo l'indice del valore massimo
        predicted_tokens = preds[..., :-1].argmax(dim=1)  
        # Tolgo il padding introdotto dal modello 
        #predicted_tokens = predicted_tokens[..., :token_ids.shape[1]]
        token_ids = token_ids[..., 1:]
        mask = (token_ids != 0).int()
        # Ottengo matrice con 1 in posizioni azzeccate e 0 in predizioni sbagliate e/o padding
        correct_predictions = (predicted_tokens == token_ids) * mask 
        # Sommiamo su batch e sequenza SENZA loop
        num_correct = correct_predictions.sum().item()
        num_non_padding = mask.sum().item()
    del logits, preds
    return num_correct, num_non_padding

def plot_losses(training_losses, validation_losses, output_path):
    plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Losses.png'))
    plt.close()

def plot_losses_samples(training_losses, output_path, avg_steps):
    plt.plot([i * avg_steps for i in range(1, len(training_losses) + 1)], training_losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Losses_steps.png'))
    plt.close()

def plot_accuracies(training_accuracies, validation_accuracies, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Accuracies.png'))
    plt.close()

def plot_learning_rate(learning_rates, output_path):
    plt.figure()
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, label='Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'learning_rate.png'))
    plt.close()

def save_results(training_losses_epochs, validation_losses_epochs, training_accuracies, validation_accuracies, learning_rates, output_path, 
                 training_losses_samples):
    np.savetxt(os.path.join(output_path, 'training_losses_epochs.txt'), training_losses_epochs, delimiter=',')
    np.savetxt(os.path.join(output_path, 'validation_losses_epochs.txt'), validation_losses_epochs, delimiter=',')
    np.savetxt(os.path.join(output_path, 'training_accuracies.txt'), training_accuracies, delimiter=',')
    np.savetxt(os.path.join(output_path, 'validation_accuracies.txt'), validation_accuracies, delimiter=',')
    np.savetxt(os.path.join(output_path, 'learning_rates.txt'), learning_rates, delimiter=',')
    np.savetxt(os.path.join(output_path, 'training_losses_samples.txt'), training_losses_samples, delimiter=',')

def run_train(rank :int, world_size :int, output_path, new_tokenizer, pad_id, learning_rate, batch_size, warmup_steps, plateau_steps, 
              decay_steps, schedule_type, steps_avg_loss, gradient_clip_norm, training_accuracy_every, gradient_accumulate_every, 
              validation_accuracy_every, total_epochs, selected_gpus):
    ddp_setup(rank, world_size, selected_gpus)
    splits_dir = os.path.join(output_path, "splits")
    num_tokens = 12 if new_tokenizer else 8
    training_set, validation_set, test_set, model, optimizer = load_train_objs(splits_dir, num_tokens, pad_id, learning_rate)
    train_data = prepare_dataloader(training_set, batch_size, rank, world_size)
    val_data = prepare_dataloader(validation_set, batch_size, rank, world_size)
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    trainer = Trainer(model, train_data, val_data, optimizer, checkpoint_path, output_path, warmup_steps, plateau_steps, decay_steps,
                       schedule_type, steps_avg_loss, gradient_clip_norm, training_accuracy_every, gradient_accumulate_every, 
                       validation_accuracy_every, rank)
    trainer.train(total_epochs)
    if trainer.rank==0:
        plot_losses(trainer.training_losses_epochs, trainer.validation_losses_epochs, output_path)
        plot_accuracies(trainer.training_accuracies, trainer.validation_accuracies, output_path)
        plot_losses_samples(trainer.training_losses_samples, output_path, steps_avg_loss)
        plot_learning_rate(trainer.learning_rates, output_path)
        save_results(trainer.training_losses_epochs, trainer.validation_losses_epochs, trainer.training_accuracies, trainer.validation_accuracies, trainer.learning_rates, 
                    output_path, trainer.training_losses_samples)
    destroy_process_group()

if __name__ == '__main__':
    
    # Initialize the argument parser
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help="input fasta file", required=True, dest='input')
    argparser.add_argument('-o', '--output', help='output directory', dest='output', type=str, required=False, default=None)
    argparser.add_argument('-p', '--platform', help='platform', dest='platform', type=str, required=False, default='CPU', choices=['CUDA', 'CPU'])
    #LA LISTA DI INDICI DEVONO ESSERE NUMERI SEPARATI DA VIRGOLE SENZA SPAZI
    argparser.add_argument('-x', '--gpu_index', help='gpu device index', dest='gpu_index', type=str, required=False, default=None)
    argparser.add_argument('-c', '--config', help='config yaml file', dest='config', type=str, required=False, default=None)
    argparser.add_argument('-t', '--tokenizer', help='which tokenizer to use', dest='tokenizer', type=str, required=False, default='old', choices=['old', 'new'])

    # Parse arguments
    args = argparser.parse_args()
    # Set device name
    if args.platform == 'CPU':
        os.environ["USE_CPU"] = "1"  # Indica di usare la CPU anche se ci sono GPU disponibili
        
    # Set output path
    output_path = file_path if args.output is None else args.output
    if os.path.isfile(output_path):
        raise Exception('Must be a directory')
    # Create output if needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Load the config file
    config_name = 'config' if args.config is None else args.config
    config_name = config_name if config_name.endswith('.yaml') else config_name + '.yaml'
    config_path = os.path.join(file_path, 'config', config_name) if args.config is None else config_name
    print('Using config file in {}'.format(config_path))

    if args.tokenizer == 'old' :
        new_tokenizer = False
    else:
        new_tokenizer = True
    
    # Load configurations
    configs = Config.from_file(config_path)

    LEARNING_RATE = configs.get("learning_rate", 0.001)
    EPOCHS = configs.get("epochs", 3)
    BATCH_SIZE = configs.get("batch_size", 1)
    GRADIENT_ACCUMULATE_EVERY = configs.get("gradient_accumulate_every", 1)
    WARMUP_STEPS = configs.get("warmup_steps", 50000)
    PLATEAU_STEPS = configs.get("plateau_steps", 50000)
    DECAY_STEPS = configs.get("decay_steps", 50000)
    SCHEDULE_TYPE = configs.get("schedule_type", "warmup_only")
    GRADIENT_CLIP_NORM = configs.get("gradient_clip_norm", 0.5)
    TRAINING_ACCURACY_EVERY = configs.get("training_accuracy_every", 48)
    VALIDATION_ACCURACY_EVERY = configs.get("validation_accuracy_every", 8)
    TRAIN_RATIO, VAL_RATIO = tuple(configs.get("splits_ratios", [0.75, 0.15]))
    STEPS_AVG_LOSS = configs.get("steps_avg_loss", 100)

    if new_tokenizer:
        tokenizer = newTokenizer()
        annotations_dir = os.path.join(output_path, "annotations")
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)
        dataset = newFastaDataset(args.input, tokenizer, annotations_dir, apply_padding=True)
    else:
        tokenizer = Tokenizer()
        dataset = FastaDataset(args.input, tokenizer, pad_to_max_length=True)

    # Create or load data splits
    splits_dir = os.path.join(output_path, "splits")
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    create_and_save_datasets_if_needed(dataset, splits_dir, TRAIN_RATIO, VAL_RATIO)

    use_cpu = os.environ.get("USE_CPU", "0") == "1"
    if not use_cpu and torch.cuda.is_available():
        if args.gpu_index is not None:
            selected_gpus = list(map(int, args.gpu_index.split(',')))  # Converte la stringa "0,1" in [0, 1]
            #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))  # Imposta le GPU visibili
            #print(os.environ["CUDA_VISIBLE_DEVICES"])
            world_size = len(selected_gpus)  # Imposta il numero di processi da lanciare
        else:
            world_size = torch.cuda.device_count()  # Usa tutte le GPU disponibili se non specificato
    else:
        world_size = 1
    '''
    if not use_cpu and torch.cuda.is_available():
        training_accuracy_every = TRAINING_ACCURACY_EVERY / (BATCH_SIZE * world_size)
    else:
        training_accuracy_every = TRAINING_ACCURACY_EVERY / BATCH_SIZE
    '''
    training_accuracy_every = TRAINING_ACCURACY_EVERY
    mp.spawn(run_train, args=(world_size, output_path, new_tokenizer, tokenizer.pad_id, LEARNING_RATE, BATCH_SIZE, 
                              WARMUP_STEPS, PLATEAU_STEPS, DECAY_STEPS, SCHEDULE_TYPE, STEPS_AVG_LOSS, GRADIENT_CLIP_NORM, 
                              training_accuracy_every, GRADIENT_ACCUMULATE_EVERY, VALIDATION_ACCURACY_EVERY, EPOCHS, selected_gpus), 
                              nprocs=world_size)
