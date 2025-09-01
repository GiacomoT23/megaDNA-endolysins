#!/usr/bin/env python3

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
import multiprocessing as mp
import re
from src.dataset.EndolysinsDataset import EndolysinsDataset
from src.endolysinsTokenizer import endolysinsTokenizer

file_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
#                               LR SCHEDULER                                  #
###############################################################################
def get_lr_schedule(optimizer, warmup_steps, plateau_steps, decay_steps, schedule_type='warmup_only'):
    """
    Crea uno scheduler che supporta:
    1. Warm-up lineare fino al valore massimo.
    2. Plateau con LR costante per plateau_steps passi.
    3. Decadimento cosinusoidale per decay_steps passi fino a LR / 10.

    schedule_type: 'warmup_only' -> dopo warmup, LR rimane costante
                   'warmup_cosine' -> plateau + cosine decay
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warm-up lineare da 0 a LR massimo
            return float(current_step) / float(max(1, warmup_steps))

        elif schedule_type == 'warmup_only':
            return 1.0  # Mantiene lr massimo

        elif schedule_type == 'warmup_cosine':
            if warmup_steps <= current_step < warmup_steps + plateau_steps:
                return 1.0  # Plateau (lr costante)
            elif warmup_steps + plateau_steps <= current_step < warmup_steps + plateau_steps + decay_steps:
                # Decadimento cosinusoidale
                decay_progress = (current_step - warmup_steps - plateau_steps) / float(decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
                return max(0.1, cosine_decay)  # Non scendere sotto 0.1*lr max
            else:
                return 0.1
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")
    return LambdaLR(optimizer, lr_lambda)


###############################################################################
#                           CHECKPOINT SAVE/LOAD                               #
###############################################################################
def save_checkpoint(model,
                    optimizer,
                    scheduler,
                    epoch,
                    training_losses_epochs,
                    validation_losses_epochs,
                    training_accuracies,
                    validation_accuracies,
                    learning_rates,
                    path,
                    training_losses_samples,
                    best_val_loss,
                    patience_counter
                    ):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_losses_epochs': training_losses_epochs,
        'validation_losses_epochs': validation_losses_epochs,
        'training_accuracies': training_accuracies,
        'validation_accuracies': validation_accuracies,
        'learning_rates': learning_rates,
        'training_losses_samples': training_losses_samples,
        # Variabili per early stopping
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter
    }
    torch.save(state, path)
    print(f'Checkpoint saved at epoch {epoch + 1}')

def load_checkpoint(path, model, optimizer, scheduler):
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.load_state_dict(state['scheduler_state_dict'])

    start_epoch = state['epoch'] + 1
    training_losses_epochs = state['training_losses_epochs']
    validation_losses_epochs = state['validation_losses_epochs']
    training_accuracies = state['training_accuracies']
    validation_accuracies = state['validation_accuracies']
    learning_rates = state['learning_rates']
    training_losses_samples = state['training_losses_samples']

    # Early stopping
    best_val_loss = state.get('best_val_loss', float('inf'))
    patience_counter = state.get('patience_counter', 0)

    print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')
    return (start_epoch,
            training_losses_epochs,
            validation_losses_epochs,
            training_accuracies,
            validation_accuracies,
            learning_rates,
            training_losses_samples,
            best_val_loss,
            patience_counter)

def load_pretrained_checkpoint(path, model):
    state = torch.load(path)
    model.load_state_dict(state['model_state_dict'])
    print('Loaded pretrained model.')


###############################################################################
#                         DATASET & DATA SPLITTING                            #
###############################################################################
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Funzione per creare o caricare i dataset
def create_or_load_datasets(dataset, split_dir, train_ratio=0.75, val_ratio=0.15):
    train_dataset_path = os.path.join(split_dir, 'train_dataset.pth')
    val_dataset_path = os.path.join(split_dir, 'val_dataset.pth')
    test_dataset_path = os.path.join(split_dir, 'test_dataset.pth')

    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path):
        print("Loading existing datasets...")
        train_dataset = torch.load(train_dataset_path, weights_only=False)
        val_dataset = torch.load(val_dataset_path, weights_only=False)
        test_dataset = torch.load(test_dataset_path, weights_only=False) if os.path.exists(test_dataset_path) else None
    else:
        print("Creating new datasets...")
        total_samples = len(dataset)
        print(f"totalsamples: {total_samples}")

        # Calcola le dimensioni dei vari set
        train_size = int(train_ratio * total_samples)
        val_size = int(val_ratio * total_samples)
        test_size = total_samples - train_size - val_size

        # Usa random_split per dividere il dataset
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # Salva i dataset
        torch.save(train_dataset, train_dataset_path)
        torch.save(val_dataset, val_dataset_path)
        if test_size > 0:
            torch.save(test_dataset, test_dataset_path)

    return train_dataset, val_dataset, test_dataset

def find_latest_checkpoint(input_path):
    # Regular expression to match file names like 'checkpoint_123.pth'
    pattern = re.compile(r'checkpoint_(\d+)\.pth')

    max_num = -1
    latest_checkpoint = None

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


###############################################################################
#                        UTILS per Training/Logging                           #
###############################################################################
def calculate_accuracy(model, token_ids):
    model.eval()
    with torch.no_grad():
        logits = model(token_ids, return_value='logits')
        preds = rearrange(logits, 'b n c -> b c n')
        # indice del valore max
        predicted_tokens = preds[..., :-1].argmax(dim=1)
        # token_ids shiftati di 1
        token_ids = token_ids[..., 1:]
        mask = (token_ids != 0).int()
        correct_predictions = (predicted_tokens == token_ids) * mask
        num_correct = correct_predictions.sum().item()
        num_non_padding = mask.sum().item()
    del logits, preds
    return num_correct, num_non_padding

def plot_losses(training_losses, validation_losses, output_path):
    # Adesso plottiamo su x = range(len(...)) in modo che parta da 0 per la baseline
    plt.plot(range(len(training_losses)), training_losses, label='Training Loss')
    plt.plot(range(len(validation_losses)), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Losses.png'))
    plt.close()

def plot_losses_samples(training_losses, output_path, avg_steps):
    # Questo rimane con steps, perchè è un tracking su steps, non su epoche
    plt.plot([i * avg_steps for i in range(1, len(training_losses) + 1)], training_losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss (by steps)')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Losses_steps.png'))
    plt.close()

def plot_accuracies(training_accuracies, validation_accuracies, output_path):
    # Anche qui partiamo da x=0
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(training_accuracies)), training_accuracies, label='Training Accuracy')
    plt.plot(range(len(validation_accuracies)), validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
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

def save_results(training_losses_epochs, validation_losses_epochs, training_accuracies, validation_accuracies, learning_rates,
                 output_path, training_losses_samples):
    np.savetxt(os.path.join(output_path, 'training_losses_epochs.txt'), training_losses_epochs, delimiter=',')
    np.savetxt(os.path.join(output_path, 'validation_losses_epochs.txt'), validation_losses_epochs, delimiter=',')
    np.savetxt(os.path.join(output_path, 'training_accuracies.txt'), training_accuracies, delimiter=',')
    np.savetxt(os.path.join(output_path, 'validation_accuracies.txt'), validation_accuracies, delimiter=',')
    np.savetxt(os.path.join(output_path, 'learning_rates.txt'), learning_rates, delimiter=',')
    np.savetxt(os.path.join(output_path, 'training_losses_samples.txt'), training_losses_samples, delimiter=',')


###############################################################################
#                                MAIN TRAINING                                #
###############################################################################
def run_train(dataset_path, config_path, output_path, device_name, new_tokenizer, model_folder, checkpoint_number, nkeep):

    # Carica config
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

    apply_padding = False if BATCH_SIZE == 1 else True
    add_gene_delimiters = new_tokenizer

    tokenizer = endolysinsTokenizer()
    dataset = EndolysinsDataset(dataset_path, tokenizer, add_gene_delimiters, apply_padding, keep_upstream=nkeep)

    # Splits
    splits_dir = os.path.join(output_path, "splits")
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)
    train_dataset, val_dataset, test_dataset = create_or_load_datasets(dataset, splits_dir, TRAIN_RATIO, VAL_RATIO)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, worker_init_fn=seed_worker)

    # Cartella per i checkpoint
    checkpoint_finetuning_dir = os.path.join(output_path, 'checkpoints_finetuning')
    if not os.path.exists(checkpoint_finetuning_dir):
        os.makedirs(checkpoint_finetuning_dir)
    checkpoint_finetuning_path = find_latest_checkpoint(checkpoint_finetuning_dir)

    checkpoint_pretrained_model = os.path.join(model_folder, "checkpoints", f"checkpoint_{checkpoint_number}.pth")

    num_tokens = 12 if add_gene_delimiters else 8

    # Modello
    model = MEGADNA(
        num_tokens=num_tokens,
        dim=(512, 256, 196),
        depth=(8, 8, 8),
        max_seq_len=(128, 64, 16),
        flash_attn=False,
        pad_id=tokenizer.pad_id
    ).to(device_name)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_lr_schedule(
        optim,
        WARMUP_STEPS,
        plateau_steps=PLATEAU_STEPS,
        decay_steps=DECAY_STEPS,
        schedule_type=SCHEDULE_TYPE
    )

    # Variabili di tracking
    if checkpoint_finetuning_path:
        (start_epoch,
         training_losses_epochs,
         validation_losses_epochs,
         training_accuracies,
         validation_accuracies,
         learning_rates,
         training_losses_samples,
         best_val_loss,
         patience_counter) = load_checkpoint(checkpoint_finetuning_path, model, optim, scheduler)
    else:
        start_epoch = 0
        training_losses_epochs = []
        validation_losses_epochs = []
        training_accuracies = []
        validation_accuracies = []
        learning_rates = []
        training_losses_samples = []
        best_val_loss = float('inf')    # per early stopping
        patience_counter = 0           # conta quante epoche consecutive senza miglioramento

        # Carico il modello pretrained come base
        load_pretrained_checkpoint(checkpoint_pretrained_model, model)

    print(f"Start epoch: {start_epoch}")
    print(f"Total epochs to run: {EPOCHS}\n")

    ############################################################################
    #  Valutazione iniziale (Epoch 0) prima di qualunque step di training      #
    ############################################################################
    if start_epoch==0:
        print("Evaluating initial training/validation loss (baseline before finetuning)...")
        model.eval()
        # 1) Calcolo training loss su tutto il train_loader
        baseline_training_loss = 0.0
        total_correct_train = 0
        total_train = 0
        with torch.no_grad():
            for i, (token_ids, _) in enumerate(train_loader):
                token_ids = token_ids.to(device_name)
                loss = model(token_ids, return_value='loss')
                baseline_training_loss += loss.item()

                # se vuoi anche la training accuracy sullo "stato base"
                # fallo come fa la routine standard
                num_correct, num_non_padding = calculate_accuracy(model, token_ids)
                total_correct_train += num_correct
                total_train += num_non_padding

        baseline_training_loss /= len(train_loader)
        baseline_training_acc = (total_correct_train / total_train) if total_train > 0 else 0.0

        # 2) Calcolo validation loss su tutto il val_loader
        baseline_validation_loss = 0.0
        total_correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i, (token_ids, _) in enumerate(val_loader):
                token_ids = token_ids.to(device_name)
                loss_val = model(token_ids, return_value='loss')
                baseline_validation_loss += loss_val.item()

                # accuracy
                num_correct, num_non_padding = calculate_accuracy(model, token_ids)
                total_correct_val += num_correct
                total_val += num_non_padding

        baseline_validation_loss /= len(val_loader)
        baseline_validation_acc = (total_correct_val / total_val) if total_val > 0 else 0.0

        print(f"Baseline Training Loss: {baseline_training_loss}, Training Acc: {baseline_training_acc*100:.2f}%")
        print(f"Baseline Validation Loss: {baseline_validation_loss}, Validation Acc: {baseline_validation_acc*100:.2f}%\n")

        # Inseriamo questi valori negli array come "epoch 0"
        training_losses_epochs.append(baseline_training_loss)
        validation_losses_epochs.append(baseline_validation_loss)
        training_accuracies.append(baseline_training_acc)
        validation_accuracies.append(baseline_validation_acc)

    ############################################################################
    #                              TRAINING LOOP                               #
    ############################################################################
    for epoch in tqdm(range(start_epoch, EPOCHS), desc='Epochs'):
        model.train()
        epoch_training_loss = 0
        total_correct_train = 0
        total_train = 0
        training_loss_last_samples = 0
        torch.cuda.empty_cache()
        gc.collect()
        optim.zero_grad()

        # --- BATCH LOOP ---
        for i, (token_ids, sequence_id) in enumerate(train_loader):
            token_ids = token_ids.to(device_name)

            # accuracy (training) ogni tot step
            if (i % TRAINING_ACCURACY_EVERY) == 0:
                num_correct, num_non_padding = calculate_accuracy(model, token_ids)
                total_correct_train += num_correct
                total_train += num_non_padding

            loss = model(token_ids, return_value='loss')
            loss.backward()
            clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

            if (i + 1) % GRADIENT_ACCUMULATE_EVERY == 0:
                optim.step()
                optim.zero_grad()

            scheduler.step()
            current_lr = optim.param_groups[0]['lr']
            learning_rates.append(current_lr)

            training_loss_last_samples += loss.item()
            if (i + 1) % STEPS_AVG_LOSS == 0:
                training_losses_samples.append(training_loss_last_samples / STEPS_AVG_LOSS)
                training_loss_last_samples = 0

        # Se rimane un "avanzo" di batch < GRADIENT_ACCUMULATE_EVERY, facciamo step
        if (i + 1) % GRADIENT_ACCUMULATE_EVERY != 0:
            optim.step()
            optim.zero_grad()

        # Salviamo la media delle ultime samples se non è un multiplo perfetto
        if (i + 1) % STEPS_AVG_LOSS != 0:
            n_last_samples = (i + 1) % STEPS_AVG_LOSS
            training_losses_samples.append(training_loss_last_samples / n_last_samples)
            training_loss_last_samples = 0

        # Fine epoch -> calcolo training loss su tutto il train_loader
        model.eval()
        with torch.no_grad():
            for token_ids, _ in train_loader:
                token_ids = token_ids.to(device_name)
                loss = model(token_ids, return_value='loss')
                epoch_training_loss += loss.item()

        avg_epoch_training_loss = epoch_training_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss (full dataset): {avg_epoch_training_loss:.4f}')

        training_accuracy = total_correct_train / total_train if total_train > 0 else 0
        print(f'Epoch {epoch+1}/{EPOCHS}, Training Accuracy: {training_accuracy * 100:.2f}%')

        training_losses_epochs.append(avg_epoch_training_loss)
        training_accuracies.append(training_accuracy)

        # --- Validation ---
        model.eval()
        epoch_validation_loss = 0
        total_correct_val = 0
        total_val = 0
        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            for i, (token_ids, sequence_id) in enumerate(val_loader):
                token_ids = token_ids.to(device_name)
                loss_val = model(token_ids, return_value='loss')
                epoch_validation_loss += loss_val.item()

                # accuracy ogni tot step
                if (i + 1) % VALIDATION_ACCURACY_EVERY == 0:
                    num_correct, num_non_padding = calculate_accuracy(model, token_ids)
                    total_correct_val += num_correct
                    total_val += num_non_padding

        avg_epoch_validation_loss = epoch_validation_loss / len(val_loader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_epoch_validation_loss:.4f}')
        validation_losses_epochs.append(avg_epoch_validation_loss)

        validation_accuracy = total_correct_val / total_val if total_val > 0 else 0
        print(f'Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {validation_accuracy * 100:.2f}%')
        validation_accuracies.append(validation_accuracy)

        # --- Early Stopping Check ---
        if avg_epoch_validation_loss < best_val_loss:
            best_val_loss = avg_epoch_validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:  # ad esempio
                print(f"Early stopping at epoch {epoch+1}, no improvement in validation loss for {patience_counter} consecutive epochs.")
                # Salviamo il checkpoint finale e poi usciamo
                new_checkpoint_path = os.path.join(checkpoint_finetuning_dir, f'checkpoint_{epoch + 1}.pth')
                save_checkpoint(model, optim, scheduler, epoch,
                                training_losses_epochs, validation_losses_epochs,
                                training_accuracies, validation_accuracies,
                                learning_rates, new_checkpoint_path,
                                training_losses_samples,
                                best_val_loss,
                                patience_counter)
                break

        # --- Salvataggio checkpoint di fine epoca ---
        new_checkpoint_path = os.path.join(checkpoint_finetuning_dir, f'checkpoint_{epoch + 1}.pth')
        save_checkpoint(model, optim, scheduler, epoch,
                        training_losses_epochs, validation_losses_epochs,
                        training_accuracies, validation_accuracies,
                        learning_rates, new_checkpoint_path,
                        training_losses_samples,
                        best_val_loss,
                        patience_counter)

        # Aggiorna i plot
        plot_losses(training_losses_epochs, validation_losses_epochs, output_path)
        plot_accuracies(training_accuracies, validation_accuracies, output_path)
        plot_losses_samples(training_losses_samples, output_path, STEPS_AVG_LOSS)

    # Fuori dal loop (caso training finito o early stopping)
    plot_losses(training_losses_epochs, validation_losses_epochs, output_path)
    plot_accuracies(training_accuracies, validation_accuracies, output_path)
    plot_losses_samples(training_losses_samples, output_path, STEPS_AVG_LOSS)
    plot_learning_rate(learning_rates, output_path)
    save_results(training_losses_epochs, validation_losses_epochs,
                 training_accuracies, validation_accuracies,
                 learning_rates, output_path,
                 training_losses_samples)

###############################################################################
#                                 __main__                                    #
###############################################################################
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help="input fasta file", required=True, dest='input')
    argparser.add_argument('-f', '--foldertofinetune', help='folder containing model to finetune', dest='folder', type=str, required=True)
    argparser.add_argument('-n', '--checkpointnumber', help='checkpoint to finetune', dest='checknumber', type=int, required=True)
    argparser.add_argument('-o', '--output', help='output directory', dest='output', type=str, required=False, default=None)
    argparser.add_argument('-p', '--platform', help='platform', dest='platform', type=str, required=False, default='CPU', choices=['CUDA', 'CPU'])
    argparser.add_argument('-x', '--gpu_index', help='gpu device index', dest='gpu_index', type=str, required=False, default=None)
    argparser.add_argument('-c', '--config', help='config yaml file', dest='config', type=str, required=False, default=None)
    argparser.add_argument('-t', '--tokenizer', help='which tokenizer was used', dest='tokenizer', type=str, required=False, default='old', choices=['old', 'new'])
    argparser.add_argument('-k', '--nkeep', help='how many nucleotides to keep upstream', dest='nkeep', type=int, required=True, default=0)

    args = argparser.parse_args()

    # Set device
    if args.platform == 'CUDA' and args.gpu_index is not None:
        device_name = f'cuda:{args.gpu_index}'
    else:
        device_name = args.platform.lower()

    # Output
    output_path = file_path if args.output is None else args.output
    if os.path.isfile(output_path):
        raise Exception('Must be a directory')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Config
    config_name = 'configfinetuning' if args.config is None else args.config
    config_name = config_name if config_name.endswith('.yaml') else config_name + '.yaml'
    config_path = os.path.join(file_path, 'config', config_name) if args.config is None else config_name
    print('Using config file in {}'.format(config_path))

    # Tokenizer
    new_tokenizer = (args.tokenizer == 'new')

    mp.set_start_method('spawn')
    run_train(args.input, config_path, output_path, device_name, new_tokenizer, args.folder, args.checknumber, args.nkeep)