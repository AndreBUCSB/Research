import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,  # Updated import
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import wandb
from transformers.integrations import WandbCallback
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import numpy as np

# ==============================
# 0. Utility Functions
# ==============================

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def contrastive_loss(embeddings, temperature=0.05):
    """
    Compute the contrastive loss using InfoNCE.

    Args:
        embeddings (torch.Tensor): Sentence embeddings of shape (batch_size, hidden_size).
        temperature (float): Temperature parameter for scaling.

    Returns:
        torch.Tensor: Contrastive loss.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = torch.arange(embeddings.size(0)).to(embeddings.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

def rtd_loss_function(rtd_logits, rtd_labels):
    """
    Compute the RTD loss.

    Args:
        rtd_logits (torch.Tensor): RTD logits of shape (batch_size*2, seq_len, vocab_size).
        rtd_labels (torch.Tensor): RTD labels of shape (batch_size*2, seq_len).

    Returns:
        torch.Tensor: RTD loss.
    """
    rtd_labels = rtd_labels.long()
    return F.cross_entropy(rtd_logits.view(-1, rtd_logits.size(-1)), rtd_labels.view(-1))

def combined_loss_function(contrastive_emb, rtd_logits, rtd_labels, rtd_weight=0.05, temperature=0.05):
    """
    Compute the combined loss from contrastive and RTD losses.

    Args:
        contrastive_emb (torch.Tensor): Sentence embeddings from original inputs.
        rtd_logits (torch.Tensor): RTD logits for both original and augmented inputs.
        rtd_labels (torch.Tensor): RTD labels indicating original (1) or replaced (0) tokens.
        rtd_weight (float): Weight for the RTD loss.
        temperature (float): Temperature parameter for contrastive loss.

    Returns:
        torch.Tensor: Combined loss.
        torch.Tensor: Contrastive loss.
        torch.Tensor: RTD loss.
    """
    # Contrastive Loss
    loss_contrastive = contrastive_loss(contrastive_emb, temperature)

    # RTD Loss
    loss_rtd = rtd_loss_function(rtd_logits, rtd_labels)

    # Total Loss
    total_loss = loss_contrastive + rtd_weight * loss_rtd
    return total_loss, loss_contrastive, loss_rtd

def augment_sentence(sentence, tokenizer, generator_model, mask_prob=0.15, max_length=512):
    """
    Generate an augmented version of the sentence by replacing masked tokens.
    """
    # Tokenize input with padding and truncation
    inputs = tokenizer(
        sentence, 
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(generator_model.device)
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Mask tokens only where attention mask is 1
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_prob).to(generator_model.device)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(generator_model.device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices = masked_indices & attention_mask.bool()  # Only mask tokens that aren't padding
    input_ids[masked_indices] = tokenizer.mask_token_id

    # Generate predictions
    with torch.no_grad():
        outputs = generator_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits

    # Replace masked tokens with predictions
    predicted_ids = torch.argmax(predictions, dim=-1)
    input_ids[masked_indices] = predicted_ids[masked_indices]

    # Decode to get the replaced sentence
    augmented_sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return augmented_sentence, masked_indices[0][:max_length].cpu().numpy()

# ==============================
# 1. Setup Logging
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# 2. Define the Dataset
# ==============================

class DAPTDataset(Dataset):
    """
    Dataset class for Domain-Adaptive Pre-Training (DAPT) using DiffCSE and RTD.
    """
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)          # Shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: (max_length)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# ==============================
# 3. Define Custom Callbacks
# ==============================

class SaveBestModelCallback(TrainerCallback):
    """
    Custom callback to save the best model based on validation loss.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.best_loss = float('inf')
        self.model = model
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Triggered at each evaluation step.
        """
        if metrics is not None:
            current_loss = metrics.get('eval_loss')
            if current_loss and current_loss < self.best_loss:
                self.best_loss = current_loss
                logger.info(f"New best loss: {self.best_loss:.4f}. Saving best model.")
                # Save model and tokenizer
                self.model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                self.tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))

# ==============================
# 4. Define the Model
# ==============================

from transformers import RobertaModel  # Ensure to import RobertaModel

class DiffCSEWithRTDModel(nn.Module):
    def __init__(self, model_name='roberta-base', rtd_loss_weight=0.05):
        super(DiffCSEWithRTDModel, self).__init__()
        # Sentence Encoder
        self.encoder = RobertaModel.from_pretrained(model_name)
        # Contrastive Head
        self.dropout = nn.Dropout(0.1)
        self.pooler = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.activation = nn.Tanh()
        # RTD Head
        self.rtd_head = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.vocab_size)
        # Loss Weight
        self.rtd_loss_weight = rtd_loss_weight

    def forward(self, input_ids, attention_mask, augmented_input_ids=None, augmented_attention_mask=None):
        """
        Forward pass for both original and augmented inputs.

        Args:
            input_ids (torch.Tensor): Original input IDs.
            attention_mask (torch.Tensor): Original attention mask.
            augmented_input_ids (torch.Tensor, optional): Augmented input IDs.
            augmented_attention_mask (torch.Tensor, optional): Augmented attention mask.

        Returns:
            tuple: (embeddings, rtd_logits, augmented_embeddings, augmented_rtd_logits)
        """
        # Encode original sentences
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        pooled_output = self.activation(self.pooler(pooled_output))  # Sentence Embeddings

        # RTD logits for original sentences
        rtd_logits = self.rtd_head(outputs.last_hidden_state)  # Shape: (batch_size, seq_len, vocab_size)

        if augmented_input_ids is not None and augmented_attention_mask is not None:
            # Encode augmented sentences
            augmented_outputs = self.encoder(input_ids=augmented_input_ids, attention_mask=augmented_attention_mask)
            augmented_pooled_output = self.dropout(augmented_outputs.pooler_output)
            augmented_pooled_output = self.activation(self.pooler(augmented_pooled_output))  # Augmented Embeddings

            # RTD logits for augmented sentences
            augmented_rtd_logits = self.rtd_head(augmented_outputs.last_hidden_state)
        else:
            augmented_pooled_output = None
            augmented_rtd_logits = None

        return pooled_output, rtd_logits, augmented_pooled_output, augmented_rtd_logits

# ==============================
# 5. Define Custom Trainer
# ==============================

from transformers import Trainer, EarlyStoppingCallback

class CustomTrainer(Trainer):
    def __init__(self, generator_model, generator_tokenizer, rtd_loss_weight=0.05, mask_prob=0.15, max_length=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_model = generator_model
        self.generator_tokenizer = generator_tokenizer
        self.rtd_loss_weight = rtd_loss_weight
        self.mask_prob = mask_prob
        self.max_length = max_length  

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Original inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size, seq_len = input_ids.size()

        # Decode original sentences
        sentences = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Generate augmented sentences and collect mask indices
        augmented_data = [
            augment_sentence(
                s, 
                self.generator_tokenizer, 
                self.generator_model, 
                mask_prob=self.mask_prob,
                max_length=self.max_length
            ) for s in sentences
        ]
        augmented_sentences, mask_indices_list = zip(*augmented_data)
        
        # Tokenize augmented sentences with same max_length
        augmented_encodings = self.tokenizer(
            augmented_sentences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(input_ids.device)
        
        augmented_input_ids = augmented_encodings['input_ids']
        augmented_attention_mask = augmented_encodings['attention_mask']
        
        # Forward pass through the model
        pooled_output, rtd_logits, augmented_pooled_output, augmented_rtd_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            augmented_input_ids=augmented_input_ids,
            augmented_attention_mask=augmented_attention_mask
        )
        
        # Create RTD labels
        # Original sentences: all tokens labeled as 1 (not replaced)
        rtd_labels_original = torch.ones(rtd_logits.size(0), rtd_logits.size(1)).to(rtd_logits.device)

        # Augmented sentences: tokens that were replaced are labeled as 0, others as 1
        rtd_labels_augmented = torch.ones(augmented_rtd_logits.size(0), augmented_rtd_logits.size(1)).to(augmented_rtd_logits.device)
        for i, mask_indices in enumerate(mask_indices_list):
            # Only set indices up to the sequence length
            valid_indices = mask_indices[mask_indices < self.max_length]
            rtd_labels_augmented[i][valid_indices] = 0.0


        # Concatenate RTD labels
        rtd_labels = torch.cat([rtd_labels_original, rtd_labels_augmented], dim=0)  # Shape: (batch_size*2, seq_len)

        # Concatenate RTD logits
        rtd_logits_combined = torch.cat([rtd_logits, augmented_rtd_logits], dim=0)  # Shape: (batch_size*2, seq_len, vocab_size)
        
        # Compute combined loss and individual components
        total_loss, loss_contrastive, loss_rtd = combined_loss_function(
            contrastive_emb=pooled_output,
            rtd_logits=rtd_logits_combined,
            rtd_labels=rtd_labels,
            rtd_weight=self.rtd_loss_weight
        )
        
        # Optionally, log individual losses
        if self.args.report_to == "wandb":
            wandb.log({
                "total_loss": total_loss.item(),
                "contrastive_loss": loss_contrastive.item(),
                "rtd_loss": loss_rtd.item()
            })

        return (total_loss, pooled_output) if return_outputs else total_loss

# ==============================
# 6. Main Training Function
# ==============================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'Models')

    # ------------------------------
    # 6.1. Configuration
    # ------------------------------
    dapt_config = {
        'batch_size': 64,
        'learning_rate': 3e-6,  
        'num_epochs': 10,
        'max_length': 512,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'weight_decay': 1e-2,
        'output_dir': os.path.join(models_dir, 'dapt_checkpoints'),
        'logging_dir': os.path.join(models_dir, 'logs'),
        'rtd_loss_weight': 0.05, 
        'mask_prob': 0.15,     
        'max_steps': 100000,
    }

    # ------------------------------
    # 6.2. Create Output Directory
    # ------------------------------
    os.makedirs(dapt_config['output_dir'], exist_ok=True)

    # ------------------------------
    # 6.3. Initialize Weights & Biases
    # ------------------------------
    wandb.init(
        project="financial-dapt",
        config=dapt_config,
        name=f"dapt_bs{dapt_config['batch_size']}_lr{dapt_config['learning_rate']}"
    )

    # ------------------------------
    # 6.4. Initialize Tokenizer and Models
    # ------------------------------
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Initialize the generator model as RoBERTaForMaskedLM
    generator_tokenizer = tokenizer  # Sharing the same tokenizer
    generator_model = RobertaForMaskedLM.from_pretrained('roberta-base')  # Updated model
    generator_model.eval()  # Set to evaluation mode
    generator_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    for param in generator_model.parameters():
        param.requires_grad = False
    
    # Initialize the main model
    model = DiffCSEWithRTDModel(
        model_name='roberta-base',
        rtd_loss_weight=dapt_config['rtd_loss_weight']
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # 6.5. Load and Prepare Dataset
    # ------------------------------
    logger.info("Loading dataset...")
    dataset_path = r"/Research/Koval Paper/Data/Output/Model Data/DAPT Data/dapt_sentences.csv"
    try:
        sentences_df = pd.read_csv(dataset_path)
        sentences = sentences_df['sentence'].tolist()
        logger.info(f"Loaded {len(sentences)} sentences.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {str(e)}")
        raise e

    # Split into training and validation sets
    train_sentences, val_sentences = train_test_split(sentences, test_size=0.1, random_state=42)
    logger.info(f"Training sentences: {len(train_sentences)}, Validation sentences: {len(val_sentences)}")

    # Create datasets
    train_dataset = DAPTDataset(
        sentences=train_sentences,
        tokenizer=tokenizer,
        max_length=dapt_config['max_length']
    )

    val_dataset = DAPTDataset(
        sentences=val_sentences,
        tokenizer=tokenizer,
        max_length=dapt_config['max_length']
    )

    # ------------------------------
    # 6.6. Define Training Arguments
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=dapt_config['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=dapt_config['num_epochs'],
        per_device_train_batch_size=dapt_config['batch_size'],
        learning_rate=dapt_config['learning_rate'],
        weight_decay=dapt_config['weight_decay'],
        warmup_steps=dapt_config['warmup_steps'],
        max_steps=100000,
        logging_steps=100,
        save_steps=5000,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=1000,
        fp16=True if torch.cuda.is_available() else False,
        logging_dir=dapt_config['logging_dir'],
        report_to="wandb",
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        max_grad_norm=dapt_config['max_grad_norm'],
        run_name=f"dapt_bs{dapt_config['batch_size']}_lr{dapt_config['learning_rate']}",
    )

    # ------------------------------
    # 6.7. Initialize Callbacks
    # ------------------------------
    save_best_callback = SaveBestModelCallback(model, tokenizer)

    # ------------------------------
    # 6.8. Initialize Trainer
    # ------------------------------
    trainer = CustomTrainer(
      generator_model=generator_model,
      generator_tokenizer=generator_tokenizer,
      rtd_loss_weight=dapt_config['rtd_loss_weight'],
      mask_prob=dapt_config['mask_prob'],  # Pass mask_prob here
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=tokenizer,
      callbacks=[save_best_callback, WandbCallback(), EarlyStoppingCallback(early_stopping_patience=3)]
    ) 

    # ------------------------------
    # 6.9. Start Training
    # ------------------------------
    try:
        logger.info("Starting DAPT training...")
        trainer.train()
        logger.info("DAPT training completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current model state...")
        trainer.save_model(os.path.join(dapt_config['output_dir'], 'interrupted_checkpoint'))
        tokenizer.save_pretrained(os.path.join(dapt_config['output_dir'], 'interrupted_checkpoint'))
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise e
    finally:
        # ------------------------------
        # 6.10. Save Final Model
        # ------------------------------
        logger.info("Saving the final model...")
        trainer.save_model(os.path.join(dapt_config['output_dir'], 'final_model'))
        tokenizer.save_pretrained(os.path.join(dapt_config['output_dir'], 'final_model'))

        # ------------------------------
        # 6.11. Finalize Weights & Biases
        # ------------------------------
        wandb.finish()

if __name__ == "__main__":
    main()
