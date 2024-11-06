import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import wandb
from transformers.integrations import WandbCallback

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
    Dataset class for Domain-Adaptive Pre-Training (DAPT) using Masked Language Modeling (MLM).
    """
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize the sentence without clean_up_tokenization_spaces
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
            # Removed: clean_up_tokenization_spaces=True
        )
        input_ids = encoding['input_ids'].squeeze(0)          # Shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: (max_length)

        # Create MLM labels (clone input_ids)
        labels = input_ids.clone()

        # Masking: 15% of tokens
        probability_matrix = torch.full(labels.shape, 0.15)
        # Do not mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        # Set labels for non-masked tokens to -100 to ignore them in loss computation
        labels[~masked_indices] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# ==============================
# 3. Define Custom Callback
# ==============================

class SaveBestModelCallback(TrainerCallback):
    """
    Custom callback to save the best model based on training loss.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.best_loss = float('inf')
        self.model = model
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Triggered at each logging step.
        """
        if logs is not None:
            current_loss = logs.get('loss')
            if current_loss and current_loss < self.best_loss:
                self.best_loss = current_loss
                logger.info(f"New best loss: {self.best_loss:.4f}. Saving best model.")
                # Save model and tokenizer
                self.model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
                self.tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))

# ==============================
# 4. Main Training Function
# ==============================

def main():
    # ------------------------------
    # 4.1. Configuration
    # ------------------------------
    dapt_config = {
        'batch_size': 32,
        'learning_rate': 5e-5,
        'num_epochs': 10,
        'max_length': 512,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'max_steps': 25000,
        'weight_decay': 1e-2,
        'output_dir': 'dapt_checkpoints',
        'logging_dir': 'logs'
    }

    # ------------------------------
    # 4.2. Create Output Directory
    # ------------------------------
    os.makedirs(dapt_config['output_dir'], exist_ok=True)

    # ------------------------------
    # 4.3. Initialize Weights & Biases
    # ------------------------------
    wandb.init(
        project="financial-dapt",
        config=dapt_config,
        name=f"dapt_bs{dapt_config['batch_size']}_lr{dapt_config['learning_rate']}"
    )

    # ------------------------------
    # 4.4. Initialize Tokenizer and Model
    # ------------------------------
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    # ------------------------------
    # 4.5. Load and Prepare Dataset
    # ------------------------------
    logger.info("Loading dataset...")
    dataset_path = "/content/drive/MyDrive/Colab Notebooks/Research/Paper Implementations/Koval Paper/Data/dapt_sentences.csv"
    try:
        sentences_df = pd.read_csv(dataset_path)
        sentences = sentences_df['sentence'].tolist()
        logger.info(f"Loaded {len(sentences)} sentences.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {str(e)}")
        raise e

    dataset = DAPTDataset(
        sentences=sentences,
        tokenizer=tokenizer,
        max_length=dapt_config['max_length']
    )

    # ------------------------------
    # 4.6. Define Training Arguments
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=dapt_config['output_dir'],
        overwrite_output_dir=True,
        num_train_epochs=dapt_config['num_epochs'],
        per_device_train_batch_size=dapt_config['batch_size'],
        learning_rate=dapt_config['learning_rate'],
        weight_decay=dapt_config['weight_decay'],
        warmup_steps=dapt_config['warmup_steps'],
        max_steps=dapt_config['max_steps'],
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="no",
        fp16=True,
        logging_dir=dapt_config['logging_dir'],
        report_to="wandb",
        gradient_accumulation_steps=1,
        load_best_model_at_end=False,
        dataloader_num_workers=4,
        max_grad_norm=dapt_config['max_grad_norm'],
        run_name=f"dapt_bs{dapt_config['batch_size']}_lr{dapt_config['learning_rate']}",
    )

    # ------------------------------
    # 4.7. Initialize Callbacks
    # ------------------------------
    save_best_callback = SaveBestModelCallback(model, tokenizer)

    # ------------------------------
    # 4.8. Initialize Trainer
    # ------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[save_best_callback]  # Removed WandbCallback to avoid duplication
    )

    # ------------------------------
    # 4.9. Start Training
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
        # 4.10. Save Final Model
        # ------------------------------
        logger.info("Saving the final model...")
        trainer.save_model(os.path.join(dapt_config['output_dir'], 'final_model'))
        tokenizer.save_pretrained(os.path.join(dapt_config['output_dir'], 'final_model'))

        # ------------------------------
        # 4.11. Finalize Weights & Biases
        # ------------------------------
        wandb.finish()

if __name__ == "__main__":
    main()
