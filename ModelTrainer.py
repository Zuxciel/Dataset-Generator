import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
import argparse
from tqdm import tqdm
import wandb
import random
from datetime import datetime
import glob

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ForthnlesConfig:
    """Configuration for Forthnles model training"""
    def __init__(self):
        # Model configuration
        self.model_name = "Forthnles"
        self.base_model = "gpt2"  # Can also use "facebook/blenderbot-400M-distill", "microsoft/DialoGPT-medium", etc.
        self.tokenizer_max_length = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Training configuration
        self.train_batch_size = 4
        self.eval_batch_size = 4
        self.gradient_accumulation_steps = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.num_train_epochs = 3
        self.warmup_steps = 500
        self.logging_steps = 100
        self.eval_steps = 500
        self.save_steps = 1000
        
        # Data configuration
        self.data_dir = "chat_datasets"
        self.output_dir = "forthnles_model"
        
        # Resume training
        self.resume_training = False
        
        # Mixed precision training
        self.fp16 = torch.cuda.is_available()
        
        # Logging configuration
        self.use_wandb = False
        self.wandb_project = "forthnles"


class ConversationDataset(Dataset):
    """Dataset for conversation examples"""
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        messages = conversation['messages']
        
        # Format the conversation as a single text
        formatted_text = ""
        for message in messages:
            role = message['role']
            content = message['content']
            formatted_text += f"{role}: {content}\n"
        
        # Tokenize the text
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare for language modeling
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class IntentDataset(Dataset):
    """Dataset for intent classification"""
    def __init__(self, df, tokenizer, max_length=128, label_map=None):
        self.utterances = df['utterance'].tolist()
        self.intents = df['intent'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label map if not provided
        if label_map is None:
            unique_intents = sorted(df['intent'].unique())
            self.label_map = {intent: i for i, intent in enumerate(unique_intents)}
        else:
            self.label_map = label_map
    
    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        utterance = self.utterances[idx]
        intent = self.intents[idx]
        label = self.label_map[intent]
        
        encoding = self.tokenizer(
            utterance,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ForthnlesTrainer:
    """Trainer for the Forthnles model"""
    def __init__(self, config=None):
        self.config = config if config else ForthnlesConfig()
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_tokenizer(self):
        """Initialize the tokenizer"""
        # First check if there's a saved tokenizer in the output directory
        conv_tokenizer_path = os.path.join(self.config.output_dir, "conversation", "final_model")
        
        if os.path.exists(conv_tokenizer_path) and self.config.resume_training:
            logger.info(f"Loading existing tokenizer from {conv_tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(conv_tokenizer_path)
        else:
            logger.info(f"Loading tokenizer based on {self.config.base_model}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Add special tokens for conversation
            special_tokens = {
                'additional_special_tokens': ['user:', 'assistant:']
            }
            tokenizer.add_special_tokens(special_tokens)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # Also add pad_token_id to the tokenizer for consistent batch processing
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def find_latest_checkpoint(self, model_type):
        """Find the latest checkpoint for a given model type"""
        checkpoint_dir = os.path.join(self.config.output_dir, model_type)
        if not os.path.exists(checkpoint_dir):
            return None
            
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        if "final_model" in os.listdir(checkpoint_dir):
            checkpoint_dirs.append(os.path.join(checkpoint_dir, "final_model"))
            
        if not checkpoint_dirs:
            return None
            
        # Sort by checkpoint number to find the latest one
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]) if "-" in x else float('inf'))
        latest_checkpoint = checkpoint_dirs[-1]
        
        return latest_checkpoint
    
    def setup_conversation_model(self):
        """Initialize the conversation model"""
        # Check for existing model to resume from
        latest_checkpoint = None
        if self.config.resume_training:
            latest_checkpoint = self.find_latest_checkpoint("conversation")
        
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            logger.info(f"Resuming conversation model from checkpoint: {latest_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
        else:
            logger.info(f"Setting up new conversation model based on {self.config.base_model}")
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
            
            # Resize token embeddings for added special tokens
            model.resize_token_embeddings(len(self.tokenizer))
        
        # Make sure the model knows about the padding token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = model
        return model
    
    def setup_intent_model(self, num_labels):
        """Initialize the intent classification model"""
        # Check for existing model to resume from
        latest_checkpoint = None
        if self.config.resume_training:
            latest_checkpoint = self.find_latest_checkpoint("intent")
        
        if latest_checkpoint and os.path.exists(latest_checkpoint):
            logger.info(f"Resuming intent model from checkpoint: {latest_checkpoint}")
            model = AutoModelForSequenceClassification.from_pretrained(latest_checkpoint)
        else:
            logger.info(f"Setting up new intent classification model based on {self.config.base_model}")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.base_model,
                num_labels=num_labels
            )
            
            # Resize token embeddings for added special tokens
            model.resize_token_embeddings(len(self.tokenizer))
        
        # Make sure the model knows about the padding token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model = model
        return model
    
    def load_conversation_data(self):
        """Load conversation datasets"""
        train_file = os.path.join(self.config.data_dir, "conversation_train.json")
        eval_file = os.path.join(self.config.data_dir, "conversation_valid.json")
        
        if not os.path.exists(train_file) or not os.path.exists(eval_file):
            logger.error(f"Training or validation data not found: {train_file} or {eval_file}")
            return None, None
        
        with open(train_file, 'r') as f:
            train_conversations = json.load(f)
        
        with open(eval_file, 'r') as f:
            eval_conversations = json.load(f)
        
        logger.info(f"Loaded {len(train_conversations)} training conversations and {len(eval_conversations)} validation conversations")
        
        train_dataset = ConversationDataset(train_conversations, self.tokenizer, self.config.tokenizer_max_length)
        eval_dataset = ConversationDataset(eval_conversations, self.tokenizer, self.config.tokenizer_max_length)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        return train_dataset, eval_dataset
    
    def load_intent_data(self):
        """Load intent classification datasets"""
        train_file = os.path.join(self.config.data_dir, "intent_train.csv")
        eval_file = os.path.join(self.config.data_dir, "intent_valid.csv")
        
        if not os.path.exists(train_file) or not os.path.exists(eval_file):
            logger.error(f"Training or validation data not found: {train_file} or {eval_file}")
            return None, None
        
        train_df = pd.read_csv(train_file)
        eval_df = pd.read_csv(eval_file)
        
        # Check if we have a saved label map from previous training
        label_map_path = os.path.join(self.config.output_dir, "intent", "final_model", "label_map.json")
        if os.path.exists(label_map_path) and self.config.resume_training:
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            logger.info(f"Loaded existing label map with {len(label_map)} intents")
        else:
            # Create label map from training data
            unique_intents = sorted(train_df['intent'].unique())
            label_map = {intent: i for i, intent in enumerate(unique_intents)}
            logger.info(f"Created new label map with {len(label_map)} intents")
        
        logger.info(f"Loaded {len(train_df)} training examples and {len(eval_df)} validation examples")
        
        train_dataset = IntentDataset(train_df, self.tokenizer, label_map=label_map)
        eval_dataset = IntentDataset(eval_df, self.tokenizer, label_map=label_map)
        
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.label_map = label_map
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, pred):
        """Compute metrics for evaluation"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_conversation_model(self):
        """Train the conversation model"""
        if self.train_dataset is None or self.eval_dataset is None:
            logger.error("Dataset not loaded. Call load_conversation_data() first.")
            return
        
        logger.info("Training conversation model...")
        
        # Find latest checkpoint to resume from
        resume_from_checkpoint = False
        if self.config.resume_training:
            latest_checkpoint = self.find_latest_checkpoint("conversation")
            if latest_checkpoint:
                resume_from_checkpoint = latest_checkpoint
                logger.info(f"Will resume training from checkpoint: {resume_from_checkpoint}")
        
        # Setup training arguments - Fix: Change evaluation_strategy to eval_strategy
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, "conversation"),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",  # Fixed: Changed from evaluation_strategy to eval_strategy
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16,
            report_to="wandb" if self.config.use_wandb else "none"
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal language modeling, not masked
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        # Start or resume training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        self.model.save_pretrained(os.path.join(self.config.output_dir, "conversation", "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.config.output_dir, "conversation", "final_model"))
        
        logger.info("Conversation model training completed!")
    
    def train_intent_model(self):
        """Train the intent classification model"""
        if self.train_dataset is None or self.eval_dataset is None:
            logger.error("Dataset not loaded. Call load_intent_data() first.")
            return
        
        logger.info("Training intent classification model...")
        
        # Find latest checkpoint to resume from
        resume_from_checkpoint = False
        if self.config.resume_training:
            latest_checkpoint = self.find_latest_checkpoint("intent")
            if latest_checkpoint:
                resume_from_checkpoint = latest_checkpoint
                logger.info(f"Will resume training from checkpoint: {resume_from_checkpoint}")
        
        # Set batch size to 1 if we know we'll have padding token issues
        # or fix the real issue by ensuring the padding token is properly set
        train_batch_size = 1 if self.config.train_batch_size > 1 and self.model.config.pad_token_id is None else self.config.train_batch_size
        eval_batch_size = 1 if self.config.eval_batch_size > 1 and self.model.config.pad_token_id is None else self.config.eval_batch_size
        
        # Setup training arguments - Fix: Change evaluation_strategy to eval_strategy
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, "intent"),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",  # Fixed: Changed from evaluation_strategy to eval_strategy
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=self.config.fp16,
            report_to="wandb" if self.config.use_wandb else "none"
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Start or resume training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        self.model.save_pretrained(os.path.join(self.config.output_dir, "intent", "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.config.output_dir, "intent", "final_model"))
        
        # Save label map
        with open(os.path.join(self.config.output_dir, "intent", "final_model", "label_map.json"), 'w') as f:
            json.dump(self.label_map, f)
        
        logger.info("Intent classification model training completed!")
    
    def train_forthnles(self):
        """Train the complete Forthnles model"""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize wandb if configured
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project, name=f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        # Train conversation model
        logger.info("Starting conversation model training...")
        self.setup_conversation_model()
        self.load_conversation_data()
        self.train_conversation_model()
        
        # Train intent model
        logger.info("Starting intent classification model training...")
        self.load_intent_data()
        num_labels = len(self.train_dataset.label_map)
        self.setup_intent_model(num_labels)
        self.train_intent_model()
        
        logger.info(f"Training completed! Models saved to {self.config.output_dir}")


class ForthnlesChat:
    """Interactive chat interface for Forthnles model"""
    def __init__(self, model_dir="forthnles_model"):
        self.model_dir = model_dir
        self.conversation_model = None
        self.intent_model = None
        self.tokenizer = None
        self.label_map = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
    
    def load_models(self):
        """Load trained models"""
        # Load conversation model
        conv_model_path = os.path.join(self.model_dir, "conversation", "final_model")
        if os.path.exists(conv_model_path):
            logger.info(f"Loading conversation model from {conv_model_path}")
            self.conversation_model = AutoModelForCausalLM.from_pretrained(conv_model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(conv_model_path)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            # Ensure the model knows about the padding token
            if self.conversation_model.config.pad_token_id is None:
                self.conversation_model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            logger.warning(f"Conversation model not found at {conv_model_path}")
        
        # Load intent model
        intent_model_path = os.path.join(self.model_dir, "intent", "final_model")
        if os.path.exists(intent_model_path):
            logger.info(f"Loading intent model from {intent_model_path}")
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path).to(self.device)
            
            # Ensure the intent model knows about the padding token
            if self.intent_model.config.pad_token_id is None and self.tokenizer is not None:
                self.intent_model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Load label map
            label_map_path = os.path.join(intent_model_path, "label_map.json")
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                # Invert the map for prediction
                self.inv_label_map = {v: k for k, v in self.label_map.items()}
            else:
                logger.warning(f"Label map not found at {label_map_path}")
        else:
            logger.warning(f"Intent model not found at {intent_model_path}")
        
        return self.conversation_model is not None and self.tokenizer is not None
    
    def predict_intent(self, text):
        """Predict intent for a given text"""
        if not self.intent_model or not self.tokenizer or not self.inv_label_map:
            logger.error("Intent model, tokenizer, or label map not loaded")
            return None
        
        # Tokenize the input
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            intent_id = predictions.item()
        
        # Get intent label
        intent = self.inv_label_map.get(intent_id, "unknown")
        return intent
    
    def generate_response(self, conversation_history):
        """Generate a response based on conversation history"""
        if not self.conversation_model or not self.tokenizer:
            logger.error("Conversation model or tokenizer not loaded")
            return "Model not loaded properly. Please check logs."
        
        # Format the conversation
        formatted_text = ""
        for message in conversation_history:
            role = message['role']
            content = message['content']
            formatted_text += f"{role}: {content}\n"
        
        formatted_text += "assistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_sequences = self.conversation_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode the generated response
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Extract only the assistant's response from the generated text
        response_part = generated_text[len(formatted_text):].strip()
        
        # If the model generated a role prefix like "user:", cut it off
        if "user:" in response_part:
            response_part = response_part.split("user:")[0].strip()
        
        return response_part
    
    def chat(self):
        """Interactive chat session with Forthnles"""
        if not self.load_models():
            logger.error("Failed to load models")
            return
        
        print("\n=== Forthnles Chat ===")
        print("Type 'exit' to end the conversation")
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Forthnles: Goodbye!")
                break
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Detect intent if intent model is available
            if self.intent_model:
                intent = self.predict_intent(user_input)
                print(f"[Detected intent: {intent}]")
            
            # Generate response
            response = self.generate_response(conversation_history)
            
            # Add assistant message to history
            conversation_history.append({"role": "assistant", "content": response})
            
            print(f"Forthnles: {response}")
            
            # Limit conversation history to last 10 messages
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]


def main():
    parser = argparse.ArgumentParser(description="Train and use Forthnles conversational AI model")
    parser.add_argument("--action", type=str, choices=["train", "chat"], default="train", 
                        help="Action to perform: train or chat")
    parser.add_argument("--data_dir", type=str, default="chat_datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--output_dir", type=str, default="forthnles_model",
                        help="Directory to save the trained model")
    parser.add_argument("--base_model", type=str, default="gpt2",
                        help="Base model to use for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    
    args = parser.parse_args()
    
    if args.action == "train":
        # Setup configuration
        config = ForthnlesConfig()
        config.data_dir = args.data_dir
        config.output_dir = args.output_dir
        config.base_model = args.base_model
        config.num_train_epochs = args.epochs
        config.train_batch_size = args.batch_size
        config.eval_batch_size = args.batch_size
        config.use_wandb = args.use_wandb
        config.resume_training = args.resume
        
        # Create and train model
        trainer = ForthnlesTrainer(config)
        trainer.train_forthnles()
    
    elif args.action == "chat":
        # Setup chat interface
        chat = ForthnlesChat(model_dir=args.output_dir)
        chat.chat()


if __name__ == "__main__":
    main()
