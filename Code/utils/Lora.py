from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
import torch

import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    LlamaConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

class Final_model(nn.Module):
    def __init__(self, model):
        super(Final_model, self).__init__()
        self.encoder = model.model if hasattr(model, "model") else model
        self.encoder.config.output_hidden_states = True
        

    def forward(self, inputs_embeds, attention_mask=None, training=True):
        if training:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        
        # Llama breaks if inputs_embeds are passed without position_ids matching the mask
        if attention_mask is None:
            attention_mask = (inputs_embeds.abs().sum(dim=-1) > 1e-4).long()
        
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        
        outputs = self.encoder(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            position_ids=position_ids, 
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        sequence_lengths = sequence_lengths.clamp(min=0)
        
        batch_size = last_hidden_state.shape[0]
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        
        pooled_output = last_hidden_state[batch_indices, sequence_lengths]
        
        return pooled_output 
    

def LLAMA(lora = False):
    ''' config = AutoConfig.from_pretrained(
                "/data/edu/PETCT/models/RBLora",
                num_labels=3,
                finetuning_task="mnli",
                cache_dir=None,
                revision="main",
                use_auth_token=None,
                cls_dropout=0.15,
                apply_lora=True,
                lora_alpha=16,
                lora_r=8,
                apply_adapter=False,
                reg_loss_wgt=0,
                masking_prob=15,
            )

    model =  AutoModelForSequenceClassification.from_pretrained(
                "/data/edu/PETCT/models/RBLora",
                from_tf=bool(".ckpt" in "/data/edu/PETCT/models/RBLora"),
                config=config,
                cache_dir=None,
                revision="main",
                use_auth_token= None,
            )'''
    if lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model =  AutoModelForSequenceClassification.from_pretrained(
            num_labels=768,
            offload_folder="offload",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=32, lora_alpha=32, lora_dropout=0.05, bias="none", 
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj" 
            ],
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)

        model.print_trainable_parameters()
    else:
        config = LlamaConfig(
            vocab_size=32000,       # Standard
            hidden_size=512,        # Reduced from 2048 (Easier to train)
            intermediate_size=1024, # Reduced from 5632
            num_hidden_layers=4,    # Reduced from 16/32 (Crucial for gradient flow)
            num_attention_heads=8,  # Reduced from 32
            num_key_value_heads=8,
            max_position_embeddings=2048
        )
        '''model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="/data/edu/PETCT/models/LL1Lora",
            num_labels=768,
            offload_folder="offload",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )'''
        model = AutoModelForSequenceClassification.from_config(config)
        
        
    model.config.pad_token_id = model.config.eos_token_id
    model = Final_model(model)

    return model


class Roberta(nn.Module):
    def __init__(self, model, lora=False):
        super().__init__()
        self.model = model
        # Using a learned CLS token is good, but let's ensure it's initialized well
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768) * 0.02) 

    def forward(self, inputs_embeds, attention_mask=None, training=True):
        if training:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        batch_size = inputs_embeds.shape[0]
        
        # 1. Prepend CLS token (T becomes T + 1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        inputs_embeds = torch.cat((cls_tokens, inputs_embeds), dim=1)

        # 2. Create the mask (Include the new CLS token)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, inputs_embeds.shape[1]), device=inputs_embeds.device)
        else:
            cls_mask = torch.ones((batch_size, 1), device=attention_mask.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        # 3. CRITICAL: The Position Offset
        # We start at 2 to avoid RoBERTa's internal padding/special indices
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(2, 2 + seq_len, device=inputs_embeds.device).expand(batch_size, -1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )
        
        # 4. Extract CLS representation
        # Standard RoBERTa uses the first token for classification
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state[:, 0, :] 
        
        return pooled_output


def DBR(lora = True, model_path="/data/edu/PETCT/models/DBLora"):
    if lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["classifier"]
        )
        model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=768,
            offload_folder="offload",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05, bias="none", 
            target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"],
            modules_to_save=["classifier"]
        )
        model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        model = Roberta(model, lora)
    else:
        model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=768,
            offload_folder="offload",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = Roberta(model, lora)
    

    return model


def Med(lora = False, model_path="/data/edu/PETCT/models/Medical"):
    if lora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.half,
            bnb_4bit_use_double_quant=True,
        )
        model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=2,
            offload_folder="offload",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.half,
            quantization_config=bnb_config
        )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none", 
            target_modules=None,
        )
        model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()
        model = Medical(model, lora)
    else:
        model =  AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_path,
            num_labels=2,
            offload_folder="offload",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.half
        )
        model = Medical(model, lora)
    

    return model


class Medical(nn.Module):
    def __init__(self, model, lora = False):
        super(Medical, self).__init__()
        self.model = model
        self.lora = lora
        if lora:
            self.model.base_model.model.score.modules_to_save.default = nn.Linear(in_features=768, out_features=768, bias = True)
        else:
            self.cls_head = model.classifier
            self.cls_head.train()
            #print(self.cls_head)
            self.model = model.roberta.encoder
            self.model.train()
            #self.model.classifier.original_module = nn.Linear(in_features=768, out_features=768, bias = True)
            
    def freeze_bottom(self):
        for n in range(0,9):
            for param in self.model.layer[n].parameters():
                param.requires_grad_(False)


    def forward(self, inputs_embeds, attention_mask= None, training = True):
        if self.lora:
            out = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
            out = out.logits
        else:
            out = self.model(inputs_embeds).last_hidden_state
            out = self.cls_head(out)
        return out