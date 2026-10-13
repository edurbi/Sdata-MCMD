import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import monai as moi
from models.Transformer import ViT3D
from utils.Float_embedding import FloatEmbed
from utils.Lora import LLAMA as DBLora
from utils.Lora import DBR as DBR
from utils.Lora import Med as Med
from utils.Transformer import Transformer
import os
import time

basic_dims = 2
mlp_dim = 4098
num_heads = 8
patch_size = 8


import torch
import torch.nn as nn

class UnfoldingAdapter(nn.Module):
    def __init__(self, chunk_size=256, llm_dim=768):
        super().__init__()
        self.chunk_size = chunk_size
        
        self.chunk_proj = nn.Sequential(
            nn.Linear(chunk_size, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [Batch, Input_Dim] (Mixed real data and zero padding)
        """
        B, D = x.shape

        input_mask = (x.abs() > 1e-4).long() 

        num_chunks = D // self.chunk_size

        chunks = x.view(B, num_chunks, self.chunk_size)
        
        mask_chunks = input_mask.view(B, num_chunks, self.chunk_size)

        projected_tokens = self.chunk_proj(chunks) 
        
        token_mask = (mask_chunks.sum(dim=-1) > 0).long() 

        return projected_tokens, token_mask

    
class Feed_forward(nn.Module):
    def __init__(self,in_channels, out_classes):
        super(Feed_forward,self).__init__()
        self.ff1 = nn.Linear(in_channels, in_channels//16)
        self.ff2 = nn.Linear(in_channels//16, 64)
        self.dropout = nn.Dropout(0.1)
        self.ff3 = nn.Linear(64, out_classes)
        
    def forward(self, vector):
        x = torch.relu(self.ff1(vector))
        x = torch.relu(self.ff2(x))
        return self.ff3(x)
    

class Identitification(nn.Module):
    def forward(self,input):
        return input
    
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size,hidden_dropout_prob=0.1,num_labels=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size*4)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size*4, num_labels)
        self.mid_norm = nn.LayerNorm(hidden_size * 4)
        self.apply(self._init_weights)
        self.act = nn.GELU()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, hidden_states):
        hidden_states = self.input_norm(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        #hidden_states = self.dropout(hidden_states)
        hidden_states = self.mid_norm(hidden_states)
        output = self.out_proj(hidden_states)
        return output 
    
    
class VisionProjection(nn.Module):
    def __init__(self, vision_dim, llm_dim, hidden_dim=4096):
        super().__init__()
        # W1: Projects features to a larger hidden dimension
        self.linear1 = nn.Linear(vision_dim, hidden_dim, bias=True)
        # Non-linearity is crucial for expressive power
        self.act = nn.GELU() 
        # W2: Projects from hidden dimension to LLM embedding dimension
        self.linear2 = nn.Linear(hidden_dim, llm_dim, bias=True)

    def forward(self, features):
        features = self.linear1(features)
        features = self.act(features)
        features = self.linear2(features)
        return features
    
class Lformer_class(nn.Module):
    def __init__(self, path, meta):
        super(Lformer_class, self).__init__()
        if path==None:
            configuration = LongformerConfig(max_position_embeddings=4098, 
                                         attention_window =  [512,512,512,512,512,512,512,512,512,512,512,512],
                                         model_type = "longformer", attention_mode = "longformer"
                                         )
            self.bert = LongformerForSequenceClassification(configuration)
        else:
            self.bert = LongformerForSequenceClassification.from_pretrained(path, device_map = "cpu")
        self.bert = self.bert.longformer.encoder
        self.meta = meta
        self.p_len = (512 - self.meta)
        
    def forward(self,inputs_embeds, attention_mask,global_attention_mask= None):
        if global_attention_mask:
            attention_mask = attention_mask * (global_attention_mask + 1)
        encoder_output = self.bert(inputs_embeds,attention_mask = attention_mask, head_mask=None,
                                        output_attentions = True, padding_len = self.p_len,
                                        output_hidden_states = False, return_dict=False)
        logits = encoder_output[0][:,0,:]
        return logits
    
    
    

class Model(nn.Module):
    def __init__(self, meta = 0, mode = "LForm", max_lab=2, task_types = "", passed = False, test=False, model_path="/data/edu/PETCT/models/Longformer"):
        super(Model, self).__init__()
        self.in_mod=1
        self.meta = int(meta)
        self.tasktypes = task_types
        self.passed = passed
        self.alpha = 0.1
        self.feature_proj = nn.Linear(256+self.meta, 256+self.meta, dtype=torch.bfloat16)
        self.models = nn.ModuleDict({"2":ViT(img_size= (256,256), num_classes = 256, num_heads= 6, depth = 6, in_chans= 1, use_flash_attn=False, fused_bias_fc=False, fused_mlp=False,fused_dropout_add_ln=not test, dim = 2).to(torch.bfloat16),
                                     "3":ViT(img_size= (128,128, 128), num_classes = 256, num_heads= 6, depth = 6, in_chans= 1, use_flash_attn=False, fused_bias_fc=False, fused_mlp=False,fused_dropout_add_ln=not test, dim = 3).to(torch.bfloat16)})
        self.mode = mode
        self.max_lab = max_lab
        self.embeddings = FloatEmbed(768).to(torch.bfloat16)
        self.trainer = nn.Sequential(
            nn.Linear(257, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, self.max_lab) 
        )
        if self.meta > 0:
            # Assuming your Llama hidden_size is 512 based on your previous config
            llama_hidden_size = 512 
            self.meta_proj = nn.Sequential(
                nn.Linear(self.meta, llama_hidden_size),
                nn.GELU(),
                nn.LayerNorm(llama_hidden_size)
            ).to(torch.bfloat16)
        self.normalize = nn.LayerNorm(256)
        self.normalize3 = nn.LayerNorm(256)
        self.missing_token = nn.Parameter(torch.randn(1, 256))
        self.mod_type_embed = nn.Embedding(10, 256)
        nn.init.normal_(self.missing_token, std=0.02)
        if mode == "Roberta":
            self.bert= DBR()
            if passed:
                self.ff = ClassificationHead(self.bert.model.config.hidden_size+self.meta, num_labels=max_lab)
            else:
                self.ff = ClassificationHead(self.bert.model.config.hidden_size+self.meta, num_labels=max_lab)
            self.adapter = UnfoldingAdapter(chunk_size=256, llm_dim=self.bert.model.config.hidden_size).to(torch.bfloat16)
            self.adapter3 = UnfoldingAdapter(chunk_size=256, llm_dim=self.bert.model.config.hidden_size).to(torch.bfloat16)
        elif mode == "Med":
            self.bert= Med()
            if passed:
                self.ff = ClassificationHead(768+256+self.meta, num_labels=max_lab)
            else:
                self.ff = ClassificationHead(768+self.meta, num_labels=max_lab)
        elif mode == "PreLF":
            self.bert = Lformer_class(model_path,self.meta)
            if passed:
                self.ff = ClassificationHead(768+256+self.meta, num_labels=max_lab)
            else:
                self.ff = ClassificationHead(768+self.meta, num_labels=max_lab)
        elif mode == "Trans":
            self.bert = moi.networks.blocks.TransformerBlock(768,768,16,0.1)
            if passed:
                self.ff = ClassificationHead(768+256+self.meta, num_labels=max_lab)
            else:
                self.ff = ClassificationHead(768+self.meta, num_labels=max_lab)   
            self.adapter = UnfoldingAdapter(chunk_size=256, llm_dim=768).to(torch.bfloat16) 
        elif mode == "DBLora":
            #Llama-2-7b-longlora-8k
            if passed:
                self.ff = ClassificationHead(512, num_labels=max_lab)
            else:
                self.ff = ClassificationHead(512, num_labels=max_lab)
            self.bert = DBLora()
            self.adapter = UnfoldingAdapter(chunk_size=256, llm_dim=self.bert.encoder.config.hidden_size).to(torch.bfloat16)
        elif mode == "MLP":
            self.embeddings = Identitification()
            self.ff = ClassificationHead(2048,num_labels=max_lab)
        
        #self.bert.register_backward_hook(self.bckwrd_hook)

    def mismatch(self, vector,siz=256):
        batches, length = vector.size()
        rem = length % siz
        size_mismatch = 0 if rem == 0 else siz - rem
        
        if size_mismatch > 0:  
            vector = F.pad(input=vector, pad=(0, size_mismatch, 0, 0), mode='constant', value=0)
            
        one = torch.ones([batches, size_mismatch])
        aux = torch.cat([(torch.rand(batches, length) < 0.15).int(), one], dim=1)
        aux = aux.to(vector.device).to(torch.int16)
        return vector, aux
    
    def move_zeros_to_end(self, tensor_list):
        
        
        stacked = torch.stack(tensor_list, dim=1) 
    
        is_zero_block = (stacked.abs().sum(dim=2) == 0)
        
        indices = torch.argsort(is_zero_block.to(torch.int), dim=1)
        
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, 256)
        reordered = torch.gather(stacked, 1, expanded_indices)
        

        new_is_zero = (reordered.abs().sum(dim=2) == 0)
        
        column_is_all_zero = new_is_zero.all(dim=0) 
        
        active_columns = reordered[:, ~column_is_all_zero, :]
        
        final_output = active_columns.reshape(active_columns.size(0), -1)
        
        
        return final_output
    
    def forward(self, images, meta, types, val=False):
        meta = meta.to(torch.bfloat16)
        batch_size = len(types)
        device = meta.device
        
        # Parse which modalities go where
        dd = [[int(idx) for idx, letter in enumerate(t) if letter == "2"] for t in types]
        td = [[int(idx) for idx, letter in enumerate(t) if letter == "3"] for t in types]
        dd_max = max([len(q) for q in dd]) if dd else 0
        td_max = max([len(q) for q in td]) if td else 0
        
        # --- PHASE 1: GATHER IMAGES FOR PARALLEL PROCESSING ---
        flat_2d_tensors, flat_2d_indices, flat_2d_mod_idx = [], [], []
        flat_3d_tensors, flat_3d_indices, flat_3d_mod_idx = [], [], []
        
        for i in range(dd_max):
            accepted = [b_idx for b_idx, value in enumerate(dd) if len(value) > i]
            if accepted:
                flat_2d_tensors.append(images["Mod"+str(i)].to(torch.bfloat16))
                flat_2d_indices.extend(accepted)
                flat_2d_mod_idx.extend([i] * len(accepted))
                
        for i in range(dd_max, dd_max + td_max):
            accepted = [b_idx for b_idx, value in enumerate(td) if len(value) > (i - dd_max)]
            if accepted:
                flat_3d_tensors.append(images["Mod"+str(i)].to(torch.bfloat16))
                flat_3d_indices.extend(accepted)
                flat_3d_mod_idx.extend([i] * len(accepted))

        # --- PHASE 2: BATCHED FORWARD PASSES ---
        all_features = {b_idx: [] for b_idx in range(batch_size)}
        all_helps = {b_idx: [] for b_idx in range(batch_size)} 
        
        if flat_2d_tensors:
            cat_2d = torch.cat(flat_2d_tensors, dim=0)
            out_2d = self.models["2"](cat_2d)[0]
            
            if not val:
                current_meta = meta[flat_2d_indices, -1].unsqueeze(1)
                temporal_help = self.trainer(torch.cat((out_2d, current_meta), dim=1))
                # Group helper outputs by patient ID
                for idx, b_idx in enumerate(flat_2d_indices):
                    all_helps[b_idx].append(temporal_help[idx])

            mod_embeds = self.mod_type_embed(torch.tensor(flat_2d_mod_idx, device=device))
            out_2d = self.normalize(out_2d + mod_embeds)
            
            for idx, b_idx in enumerate(flat_2d_indices):
                all_features[b_idx].append(out_2d[idx])
                
        if flat_3d_tensors:
            cat_3d = torch.cat(flat_3d_tensors, dim=0)
            out_3d = self.models["3"](cat_3d)[0]
            
            if not val:
                current_meta = meta[flat_3d_indices, -1].unsqueeze(1)
                temporal_help = self.trainer(torch.cat((out_3d, current_meta), dim=1))
                # Group helper outputs by patient ID
                for idx, b_idx in enumerate(flat_3d_indices):
                    all_helps[b_idx].append(temporal_help[idx])

            mod_embeds = self.mod_type_embed(torch.tensor(flat_3d_mod_idx, device=device))
            out_3d = self.normalize3(out_3d + mod_embeds)
            
            for idx, b_idx in enumerate(flat_3d_indices):
                all_features[b_idx].append(out_3d[idx])
                
        # --- PHASE 3: SEQUENCE AND HELPER RECONSTRUCTION ---
        max_mods = dd_max + td_max
        b_flat = []
        
        for b_idx in range(batch_size):
            feats = all_features[b_idx]
            num_missing = max_mods - len(feats)
            if num_missing > 0:
                padding = [torch.zeros(256, device=device, dtype=torch.bfloat16) for _ in range(num_missing)]
                feats.extend(padding)
            b_flat.append(torch.cat(feats, dim=0))
            
        b = torch.stack(b_flat, dim=0) 
        
        b_list_for_sorting = [b[:, k*256:(k+1)*256] for k in range(max_mods)]
        if len(b_list_for_sorting) > 0:
            b = self.move_zeros_to_end(b_list_for_sorting)

        # Aggregation: Max pool valid modalities per patient
        if not val:
            helper_list = []
            for b_idx in range(batch_size):
                patient_helps = all_helps[b_idx]
                if len(patient_helps) > 0:
                    stacked = torch.stack(patient_helps, dim=0) 
                    normed = F.layer_norm(stacked, stacked.shape[1:])
                    max_help, _ = torch.max(normed, dim=0)
                    helper_list.append(max_help)
                else:
                    helper_list.append(torch.zeros(self.max_lab, device=device, dtype=torch.bfloat16))
            helper = torch.stack(helper_list, dim=0)
        else:
            helper = torch.zeros((batch_size, self.max_lab), device=device, dtype=torch.bfloat16)

        # --- PHASE 4: ADAPTER AND METADATA INJECTION ---
        b, _ = self.mismatch(b, 256)
        
        if self.mode in ["PreLF", "DBLora"]:
            image_tokens, image_mask = self.adapter(b) 
            
            if self.meta > 0:
                meta_tokens = self.meta_proj(meta).unsqueeze(1) 
                meta_mask = torch.ones((batch_size, 1), device=device, dtype=torch.long)
                
                inputs_embeds = torch.cat([meta_tokens, image_tokens], dim=1)
                attention_mask = torch.cat([meta_mask, image_mask], dim=1)
            else:
                inputs_embeds = image_tokens
                attention_mask = image_mask
                
            output = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            
        else:
            output = self.bert(inputs_embeds=b, attention_mask=None, training=not val)
            
        output = self.ff(output)
        
        return (output, b, helper)

 