import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM
from utils.get_tokenizer import get_tokenizer
from utils.layers import TextFcLayer
from utils.optim import config_optimizer
import numpy as np


class XGenModel(pl.LightningModule):
    """
    GVLLModel.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # Loadding LLama2
        print('Loading LLAMA-2, may need a while')
        self.llm_tokenizer, self.gen_token_idx = get_tokenizer(args.llm_model, args.num_tokens)

        if self.args.accelerator == "gpu":
            self.llm_model = LlamaForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype=torch.float16
                )
        else:
            self.llm_model = LlamaForCausalLM.from_pretrained(
                args.llm_model
                )

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.input_embeddings = self.llm_model.get_input_embeddings()

        if args.freeze_llm:
            print('Freezing the LM')
            self.llm_model.eval()
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            self.llm_model.train()
        print('Loading LLAMA-2 Done')

        # GXLLMapper: mapping text to chest xray
        self.gvll_mapper = TextFcLayer(self.llm_model.config.hidden_size, self.args.gen_emb_dim, num_input_tokens=self.args.num_tokens, num_output_tokens=self.args.num_feature_tokens)

        self.val_step_outputs = []
        self.mse_loss = nn.MSELoss()
        self.val_score = 100.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location='cpu')['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    # For generation mode
    def generation(self, labels, caption_len, input_prefix=None, training=False):
        input_embs = self.input_embeddings(labels) # bs words_length, llm_emb_dim
        batch_size, _ = labels.shape 
        last_embedding_idx = caption_len - 1
        full_labels =  labels.masked_fill(labels == 0, -100)

        output = self.llm_model(inputs_embeds=input_embs,
                        labels=full_labels,
                        output_hidden_states=True)

        idx = -1
        num_tokens = self.hparams.num_tokens
        input_hidden_state = torch.stack([output.hidden_states[idx][i, last_embedding_idx[i]-num_tokens+1:last_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)
        input_embedding = torch.stack([input_embs[i, last_embedding_idx[i]-num_tokens+1:last_embedding_idx[i]+1, :] for i in range(batch_size)], axis=0)

        if training:
            self.gvll_mapper.to(torch.float32)
        else:
            self.gvll_mapper.to(torch.float16)
 
        last_embedding = self.gvll_mapper(input_hidden_state, input_embedding)  # (N, seq_len, 2048)
        return output, full_labels, last_embedding


    def get_last_embedding(self, prompt):
        out = self.llm_tokenizer(prompt, return_tensors="pt")
        input_ids = out.input_ids[0]
        input_ids = torch.cat([input_ids, torch.tensor(self.gen_token_idx)])
        input_embs = self.input_embeddings(input_ids)
        input_embs = input_embs.unsqueeze(0)
        output = self.llm_model(inputs_embeds=input_embs,
                        labels=input_ids.unsqueeze(0),
                        output_hidden_states=True)
        
        input_hidden_state = output.hidden_states[-1][:, -self.hparams.num_tokens:, :]
        input_embedding = input_embs[:, -self.hparams.num_tokens:, :]
        last_embedding = self.gvll_mapper(input_hidden_state, input_embedding)
        return last_embedding
    

    def training_step(self, batch, batch_idx):
        to_log = {}
        model_output, full_labels, last_embedding = self.generation(labels = batch['input_ids'],
                                                    caption_len = batch['len_report'],
                                                    training=True)                
        ce_loss = model_output.loss
        mse_loss = self.mse_loss(batch['image_emb'], last_embedding) * self.hparams.gen_loss_scale * 5
        to_log['mse_loss'] = mse_loss
        to_log['ce_loss'] = ce_loss

        loss = ce_loss + mse_loss
        to_log['loss'] = loss
        self.log_dict(to_log, prog_bar=True)
        return loss

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_val_loss{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, batch, batch_idx):
        to_log = {}
        model_output, full_labels, last_embedding = self.generation(labels = batch['input_ids'],
                                                    caption_len = batch['len_report'],
                                                    training=False)
        logits = model_output.logits
        ce_loss = model_output.loss
        to_log['val_ce_loss'] = ce_loss    
        mse_loss = self.mse_loss(batch['image_emb']/255, last_embedding) * self.hparams.gen_loss_scale
        to_log['val_mse_loss'] = mse_loss

        loss = ce_loss + mse_loss
        to_log['val_loss'] = loss

        self.log_dict(to_log, sync_dist=True, prog_bar=True)
        self.val_step_outputs.append({"val_loss": loss})
        return to_log
    
    def decode(self, output_token):
        output_text = self.llm_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.replace('<unk>', '').replace('<s>', '').replace('</s>', '').strip()
        return output_text

    def on_validation_epoch_end(self):
        val_loss = []
        for i in self.val_step_outputs:
            val_loss.append(i['val_loss'].item())

        val_loss = np.mean(val_loss)
        if self.trainer.local_rank == 0:
            if val_loss < self.val_score:
                self.save_checkpoint(val_loss)
                self.val_score = val_loss
                
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()