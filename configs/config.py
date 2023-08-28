import argparse
import ast


parser = argparse.ArgumentParser(description="hyper-parameter for XRay-Gen")

# ====================== Dataset Config ===========================
parser.add_argument('--dataset', metavar='DATASET', default='./data/mimic_cxr/annotation.json', help='training datasets to use')
parser.add_argument('--base_dir', default='./data/mimic_cxr', type=str, help='Dataset directory containing image folders.')
parser.add_argument('--feature_path', default='./data/mimic_cxr/feature.pickle', type=str, help='feature of images')
parser.add_argument('--savedmodel_path', default='./save/debug', type=str, help='Dataset directory containing image folders.')
parser.add_argument('--ckpt_file', default=None, type=str, help='Dataset directory containing image folders.')
parser.add_argument('--delta_file', default=None, type=str, help='Dataset directory containing image folders.')

# ====================== Model Config ===========================
parser.add_argument('--llm_model', default='/apdcephfs/share_916081/vinnylywang/zhanyuwang/Data/Checkpoints/Llama-2-7b-chat-hf/', help='LLM to use, meta-llama/Llama-2-7b-chat-hf')
parser.add_argument('--freeze_llm', default=True, type=lambda x: (str(x).lower() == 'true'), help="freeze llm model or not")

# ====================== Training Config ===========================
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size for training')
parser.add_argument('--val_batch_size', default=4, type=int, help='mini-batch size for validation')
parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N', help='Number of batches loaded in advance by each worker')
parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--num_tokens', default=8, type=int, metavar='N', help='Number of [IMG] tokens to use.')
parser.add_argument('--projection_dim', default=768, type=int, metavar='N', help='dimension of chest xray feature')
parser.add_argument('--num_feature_tokens', default=1024, type=int, metavar='N', help='Number of chest xray feature')

# ====================== Pytorch Lightning ===========================
parser.add_argument('--cap_loss_scale', type=float, default=1.0, help="Scale on captioning loss.")
parser.add_argument('--gen_loss_scale', type=float, default=1.0, help="Scale on retrieval loss.")
parser.add_argument('--gen_emb_dim', default=768, type=int, metavar='N', help='Embedding dimension for generation.')

# ====================== Decoding ===========================
parser.add_argument('--max_length', default=100, type=int, metavar='N', help='Maximum length to truncate captions / generations to.')

# ====================== Pytorch Lightning ===========================
parser.add_argument('--devices', type=int, default=1, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
parser.add_argument('--strategy', type=str, default="ddp", help='default ddp for multi-gpus')
parser.add_argument('--precision', type=str, default='16', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
parser.add_argument('--max_steps', default=1500000, type=int, metavar='N', help='Stop training after this number of steps. ')
parser.add_argument('--max_epochs', type=int, default=30, help='Stop training once this number of epochs is reached')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates gradients over k batches before stepping the optimizer')
parser.add_argument('--log_every_n_steps', type=int, default=20, help='How often to log within steps')
parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')
