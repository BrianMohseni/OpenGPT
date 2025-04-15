from accelerate import Accelerator
import argparse
from datasets import load_dataset, interleave_datasets
from itertools import cycle
from model_utils import ModelConfig, Model
from safetensors.torch import load_file
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer
#from dataset_utils import TokenizedSFTDataset


class TokenizedDataset(IterableDataset):
    def __init__(self, hf_dataset, hf_key, dataset_type, tokenizer, max_length):
        self.hf_dataset = hf_dataset
        self.hf_key = hf_key
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_length = max_length + 1

    def __iter__(self):
        buffer = []
        for data in self.hf_dataset:

            if self.dataset_type == "conversation":
                token_ids = self.tokenizer.apply_chat_template(data[self.hf_key])

            elif self.dataset_type == "qa":
                text = [{"content": data["instruction"], "role": "user"},
                        {"content": data["output"], "role": "assistant"}]
                token_ids = self.tokenizer.apply_chat_template(text)


            else:
                token_ids = self.tokenizer.encode(data[self.hf_key])
            buffer.extend(token_ids)

            while len(buffer) >= self.max_length:
                x = buffer[:self.max_length]
                # buffer = buffer[self.max_length:]
                buffer = []
                yield torch.tensor(x, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="train config for attn")
    parser.add_argument("--enable_fsdp", action="store_true", default=True,  help="boolean: enable/disable sharded model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="optimizers weight decay")
    parser.add_argument("--train_steps", type=int, default=100_000, help="train steps")
    parser.add_argument("--max_length", type=int, default=1024, help="model max context")
    parser.add_argument("--tokenizer", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="hf tokenizer")
    parser.add_argument("--num_layers", type=int, default=12, help="number of decoder layers")
    parser.add_argument("--num_heads", type=int, default=12, help="number of attention heads")
    parser.add_argument("--d_model", type=int, default=768, help="dimensions of model")
    parser.add_argument("--rate", type=float, default=.1, help="dropout rate")
    parser.add_argument("--bf16", action="store_true", default=True, help="enable bfloat16 training")
    parser.add_argument("--tf32", action="store_true", default=True, help="enable TensorFloat32 training")
    parser.add_argument("--dataset_name", type=str, default="Skylion007/openwebtext", help="name of huggingface dataset")
    parser.add_argument("--dataset_subset", type=str, default="default", help="huggingface dataset subset")
    parser.add_argument("--dataset_key", type=str, default="text", help="key for data in dataset")
    parser.add_argument("--dataset_type", type=str, default="pretraining",help="pretrain for pretraining, posttrain for conversational json")
    parser.add_argument("--savepath", type=str, default="model.pt", help="savepath for weights from training, should end in pt.")
    parser.add_argument("--save_interval", type=int, default=1000, help="how many steps between saves during training")
    args = parser.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    mixed_precision = "bf16" if args.bf16 else None
    accelerator = Accelerator(mixed_precision=mixed_precision)

    dataset = load_dataset(args.dataset_name, args.dataset_subset, split="train", streaming=True).shuffle(seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = cycle(dataset)
    config = ModelConfig(d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, rate=args.rate,
                         max_length=args.max_length, tokenizer=tokenizer)

    train_dataset = TokenizedDataset(dataset, args.dataset_key, args.dataset_type, tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    model_instance = Model(config)

    torch.compile(model_instance)

    optimizer = AdamW(model_instance.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model_instance, optimizer, train_dataloader = accelerator.prepare(model_instance, optimizer, train_dataloader)

    model_instance.train()
    step = 0

    for batch in train_dataloader:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits, loss = model_instance(inputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0 and accelerator.is_main_process:
            print(f"Step: {step} | Loss: {loss.item()}")

        step += 1

        if step % args.save_interval == 0 and accelerator.is_main_process:
            if args.enable_fsdp:
                accelerator.save_model(model_instance, "model_checkpoint")

            else:
                torch.save(model_instance.state_dict(), args.savepath)

        if step > args.train_steps:
            break

    accelerator.wait_for_everyone()
    print("training complete")


if __name__ == "__main__":
    main()
