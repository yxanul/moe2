import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Iterator, Dict
import random


class FineWebEduDataset(IterableDataset):
    """Streaming dataset for FineWeb-Edu with GPT-2 tokenizer."""
    
    def __init__(
        self,
        max_seq_len: int = 2048,
        tokenizer_name: str = "gpt2",
        streaming: bool = True,
        buffer_size: int = 10000,
        seed: int = 42,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.streaming = streaming
        self.buffer_size = buffer_size
        self.seed = int(seed)
        self._epoch = 0
        
        # Initialize GPT-2 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # GPT-2 doesn't have a pad token by default, set it to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Verify vocab size (GPT-2 has 50257 tokens, we pad to 50304)
        self.original_vocab_size = len(self.tokenizer)  # 50257
        self.padded_vocab_size = 50304  # Next multiple of 128
        
        # Load base streaming dataset (unshuffled here)
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=streaming
        )

    def set_epoch(self, epoch: int):
        """Set current epoch for reshuffling streamed data."""
        self._epoch = int(epoch)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        # Prepare per-epoch shuffled and worker-sharded stream
        ds = self.dataset
        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
            # Reshuffle per epoch if supported
            try:
                ds.set_epoch(self._epoch)
            except Exception:
                pass

        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers and worker.num_workers > 1:
            try:
                ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
            except Exception:
                pass

        for sample in ds:
            text = sample.get('text', '')
            
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue
            
            # Tokenize the text
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors='pt',
                padding='max_length',
                return_attention_mask=True,
            )
            
            # Extract tensors
            input_ids = tokens['input_ids'].squeeze(0)  # [seq_len]
            attention_mask = tokens['attention_mask'].squeeze(0)  # [seq_len]
            
            # Create labels (shifted input_ids for next-token prediction)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding in loss
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }


class DataCollator:
    """Custom data collator for batching."""
    
    def __init__(self, pad_token_id: int = 50256):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """Collate batch of samples."""
        # Stack tensors
        input_ids = torch.stack([sample['input_ids'] for sample in batch])
        attention_mask = torch.stack([sample['attention_mask'] for sample in batch])
        labels = torch.stack([sample['labels'] for sample in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def create_dataloader(
    batch_size: int,
    max_seq_len: int = 2048,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    """Create a DataLoader for FineWeb-Edu dataset."""
    
    # Create dataset
    dataset = FineWebEduDataset(
        max_seq_len=max_seq_len,
        tokenizer_name="gpt2",
        streaming=True,
        buffer_size=10000,
    )
    
    # Create data collator
    collator = DataCollator(pad_token_id=dataset.tokenizer.pad_token_id)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=True,
    )
    
    return dataloader, dataset.tokenizer


def calculate_dataset_stats(tokenizer_name: str = "gpt2"):
    """Calculate and print dataset statistics."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    original_vocab = len(tokenizer)
    
    # Calculate padding
    def next_multiple_of_n(value, n=128):
        return ((value + n - 1) // n) * n
    
    padded_vocab = next_multiple_of_n(original_vocab, n=128)
    
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Original vocab size: {original_vocab}")
    print(f"Padded vocab size: {padded_vocab}")
    print(f"Padding added: {padded_vocab - original_vocab}")
    
    return original_vocab, padded_vocab


if __name__ == "__main__":
    # Test dataset loading
    print("Testing FineWeb-Edu dataset loading...")
    
    # Print stats
    original_vocab, padded_vocab = calculate_dataset_stats()
    
    # Create a small test dataloader
    dataloader, tokenizer = create_dataloader(
        batch_size=2,
        max_seq_len=512,
        num_workers=0,  # For testing
    )
    
    # Test iteration
    print("\nTesting data iteration...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        if i >= 2:  # Just test a few batches
            break
    
    print("\nDataset test completed successfully!")
