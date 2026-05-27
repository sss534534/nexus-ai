"""
Data Processing and Tokenization for Nexus-7B
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Iterator, Callable
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from tqdm import tqdm

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

try:
    from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, normalizers
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer
    HUGGINGFACE_TOKENIZER_AVAILABLE = True
except ImportError:
    HUGGINGFACE_TOKENIZER_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    tokenizer_type: str = "sentencepiece"
    tokenizer_path: str = "tokenizer/nexus_tokenizer.model"
    
    max_seq_length: int = 8192
    min_seq_length: int = 10
    
    train_file: str = "data/train.jsonl"
    valid_file: str = "data/valid.jsonl"
    test_file: str = "data/test.jsonl"
    
    num_workers: int = 16
    batch_size: int = 4
    shuffle: bool = True
    seed: int = 42
    
    # Special tokens
    bos_token: str = "<|beginoftext|>"
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    
    # Tokenizer training
    vocab_size: int = 152064
    character_coverage: float = 0.99995
    model_type: str = "bpe"
    
    # Data augmentation
    augmentation_enabled: bool = False
    augmentation_prob: float = 0.1


class NexusTokenizer:
    """Tokenizer wrapper supporting SentencePiece and HuggingFace tokenizers."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load tokenizer based on configuration."""
        tokenizer_path = Path(self.config.tokenizer_path)
        
        if self.config.tokenizer_type == "sentencepiece":
            if not SENTENCEPIECE_AVAILABLE:
                raise ImportError("sentencepiece is not installed")
            
            if tokenizer_path.exists():
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.load(str(tokenizer_path))
                logger.info(f"Loaded SentencePiece tokenizer from {tokenizer_path}")
            else:
                logger.warning(f"Tokenizer not found at {tokenizer_path}, will need to train")
        
        elif self.config.tokenizer_type == "huggingface":
            if not HUGGINGFACE_TOKENIZER_AVAILABLE:
                raise ImportError("tokenizers library is not installed")
            
            if tokenizer_path.exists():
                self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
                logger.info(f"Loaded HuggingFace tokenizer from {tokenizer_path}")
            else:
                logger.warning(f"Tokenizer not found at {tokenizer_path}, will need to train")
        
        else:
            raise ValueError(f"Unknown tokenizer type: {self.config.tokenizer_type}")
    
    def train(
        self,
        input_files: List[str],
        vocab_size: Optional[int] = None,
        output_path: Optional[str] = None,
    ):
        """Train tokenizer from input files."""
        vocab_size = vocab_size or self.config.vocab_size
        output_path = output_path or self.config.tokenizer_path
        
        if self.config.tokenizer_type == "sentencepiece":
            self._train_sentencepiece(input_files, vocab_size, output_path)
        elif self.config.tokenizer_type == "huggingface":
            self._train_huggingface(input_files, vocab_size, output_path)
        
        # Reload tokenizer after training
        self._load_tokenizer()
    
    def _train_sentencepiece(
        self,
        input_files: List[str],
        vocab_size: int,
        output_path: str,
    ):
        """Train SentencePiece tokenizer."""
        special_tokens = [
            self.config.pad_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.unk_token,
        ]
        
        # Additional special tokens for vertical domain
        domain_special_tokens = [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|instruction|>",
            "<|response|>",
            "<|context|>",
            "<|query|>",
            "<|answer|>",
        ]
        
        all_special_tokens = special_tokens + domain_special_tokens
        
        spm.SentencePieceTrainer.train(
            input=input_files,
            vocab_size=vocab_size,
            model_type=self.config.model_type,
            model_prefix=output_path.replace('.model', ''),
            character_coverage=self.config.character_coverage,
            user_defined_symbols=all_special_tokens,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=self.config.num_workers,
            train_extremely_large_corpus=True,
            max_sentence_length=16384,
        )
        
        logger.info(f"Trained SentencePiece tokenizer with vocab_size={vocab_size}")
    
    def _train_huggingface(
        self,
        input_files: List[str],
        vocab_size: int,
        output_path: str,
    ):
        """Train HuggingFace BPE tokenizer."""
        tokenizer = Tokenizer(models.BPE())
        
        # Normalizer
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ])
        
        # Pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        
        # Trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[
                self.config.pad_token,
                self.config.bos_token,
                self.config.eos_token,
                self.config.unk_token,
            ],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        
        # Train
        tokenizer.train(files=input_files, trainer=trainer)
        
        # Decoder
        tokenizer.decoder = decoders.ByteLevel()
        
        # Save
        tokenizer.save(output_path)
        logger.info(f"Trained HuggingFace tokenizer with vocab_size={vocab_size}")
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        if self.config.tokenizer_type == "sentencepiece":
            tokens = self.tokenizer.encode(text, out_type=int)
        else:
            tokens = self.tokenizer.encode(text).ids
        
        # Add special tokens
        if add_bos:
            tokens = [self.bos_token_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token_id]
        
        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.eos_token_id]
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        if skip_special_tokens:
            special_ids = [self.bos_token_id, self.eos_token_id, self.pad_token_id]
            token_ids = [t for t in token_ids if t not in special_ids]
        
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids)
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts to tensors."""
        all_tokens = []
        
        for text in texts:
            tokens = self.encode(text, max_length=max_length if truncation else None)
            all_tokens.append(tokens)
        
        # Find max length in batch
        max_len = max(len(t) for t in all_tokens)
        if max_length is not None and padding:
            max_len = min(max_len, max_length)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        
        for tokens in all_tokens:
            # Truncate if needed
            if truncation and len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            # Pad
            padding_length = max_len - len(tokens)
            padded_tokens = tokens + [self.pad_token_id] * padding_length
            
            # Attention mask
            mask = [1] * len(tokens) + [0] * padding_length
            
            input_ids.append(padded_tokens)
            attention_mask.append(mask)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.get_piece_size()
        else:
            return self.tokenizer.get_vocab_size()
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.bos_id()
        else:
            return self.tokenizer.token_to_id(self.config.bos_token)
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.eos_id()
        else:
            return self.tokenizer.token_to_id(self.config.eos_token)
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.pad_id()
        else:
            return self.tokenizer.token_to_id(self.config.pad_token)
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        if self.config.tokenizer_type == "sentencepiece":
            return self.tokenizer.unk_id()
        else:
            return self.tokenizer.token_to_id(self.config.unk_token)


class NexusDataset(IterableDataset):
    """Iterable dataset for efficient training data loading."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: NexusTokenizer,
        max_seq_length: int,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        
        # Count total samples
        self._count_samples()
    
    def _count_samples(self):
        """Count total number of samples in dataset."""
        self.total_samples = 0
        
        if self.data_path.is_file():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    self.total_samples += 1
        elif self.data_path.is_dir():
            for file_path in self.data_path.glob('*.jsonl'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for _ in f:
                        self.total_samples += 1
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset samples."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Multi-worker data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Split data across workers
            files = self._get_files()
            files_per_worker = len(files) // num_workers
            worker_files = files[worker_id * files_per_worker:(worker_id + 1) * files_per_worker]
        else:
            worker_files = self._get_files()
        
        while True:
            for file_path in worker_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    if self.shuffle:
                        np.random.seed(self.seed)
                        np.random.shuffle(lines)
                    
                    for line in lines:
                        sample = self._process_line(line)
                        if sample is not None:
                            yield sample
            
            if not self.infinite:
                break
    
    def _get_files(self) -> List[Path]:
        """Get list of data files."""
        if self.data_path.is_file():
            return [self.data_path]
        else:
            return sorted(self.data_path.glob('*.jsonl'))
    
    def _process_line(self, line: str) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single line from data file."""
        try:
            data = json.loads(line)
            
            # Extract text based on format
            if isinstance(data, dict):
                if 'text' in data:
                    text = data['text']
                elif 'input' in data and 'output' in data:
                    # Instruction format
                    text = self._format_instruction(data)
                elif 'prompt' in data and 'response' in data:
                    # Chat format
                    text = self._format_chat(data)
                else:
                    text = str(data)
            else:
                text = str(data)
            
            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
                max_length=self.max_seq_length,
            )
            
            if len(tokens) < 10:  # Skip too short sequences
                return None
            
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(tokens, dtype=torch.long),
            }
        
        except Exception as e:
            logger.warning(f"Error processing line: {e}")
            return None
    
    def _format_instruction(self, data: Dict) -> str:
        """Format instruction-response data."""
        instruction = data.get('input', '')
        response = data.get('output', '')
        
        return f"<|instruction|>\n{instruction}\n<|response|>\n{response}"
    
    def _format_chat(self, data: Dict) -> str:
        """Format chat conversation data."""
        prompt = data.get('prompt', '')
        response = data.get('response', '')
        
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    
    def __len__(self) -> int:
        """Return approximate dataset length."""
        return self.total_samples


class DataProcessor:
    """Data processing pipeline for training."""
    
    def __init__(self, config: DataConfig, tokenizer: NexusTokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def create_datasets(
        self,
        train_path: Optional[str] = None,
        valid_path: Optional[str] = None,
        test_path: Optional[str] = None,
    ) -> Dict[str, NexusDataset]:
        """Create train, validation, and test datasets."""
        train_path = train_path or self.config.train_file
        valid_path = valid_path or self.config.valid_file
        test_path = test_path or self.config.test_file
        
        datasets = {}
        
        if Path(train_path).exists():
            datasets['train'] = NexusDataset(
                train_path,
                self.tokenizer,
                self.config.max_seq_length,
                shuffle=self.config.shuffle,
                seed=self.config.seed,
                infinite=True,
            )
        
        if Path(valid_path).exists():
            datasets['valid'] = NexusDataset(
                valid_path,
                self.tokenizer,
                self.config.max_seq_length,
                shuffle=False,
                infinite=False,
            )
        
        if Path(test_path).exists():
            datasets['test'] = NexusDataset(
                test_path,
                self.tokenizer,
                self.config.max_seq_length,
                shuffle=False,
                infinite=False,
            )
        
        return datasets
    
    def create_dataloaders(
        self,
        datasets: Dict[str, NexusDataset],
        batch_size: Optional[int] = None,
    ) -> Dict[str, DataLoader]:
        """Create dataloaders for datasets."""
        batch_size = batch_size or self.config.batch_size
        
        dataloaders = {}
        
        for name, dataset in datasets.items():
            # Collate function
            collate_fn = self._create_collate_fn()
            
            dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
        
        return dataloaders
    
    def _create_collate_fn(self) -> Callable:
        """Create collate function for batching."""
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Find max length in batch
            max_len = min(
                max(sample['input_ids'].shape[0] for sample in batch),
                self.config.max_seq_length
            )
            
            input_ids = []
            labels = []
            attention_mask = []
            
            for sample in batch:
                ids = sample['input_ids']
                lbls = sample['labels']
                
                # Truncate if needed
                if ids.shape[0] > max_len:
                    ids = ids[:max_len]
                    lbls = lbls[:max_len]
                
                # Pad
                padding_length = max_len - ids.shape[0]
                
                padded_ids = torch.cat([
                    ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])
                
                padded_labels = torch.cat([
                    lbls,
                    torch.full((padding_length,), -100, dtype=torch.long)  # Ignore padding in loss
                ])
                
                mask = torch.cat([
                    torch.ones(ids.shape[0], dtype=torch.long),
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                
                input_ids.append(padded_ids)
                labels.append(padded_labels)
                attention_mask.append(mask)
            
            return {
                "input_ids": torch.stack(input_ids),
                "labels": torch.stack(labels),
                "attention_mask": torch.stack(attention_mask),
            }
        
        return collate_fn
    
    def preprocess_raw_data(
        self,
        raw_dir: str,
        output_dir: str,
        format: str = "jsonl",
    ):
        """Preprocess raw data files into training format."""
        raw_path = Path(raw_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all files
        all_texts = []
        
        for file_path in tqdm(list(raw_path.glob('*')), desc="Processing files"):
            if file_path.suffix in ['.txt', '.json', '.jsonl', '.csv']:
                texts = self._process_file(file_path, format)
                all_texts.extend(texts)
        
        # Split into train/valid/test
        total = len(all_texts)
        train_end = int(total * 0.9)
        valid_end = int(total * 0.95)
        
        train_data = all_texts[:train_end]
        valid_data = all_texts[train_end:valid_end]
        test_data = all_texts[valid_end:]
        
        # Write to output files
        self._write_jsonl(train_data, output_path / 'train.jsonl')
        self._write_jsonl(valid_data, output_path / 'valid.jsonl')
        self._write_jsonl(test_data, output_path / 'test.jsonl')
        
        logger.info(f"Processed {total} samples into {output_dir}")
    
    def _process_file(self, file_path: Path, format: str) -> List[Dict]:
        """Process a single file."""
        texts = []
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= self.config.min_seq_length:
                        texts.append({'text': line})
        
        elif file_path.suffix in ['.json', '.jsonl']:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data)
        
        elif file_path.suffix == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    texts.append(row)
        
        return texts
    
    def _write_jsonl(self, data: List[Dict], output_path: Path):
        """Write data to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


class VerticalDomainProcessor(DataProcessor):
    """Specialized processor for vertical domain data."""
    
    def __init__(self, config: DataConfig, tokenizer: NexusTokenizer, domain: str):
        super().__init__(config, tokenizer)
        self.domain = domain
    
    def _format_domain_data(self, data: Dict) -> str:
        """Format domain-specific data."""
        # Domain-specific formatting templates
        templates = {
            "finance": self._format_finance,
            "medical": self._format_medical,
            "legal": self._format_legal,
            "code": self._format_code,
        }
        
        formatter = templates.get(self.domain, self._format_instruction)
        return formatter(data)
    
    def _format_finance(self, data: Dict) -> str:
        """Format financial domain data."""
        context = data.get('context', '')
        question = data.get('question', '')
        answer = data.get('answer', '')
        
        return f"<|context|>\n{context}\n<|query|>\n{question}\n<|answer|>\n{answer}"
    
    def _format_medical(self, data: Dict) -> str:
        """Format medical domain data."""
        symptoms = data.get('symptoms', '')
        diagnosis = data.get('diagnosis', '')
        treatment = data.get('treatment', '')
        
        return f"<|context|>\n症状: {symptoms}\n<|query|>\n诊断结果\n<|answer|>\n{diagnosis}\n治疗方案: {treatment}"
    
    def _format_legal(self, data: Dict) -> str:
        """Format legal domain data."""
        case = data.get('case', '')
        question = data.get('question', '')
        analysis = data.get('analysis', '')
        
        return f"<|context|>\n{case}\n<|query|>\n{question}\n<|answer|>\n{analysis}"
    
    def _format_code(self, data: Dict) -> str:
        """Format code domain data."""
        description = data.get('description', '')
        code = data.get('code', '')
        language = data.get('language', 'python')
        
        return f"<|instruction|>\n{description}\nLanguage: {language}\n<|response|>\n```{language}\n{code}\n```"