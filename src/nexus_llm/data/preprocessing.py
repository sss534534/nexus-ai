"""
Data Preprocessing and Cleaning Module
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import hashlib

import numpy as np
from tqdm import tqdm

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Text cleaning
    clean_text: bool = True
    remove_html: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    
    # Unicode handling
    normalize_unicode: bool = True
    target_encoding: str = "utf-8"
    
    # Case handling
    lowercase: bool = False
    uppercase: bool = False
    
    # Punctuation
    remove_punctuation: bool = False
    normalize_punctuation: bool = True
    
    # Numbers
    remove_numbers: bool = False
    normalize_numbers: bool = False
    
    # Filtering
    min_length: int = 10
    max_length: int = 8192
    min_word_count: int = 5
    max_word_count: Optional[int] = None
    
    # Language detection
    language_detection: bool = False
    target_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    min_confidence: float = 0.9
    
    # Deduplication
    deduplication: bool = True
    dedup_method: str = "exact"  # exact, fuzzy, minhash
    dedup_threshold: float = 0.95
    
    # Quality filtering
    quality_filter: bool = True
    max_repetition_ratio: float = 0.5
    max_special_char_ratio: float = 0.3
    min_alpha_ratio: float = 0.5
    
    # Content filtering
    filter_profanity: bool = False
    filter_pii: bool = False
    filter_code_blocks: bool = False
    
    # Augmentation
    augmentation: bool = False
    aug_methods: List[str] = field(default_factory=lambda: ["synonym"])
    aug_probability: float = 0.1


class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_char_pattern = re.compile(r'[^\w\s\u4e00-\u9fff]')
    
    def clean(self, text: str) -> str:
        """Apply all cleaning operations."""
        
        if not self.config.clean_text:
            return text
        
        # Remove HTML
        if self.config.remove_html and BS4_AVAILABLE:
            text = self._remove_html(text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.email_pattern.sub('', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.phone_pattern.sub('', text)
        
        # Normalize unicode
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)
        
        # Normalize punctuation
        if self.config.normalize_punctuation:
            text = self._normalize_punctuation(text)
        
        # Case handling
        if self.config.lowercase:
            text = text.lower()
        elif self.config.uppercase:
            text = text.upper()
        
        # Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKC', text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Replace various quote marks
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Replace various dash marks
        text = text.replace('—', '-').replace('–', '-')
        
        # Replace various space marks
        text = text.replace('\u00A0', ' ')
        
        return text


class LanguageDetector:
    """Language detection for text filtering."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        if not LANGDETECT_AVAILABLE and config.language_detection:
            logger.warning("langdetect not available, language detection disabled")
    
    def detect(self, text: str) -> Tuple[str, float]:
        """Detect language of text."""
        
        if not LANGDETECT_AVAILABLE or not self.config.language_detection:
            return "unknown", 1.0
        
        try:
            from langdetect import detect, detect_langs
            
            # Get language probabilities
            lang_probs = detect_langs(text)
            
            if lang_probs:
                top_lang = str(lang_probs[0]).split(':')[0]
                confidence = float(str(lang_probs[0]).split(':')[1])
                return top_lang, confidence
            
            return detect(text), 1.0
        
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    def is_target_language(self, text: str) -> bool:
        """Check if text is in target language."""
        
        if not self.config.language_detection:
            return True
        
        lang, confidence = self.detect(text)
        
        if lang in self.config.target_languages and confidence >= self.config.min_confidence:
            return True
        
        return False


class Deduplicator:
    """Text deduplication using various methods."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.seen_minhashes: Set[str] = set()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        
        if not self.config.deduplication:
            return False
        
        if self.config.dedup_method == "exact":
            return self._is_exact_duplicate(text)
        elif self.config.dedup_method == "fuzzy":
            return self._is_fuzzy_duplicate(text)
        elif self.config.dedup_method == "minhash":
            return self._is_minhash_duplicate(text)
        
        return False
    
    def _is_exact_duplicate(self, text: str) -> bool:
        """Check for exact duplicates using hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def _is_fuzzy_duplicate(self, text: str) -> bool:
        """Check for fuzzy duplicates using n-gram similarity."""
        # Simple n-gram based similarity
        text_hash = self._compute_ngram_hash(text)
        
        for seen_hash in self.seen_hashes:
            similarity = self._compute_similarity(text_hash, seen_hash)
            if similarity >= self.config.dedup_threshold:
                return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def _is_minhash_duplicate(self, text: str) -> bool:
        """Check for duplicates using MinHash."""
        # Simplified MinHash implementation
        shingles = self._get_shingles(text)
        minhash = self._compute_minhash(shingles)
        
        for seen_minhash in self.seen_minhashes:
            similarity = self._estimate_jaccard(minhash, seen_minhash)
            if similarity >= self.config.dedup_threshold:
                return True
        
        self.seen_minhashes.add(minhash)
        return False
    
    def _compute_ngram_hash(self, text: str, n: int = 3) -> str:
        """Compute n-gram hash."""
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        ngram_str = ''.join(sorted(set(ngrams)))
        return hashlib.sha256(ngram_str.encode()).hexdigest()
    
    def _compute_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hashes."""
        # Simple character-level similarity
        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / max(len(hash1), len(hash2))
    
    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Get k-shingles from text."""
        return set(text[i:i+k] for i in range(len(text)-k+1))
    
    def _compute_minhash(self, shingles: Set[str], num_hashes: int = 128) -> str:
        """Compute MinHash signature."""
        signatures = []
        
        for i in range(num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = int(hashlib.md5(f"{shingle}:{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            signatures.append(min_hash)
        
        return ','.join(map(str, signatures))
    
    def _estimate_jaccard(self, minhash1: str, minhash2: str) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        sig1 = list(map(int, minhash1.split(',')))
        sig2 = list(map(int, minhash2.split(',')))
        
        matches = sum(a == b for a, b in zip(sig1, sig2))
        return matches / len(sig1)


class QualityFilter:
    """Quality filtering for text data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def filter(self, text: str) -> Tuple[bool, str]:
        """Apply quality filters."""
        
        if not self.config.quality_filter:
            return True, ""
        
        # Length checks
        if len(text) < self.config.min_length:
            return False, f"Text too short: {len(text)} < {self.config.min_length}"
        
        if len(text) > self.config.max_length:
            return False, f"Text too long: {len(text)} > {self.config.max_length}"
        
        # Word count checks
        words = text.split()
        if len(words) < self.config.min_word_count:
            return False, f"Too few words: {len(words)} < {self.config.min_word_count}"
        
        if self.config.max_word_count and len(words) > self.config.max_word_count:
            return False, f"Too many words: {len(words)} > {self.config.max_word_count}"
        
        # Repetition check
        if self._has_excessive_repetition(text):
            return False, "Excessive repetition detected"
        
        # Special character ratio
        special_char_ratio = self._compute_special_char_ratio(text)
        if special_char_ratio > self.config.max_special_char_ratio:
            return False, f"Too many special characters: {special_char_ratio:.2f}"
        
        # Alpha character ratio
        alpha_ratio = self._compute_alpha_ratio(text)
        if alpha_ratio < self.config.min_alpha_ratio:
            return False, f"Too few alphabetic characters: {alpha_ratio:.2f}"
        
        return True, ""
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive repetition."""
        words = text.split()
        if not words:
            return False
        
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)[0][1]
        repetition_ratio = most_common / len(words)
        
        return repetition_ratio > self.config.max_repetition_ratio
    
    def _compute_special_char_ratio(self, text: str) -> float:
        """Compute ratio of special characters."""
        if not text:
            return 0.0
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_chars / len(text)
    
    def _compute_alpha_ratio(self, text: str) -> float:
        """Compute ratio of alphabetic characters."""
        if not text:
            return 0.0
        
        alpha_chars = sum(1 for c in text if c.isalpha())
        return alpha_chars / len(text)


class DataAugmenter:
    """Data augmentation for training data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Synonym dictionaries (simplified)
        self.synonyms: Dict[str, List[str]] = {
            "good": ["great", "excellent", "wonderful"],
            "bad": ["poor", "terrible", "awful"],
            "happy": ["joyful", "cheerful", "delighted"],
        }
    
    def augment(self, text: str) -> str:
        """Apply data augmentation."""
        
        if not self.config.augmentation:
            return text
        
        import random
        
        if random.random() > self.config.aug_probability:
            return text
        
        words = text.split()
        
        for method in self.config.aug_methods:
            if method == "synonym":
                words = self._synonym_replacement(words)
            elif method == "deletion":
                words = self._random_deletion(words)
            elif method == "swap":
                words = self._random_swap(words)
        
        return ' '.join(words)
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """Replace words with synonyms."""
        import random
        
        new_words = words.copy()
        
        for i, word in enumerate(new_words):
            if word.lower() in self.synonyms and random.random() < 0.3:
                new_words[i] = random.choice(self.synonyms[word.lower()])
        
        return new_words
    
    def _random_deletion(self, words: List[str], p: float = 0.1) -> List[str]:
        """Randomly delete words."""
        import random
        
        if len(words) <= 1:
            return words
        
        return [word for word in words if random.random() > p]
    
    def _random_swap(self, words: List[str], n: int = 1) -> List[str]:
        """Randomly swap word positions."""
        import random
        
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words


class DataPreprocessor:
    """Main data preprocessing pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.text_cleaner = TextCleaner(self.config)
        self.language_detector = LanguageDetector(self.config)
        self.deduplicator = Deduplicator(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.augmenter = DataAugmenter(self.config)
        
        # Statistics
        self.stats = {
            "total": 0,
            "cleaned": 0,
            "filtered": 0,
            "duplicates": 0,
            "augmented": 0,
        }
    
    def process(self, text: str, apply_augmentation: bool = False) -> Optional[str]:
        """Process a single text sample."""
        
        self.stats["total"] += 1
        
        # Clean text
        cleaned_text = self.text_cleaner.clean(text)
        
        if not cleaned_text:
            self.stats["filtered"] += 1
            return None
        
        self.stats["cleaned"] += 1
        
        # Language detection
        if not self.language_detector.is_target_language(cleaned_text):
            self.stats["filtered"] += 1
            return None
        
        # Quality filtering
        is_valid, reason = self.quality_filter.filter(cleaned_text)
        if not is_valid:
            logger.debug(f"Quality filter rejected: {reason}")
            self.stats["filtered"] += 1
            return None
        
        # Deduplication
        if self.deduplicator.is_duplicate(cleaned_text):
            self.stats["duplicates"] += 1
            return None
        
        # Augmentation
        if apply_augmentation and self.config.augmentation:
            cleaned_text = self.augmenter.augment(cleaned_text)
            self.stats["augmented"] += 1
        
        return cleaned_text
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        text_column: str = "text",
        apply_augmentation: bool = False,
    ):
        """Process a data file."""
        
        logger.info(f"Processing {input_path} -> {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc="Processing"):
                try:
                    data = json.loads(line)
                    
                    # Extract text
                    if isinstance(data, dict):
                        text = data.get(text_column, '')
                    else:
                        text = str(data)
                    
                    # Process
                    processed_text = self.process(text, apply_augmentation)
                    
                    if processed_text:
                        # Update data
                        if isinstance(data, dict):
                            data[text_column] = processed_text
                        else:
                            data = {text_column: processed_text}
                        
                        # Write
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON: {line[:100]}")
                    continue
        
        logger.info(f"Processing complete. Statistics: {self.stats}")
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.jsonl",
        text_column: str = "text",
        apply_augmentation: bool = False,
    ):
        """Process all files in a directory."""
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        for input_file in input_dir.glob(pattern):
            output_file = output_dir / input_file.relative_to(input_dir)
            self.process_file(input_file, output_file, text_column, apply_augmentation)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            "total": 0,
            "cleaned": 0,
            "filtered": 0,
            "duplicates": 0,
            "augmented": 0,
        }


# Convenience functions
def clean_text(text: str, **kwargs) -> str:
    """Clean text with default settings."""
    config = PreprocessingConfig(**kwargs)
    cleaner = TextCleaner(config)
    return cleaner.clean(text)


def filter_quality(text: str, **kwargs) -> Tuple[bool, str]:
    """Filter text by quality."""
    config = PreprocessingConfig(**kwargs)
    quality_filter = QualityFilter(config)
    return quality_filter.filter(text)


def remove_duplicates(texts: List[str], method: str = "exact") -> List[str]:
    """Remove duplicate texts."""
    config = PreprocessingConfig(deduplication=True, dedup_method=method)
    deduplicator = Deduplicator(config)
    
    unique_texts = []
    for text in texts:
        if not deduplicator.is_duplicate(text):
            unique_texts.append(text)
    
    return unique_texts
