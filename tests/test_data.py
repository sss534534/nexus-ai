"""
Tests for Data Processing Module
"""

import pytest
import json
import tempfile
from pathlib import Path

from nexus_llm.data import (
    NexusTokenizer,
    DataProcessor,
    DataConfig,
    NexusDataset,
)
from nexus_llm.data.preprocessing import (
    TextCleaner,
    LanguageDetector,
    Deduplicator,
    QualityFilter,
    DataPreprocessor,
    PreprocessingConfig,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""
    
    def test_default_config(self):
        """Test default preprocessing configuration."""
        config = PreprocessingConfig()
        
        assert config.clean_text is True
        assert config.remove_html is True
        assert config.min_length == 10
        assert config.max_length == 8192
        assert config.deduplication is True
    
    def test_custom_config(self):
        """Test custom preprocessing configuration."""
        config = PreprocessingConfig(
            min_length=50,
            max_length=2048,
            deduplication=False,
        )
        
        assert config.min_length == 50
        assert config.max_length == 2048
        assert config.deduplication is False


class TestTextCleaner:
    """Tests for TextCleaner."""
    
    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        config = PreprocessingConfig(normalize_whitespace=True)
        cleaner = TextCleaner(config)
        
        text = "Hello    world\t\t\n\n  test"
        cleaned = cleaner.clean(text)
        
        assert "  " not in cleaned
        assert cleaned == "Hello world test"
    
    def test_remove_urls(self):
        """Test URL removal."""
        config = PreprocessingConfig(remove_urls=True)
        cleaner = TextCleaner(config)
        
        text = "Visit https://example.com for more info"
        cleaned = cleaner.clean(text)
        
        assert "https://example.com" not in cleaned
    
    def test_remove_emails(self):
        """Test email removal."""
        config = PreprocessingConfig(remove_emails=True)
        cleaner = TextCleaner(config)
        
        text = "Contact us at test@example.com"
        cleaned = cleaner.clean(text)
        
        assert "test@example.com" not in cleaned
    
    def test_normalize_punctuation(self):
        """Test punctuation normalization."""
        config = PreprocessingConfig(normalize_punctuation=True)
        cleaner = TextCleaner(config)
        
        text = '"Hello" — world'
        cleaned = cleaner.clean(text)
        
        assert '"' in cleaned  # Normalized quote
        assert '—' not in cleaned  # Replaced with -


class TestQualityFilter:
    """Tests for QualityFilter."""
    
    def test_length_filter(self):
        """Test length-based filtering."""
        config = PreprocessingConfig(
            min_length=10,
            max_length=100,
        )
        filter_obj = QualityFilter(config)
        
        # Too short
        is_valid, _ = filter_obj.filter("Hi")
        assert is_valid is False
        
        # Valid length
        is_valid, _ = filter_obj.filter("This is a valid text with enough words.")
        assert is_valid is True
    
    def test_word_count_filter(self):
        """Test word count filtering."""
        config = PreprocessingConfig(min_word_count=5)
        filter_obj = QualityFilter(config)
        
        is_valid, _ = filter_obj.filter("One two three four")
        assert is_valid is False
        
        is_valid, _ = filter_obj.filter("One two three four five six")
        assert is_valid is True
    
    def test_repetition_filter(self):
        """Test repetition detection."""
        config = PreprocessingConfig(max_repetition_ratio=0.3)
        filter_obj = QualityFilter(config)
        
        # High repetition
        text = "word " * 20 + "other words here"
        is_valid, _ = filter_obj.filter(text)
        assert is_valid is False
        
        # Low repetition
        text = "The quick brown fox jumps over the lazy dog"
        is_valid, _ = filter_obj.filter(text)
        assert is_valid is True
    
    def test_special_char_ratio(self):
        """Test special character ratio filtering."""
        config = PreprocessingConfig(max_special_char_ratio=0.3)
        filter_obj = QualityFilter(config)
        
        # Too many special chars
        text = "!!!@@@###$$$%%%"
        is_valid, _ = filter_obj.filter(text)
        assert is_valid is False
        
        # Normal text
        text = "Hello, world! How are you?"
        is_valid, _ = filter_obj.filter(text)
        assert is_valid is True


class TestDeduplicator:
    """Tests for Deduplicator."""
    
    def test_exact_deduplication(self):
        """Test exact duplicate detection."""
        config = PreprocessingConfig(
            deduplication=True,
            dedup_method="exact",
        )
        dedup = Deduplicator(config)
        
        text1 = "This is a unique text"
        text2 = "This is a unique text"
        text3 = "This is different"
        
        assert dedup.is_duplicate(text1) is False
        assert dedup.is_duplicate(text2) is True  # Duplicate
        assert dedup.is_duplicate(text3) is False
    
    def test_fuzzy_deduplication(self):
        """Test fuzzy duplicate detection."""
        config = PreprocessingConfig(
            deduplication=True,
            dedup_method="fuzzy",
            dedup_threshold=0.9,
        )
        dedup = Deduplicator(config)
        
        text1 = "The quick brown fox jumps"
        text2 = "The quick brown fox jumps"  # Exact duplicate
        text3 = "A completely different text"
        
        assert dedup.is_duplicate(text1) is False
        assert dedup.is_duplicate(text2) is True
        assert dedup.is_duplicate(text3) is False


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""
    
    def test_process_text(self):
        """Test text processing pipeline."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        
        text = "  Hello   world!  "
        result = preprocessor.process(text)
        
        assert result is not None
        assert result == "Hello world!"
    
    def test_process_invalid_text(self):
        """Test processing of invalid text."""
        config = PreprocessingConfig(min_length=100)
        preprocessor = DataPreprocessor(config)
        
        text = "Too short"
        result = preprocessor.process(text)
        
        assert result is None
    
    def test_process_file(self):
        """Test file processing."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"text": "Hello world"}) + "\n")
            f.write(json.dumps({"text": "Test text"}) + "\n")
            input_path = f.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            preprocessor.process_file(Path(input_path), Path(output_path))
            
            # Check output
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 2
        finally:
            Path(input_path).unlink()
            Path(output_path).unlink()
    
    def test_statistics(self):
        """Test statistics tracking."""
        config = PreprocessingConfig()
        preprocessor = DataPreprocessor(config)
        
        # Process some texts
        preprocessor.process("Valid text here")
        preprocessor.process("Short")  # Will be filtered
        preprocessor.process("Another valid text")
        
        stats = preprocessor.get_statistics()
        
        assert stats["total"] == 3
        assert stats["filtered"] >= 1


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_config(self):
        """Test default data configuration."""
        config = DataConfig()
        
        assert config.tokenizer_type == "sentencepiece"
        assert config.max_seq_length == 8192
        assert config.batch_size == 4
    
    def test_domain_data_paths(self):
        """Test domain-specific data paths."""
        config = DataConfig()
        
        assert "general" in config.domain_data
        assert "code" in config.domain_data
        assert "finance" in config.domain_data


class TestNexusDataset:
    """Tests for NexusDataset."""
    
    def test_dataset_creation(self, tmp_path):
        """Test dataset creation."""
        # Create test data file
        data_file = tmp_path / "test.jsonl"
        with open(data_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Sample text {i}"}) + "\n")
        
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return [1, 2, 3, 4, 5]
            
            @property
            def pad_token_id(self):
                return 0
        
        dataset = NexusDataset(
            data_path=str(data_file),
            tokenizer=MockTokenizer(),
            max_seq_length=512,
        )
        
        assert dataset.total_samples == 10


# Convenience function tests
class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_clean_text(self):
        """Test clean_text function."""
        from nexus_llm.data.preprocessing import clean_text
        
        text = "  Hello   world  "
        cleaned = clean_text(text)
        
        assert cleaned == "Hello world"
    
    def test_filter_quality(self):
        """Test filter_quality function."""
        from nexus_llm.data.preprocessing import filter_quality
        
        text = "Hi"
        is_valid, reason = filter_quality(text, min_length=10)
        
        assert is_valid is False
        assert "short" in reason.lower()
    
    def test_remove_duplicates(self):
        """Test remove_duplicates function."""
        from nexus_llm.data.preprocessing import remove_duplicates
        
        texts = ["Hello", "Hello", "World"]
        unique = remove_duplicates(texts)
        
        assert len(unique) == 2
        assert "Hello" in unique
        assert "World" in unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
