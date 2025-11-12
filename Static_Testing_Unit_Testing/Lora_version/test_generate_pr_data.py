"""
Comprehensive test suite for synthetic PR data generator.

Tests cover:
- API integration and mocking
- Data parsing and validation
- File I/O operations
- Error handling
- Edge cases
"""

import json
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest


class TestGeneratePRAndReview:
    """Test suite for generate_pr_and_review function."""

    @pytest.fixture
    def mock_groq_response(self):
        """Fixture providing a properly formatted mock LLM response."""
        return """PR Title: Add user authentication feature
PR Description: Implements JWT-based authentication system
Code Diff:
```python
+def authenticate_user(username, password):
+    token = generate_jwt(username)
+    return token
-def login(user):
-    return None
```
Review: LGTM! Clean implementation of JWT auth."""

    @pytest.fixture
    def mock_groq_client(self, mock_groq_response):
        """Fixture providing a mocked Groq client."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = mock_groq_response
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def temp_output_file(self):
        """Fixture providing a temporary output file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for all tests."""
        with patch.dict(os.environ, {'API_KEY': 'test_key_12345', 'GROQ_API_KEY': 'test_key_12345'}):
            yield

    def test_generate_basic_functionality(self, mock_groq_client, temp_output_file):
        """Test basic generation of PR data."""
        # Mock both the Groq class and the module-level client
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            # Force reimport to use mocked Groq
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        # Verify file was created
        assert os.path.exists(temp_output_file)
        
        # Verify content
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert 'prompt' in data
            assert 'completion' in data

    def test_generate_multiple_samples(self, mock_groq_client, temp_output_file):
        """Test generation of multiple PR samples."""
        n_samples = 5
        
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=n_samples, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == n_samples

    def test_prompt_structure(self, mock_groq_client, temp_output_file):
        """Test that generated prompt contains required fields."""
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            prompt = data['prompt']
            
            assert 'PR Title:' in prompt
            assert 'PR Description:' in prompt
            assert 'Code Diff:' in prompt
            assert 'Review:' in prompt

    def test_completion_format(self, mock_groq_client, temp_output_file):
        """Test that completion has proper format."""
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            completion = data['completion']
            
            # Should start with a space
            assert completion.startswith(' ')
            # Should contain review text
            assert len(completion.strip()) > 0

    def test_bold_formatting_removal(self, temp_output_file):
        """Test that bold markdown formatting is removed."""
        mock_response_with_bold = """**PR Title:** Add feature
**PR Description:** Description here
Code Diff:
```python
+new_code()
```
**Review:** Looks good"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = mock_response_with_bold
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            # Should not contain ** in the parsed content
            assert '**' not in data['prompt']

    def test_code_diff_extraction(self, mock_groq_client, temp_output_file):
        """Test that code diff blocks are properly extracted."""
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            prompt = data['prompt']
            
            # Should contain code block markers
            assert '```' in prompt

    def test_parsing_error_handling(self, temp_output_file):
        """Test handling of unparseable responses."""
        malformed_response = "This is a completely malformed response without proper structure"
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = malformed_response
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            # Should not raise an exception
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        # Verify file was still created with fallback data
        assert os.path.exists(temp_output_file)
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            assert 'prompt' in data
            assert 'completion' in data

    def test_missing_diff_markers(self, temp_output_file):
        """Test handling when code diff markers are missing."""
        response_no_diff = """PR Title: Test
PR Description: Test description
Review: Test review"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_no_diff
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        # Should still generate data
        assert os.path.exists(temp_output_file)

    def test_api_parameters(self, temp_output_file):
        """Test that API is called with correct parameters."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "PR Title: Test\nPR Description: Test\nCode Diff:\n```\n+code\n```\nReview: Good"
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        # Verify API was called with correct parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'llama-3.1-8b-instant'
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 400
        assert len(call_args[1]['messages']) == 1
        assert call_args[1]['messages'][0]['role'] == 'user'

    def test_jsonl_format(self, mock_groq_client, temp_output_file):
        """Test that output file is valid JSONL format."""
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=3, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Each line should be valid JSON
            for line in lines:
                data = json.loads(line)
                assert isinstance(data, dict)

    def test_file_encoding_utf8(self, temp_output_file):
        """Test that file is written with UTF-8 encoding."""
        # Create response with unicode characters
        unicode_response = """PR Title: Add émojis 🚀
PR Description: Support für unicode
Code Diff:
```python
+# Comment with 中文
```
Review: Looks good! ✓"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = unicode_response
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        # Should be able to read with UTF-8 encoding
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'émojis' in content or '🚀' in content

    def test_default_parameters(self, mock_groq_client):
        """Test function with default parameters."""
        output_file = 'train.jsonl'
        try:
            with patch('generate_pr_data.Groq', return_value=mock_groq_client):
                if 'generate_pr_data' in sys.modules:
                    del sys.modules['generate_pr_data']
                
                import generate_pr_data
                generate_pr_data.client = mock_groq_client
                generate_pr_data.generate_pr_and_review()
            
            # Should create default file
            assert os.path.exists(output_file)
        finally:
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_output_file_creation(self, mock_groq_client):
        """Test that output file is created if it doesn't exist."""
        output_path = 'test_nonexistent_dir/output.jsonl'
        
        # Ensure parent directory exists
        os.makedirs('test_nonexistent_dir', exist_ok=True)
        
        try:
            with patch('generate_pr_data.Groq', return_value=mock_groq_client):
                if 'generate_pr_data' in sys.modules:
                    del sys.modules['generate_pr_data']
                
                import generate_pr_data
                generate_pr_data.client = mock_groq_client
                generate_pr_data.generate_pr_and_review(n=1, output_file=output_path)
            
            assert os.path.exists(output_path)
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists('test_nonexistent_dir'):
                os.rmdir('test_nonexistent_dir')

    def test_empty_review_fallback(self, temp_output_file):
        """Test fallback when review cannot be parsed."""
        response_no_review = """PR Title: Test
PR Description: Test description
Code Diff:
```python
+code
```"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_no_review
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            # Should have empty completion on parse failure
            assert data['completion'] == ' '

    def test_api_call_count(self, mock_groq_client, temp_output_file):
        """Test that API is called n times."""
        n_samples = 3
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=n_samples, output_file=temp_output_file)
        
        # Verify API was called n times
        assert mock_groq_client.chat.completions.create.call_count == n_samples

    def test_console_output(self, mock_groq_client, temp_output_file, capsys):
        """Test that success message is printed."""
        with patch('generate_pr_data.Groq', return_value=mock_groq_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_groq_client
            generate_pr_data.generate_pr_and_review(n=2, output_file=temp_output_file)
        
        captured = capsys.readouterr()
        assert '✅' in captured.out
        assert '2 samples' in captured.out
        assert temp_output_file in captured.out

    def test_whitespace_handling(self, temp_output_file):
        """Test handling of extra whitespace in responses."""
        response_with_whitespace = """  PR Title:   Test Title  
  PR Description:   Test Description  
Code Diff:
```python
+code
```
  Review:   Test Review  """
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_with_whitespace
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=1, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            # Should have trimmed whitespace
            assert 'Test Title' in data['prompt']


class TestEnvironmentSetup:
    """Tests for environment and setup."""

    @patch.dict(os.environ, {'API_KEY': 'test_key_123'})
    def test_api_key_loading(self):
        """Test that API key is loaded from environment."""
        api_key = os.getenv('API_KEY')
        assert api_key == 'test_key_123'

    @patch.dict(os.environ, {'API_KEY': 'test_key_123', 'GROQ_API_KEY': 'test_key_123'})
    @patch('generate_pr_data.load_dotenv')
    def test_env_file_loading(self, mock_load_dotenv):
        """Test that dotenv is called to load environment."""
        if 'generate_pr_data' in sys.modules:
            del sys.modules['generate_pr_data']
        
        with patch('generate_pr_data.Groq'):
            import generate_pr_data
            mock_load_dotenv.assert_called()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for all tests."""
        with patch.dict(os.environ, {'API_KEY': 'test_key_12345', 'GROQ_API_KEY': 'test_key_12345'}):
            yield

    def test_zero_samples(self, temp_output_file):
        """Test generation with n=0."""
        mock_client = MagicMock()
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=0, output_file=temp_output_file)
        
        # Should create empty file
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 0

    def test_large_batch(self, temp_output_file):
        """Test generation with large number of samples."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "PR Title: T\nPR Description: D\nCode Diff:\n```\n+c\n```\nReview: R"
        
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('generate_pr_data.Groq', return_value=mock_client):
            if 'generate_pr_data' in sys.modules:
                del sys.modules['generate_pr_data']
            
            import generate_pr_data
            generate_pr_data.client = mock_client
            generate_pr_data.generate_pr_and_review(n=100, output_file=temp_output_file)
        
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=generate_pr_data', '--cov-report=html'])