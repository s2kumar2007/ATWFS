from pathlib import Path


def test_runtime_dependencies():
    text = Path('requirements.txt').read_text()
    for name in ('onnxruntime', 'onnx', 'pyserial', 'pyflakes'):
        assert name in text.splitlines()
