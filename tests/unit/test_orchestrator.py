import pytest
from src.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    def test_initialization(self):
        orchestrator = PipelineOrchestrator()
        assert orchestrator is not None

    def test_get_loader_pdf(self):
        from pathlib import Path
        orch = PipelineOrchestrator()
        loader = orch._get_loader(Path("test.pdf"))
        from src.parser.loader import PDFLoader
        assert isinstance(loader, PDFLoader)

    def test_get_loader_docx(self):
        from pathlib import Path
        orch = PipelineOrchestrator()
        loader = orch._get_loader(Path("test.docx"))
        from src.parser.loader import WordLoader
        assert isinstance(loader, WordLoader)

    def test_get_loader_unsupported(self):
        from pathlib import Path
        orch = PipelineOrchestrator()
        with pytest.raises(ValueError, match="不支持的文件格式"):
            orch._get_loader(Path("test.txt"))
