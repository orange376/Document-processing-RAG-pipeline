import pytest
from pathlib import Path
from src.parser.loader.pdf_loader import PDFLoader


@pytest.fixture
def sample_pdf():
    """生成一个测试用 PDF（内存中写入临时文件）"""
    import fitz
    tmpdir = Path(__file__).resolve().parent.parent / "fixtures"
    tmpdir.mkdir(parents=True, exist_ok=True)
    pdf_path = tmpdir / "sample_test.pdf"

    if not pdf_path.exists():
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "这是标题", fontsize=16)
        page.insert_text((50, 150), "这是正文段落，包含一些测试内容。", fontsize=11)
        page.insert_text((50, 200), "这是第二段。", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()

    return str(pdf_path)


class TestPDFLoader:
    def test_load_basic(self, sample_pdf):
        loader = PDFLoader()
        doc = loader.load(sample_pdf)

        assert doc.filename == "sample_test.pdf"
        assert doc.file_type == "pdf"
        assert doc.total_pages == 1
        assert len(doc.pages) == 1

        page = doc.pages[0]
        assert page.width > 0
        assert page.height > 0
        assert len(page.blocks) > 0

    def test_file_not_found(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.pdf")

    def test_wrong_extension(self, tmp_path):
        loader = PDFLoader()
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            loader.load(str(txt_file))
