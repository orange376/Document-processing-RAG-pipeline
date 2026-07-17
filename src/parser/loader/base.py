from abc import ABC, abstractmethod
from src.domain import Document


class DocumentLoader(ABC):
    """文档加载器的抽象基类"""

    @abstractmethod
    def load(self, path: str) -> Document:
        """加载文档，返回 Document 实例"""
        ...
