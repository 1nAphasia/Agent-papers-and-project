from typing import Dict, Any, List, Optional, Union
import numpy as np
from pathlib import Path
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)
from config.logger import get_logger
from langchain_community.embeddings import DashScopeEmbeddings

logger = get_logger(__name__)

class DocumentProcessor:
    """文档处理器：清洗、分块、向量化"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-v4",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embedding_model = DashScopeEmbeddings(model=embedding_model,dashscope_api_key='sk-eed6accea0594ebabe804410af709a80')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # 文件类型到加载器的映射
        self._loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader, 
            ".pdf": PyPDFLoader,
            ".html": UnstructuredHTMLLoader
        }

    async def process_file(
        self, 
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        处理单个文件:加载、分块、向量化
        返回可直接传入add_document的文档列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # 1. 加载文档
        loader_cls = self._loaders.get(file_path.suffix.lower())
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        loader = loader_cls(str(file_path))
        documents = await asyncio.to_thread(loader.load)
        
        # 2. 文本分块
        chunks = self.text_splitter.split_documents(documents)
        
        # 3. 向量化(批处理以提高效率)
        texts = [chunk.page_content for chunk in chunks]
        embeddings: List[List[float]] = []
        batch_size = 10

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 调用阻塞的 embed_documents 放到线程池执行，避免阻塞事件循环
            emb_batch = await asyncio.to_thread(self.embedding_model.embed_documents, batch)

            for item in emb_batch:
                embeddings.append(item)
            
        
        # 4. 准备返回数据
        results = []
        base_metadata = metadata or {}
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                **base_metadata,
                "source": str(file_path),
                "chunk_index": i,
                **chunk.metadata  # 保留原文档元数据
            }
            
            results.append({
                "id_str": f"{file_path.stem}_{i}",
                "text": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
            
        return results

    async def process_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """处理整个目录下的文档"""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
            
        pattern = "**/*" if recursive else "*"
        all_results = []
        
        for file_path in dir_path.glob(pattern):
            if file_path.suffix.lower() in self._loaders:
                try:
                    results = await self.process_file(file_path, metadata)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    
        return all_results


if __name__ == "__main__":
    import json
    import os

    async def main():
        # 初始化处理器
        dp = DocumentProcessor()

        # 假设我们要处理的目录路径
        input_dir = "./docs"   # 例如: 放了若干 txt/md/pdf/html
        output_path = "./processed_results.json"

        # 执行异步目录处理
        all_results = await dp.process_directory(input_dir)

        # 输出结果统计
        print(f"✅ 共处理 {len(all_results)} 个文档块")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 写入到 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"📝 结果已保存至: {output_path}")

    # 运行异步任务
    asyncio.run(main())
