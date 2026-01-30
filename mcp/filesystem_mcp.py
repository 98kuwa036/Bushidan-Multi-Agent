"""
Bushidan Multi-Agent System v9.1 - Filesystem MCP

File system operations MCP for Ashigaru execution layer.
Provides secure file read/write/search operations.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.logger import get_logger


logger = get_logger(__name__)


class FilesystemMCP:
    """
    Filesystem MCP - File operations for Ashigaru
    
    Provides secure file system operations:
    - File reading and writing
    - Directory listing
    - File search
    - Path validation and sandboxing
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize filesystem MCP"""
        logger.info("📁 Initializing Filesystem MCP...")
        
        # Ensure base path exists
        self.base_path.mkdir(exist_ok=True)
        
        self.initialized = True
        logger.info(f"✅ Filesystem MCP initialized - Base: {self.base_path}")
    
    def _validate_path(self, path: str) -> Path:
        """Validate and resolve path within base directory"""
        
        target_path = (self.base_path / path).resolve()
        
        # Ensure path is within base directory (security)
        if not str(target_path).startswith(str(self.base_path)):
            raise ValueError(f"Path outside allowed directory: {path}")
        
        return target_path
    
    async def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read file content"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            file_path = self._validate_path(path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            
            logger.info(f"📖 Read file: {path} ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to read file {path}: {e}")
            raise
    
    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write content to file"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            file_path = self._validate_path(path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)
            
            logger.info(f"📝 Wrote file: {path} ({len(content)} chars)")
            
        except Exception as e:
            logger.error(f"❌ Failed to write file {path}: {e}")
            raise
    
    async def list_directory(self, path: str = ".") -> List[Dict[str, Any]]:
        """List directory contents"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            dir_path = self._validate_path(path)
            
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")
            
            if not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {path}")
            
            entries = []
            for item in dir_path.iterdir():
                try:
                    stat = item.stat()
                    entries.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "path": str(item.relative_to(self.base_path))
                    })
                except OSError:
                    continue  # Skip items we can't access
            
            logger.info(f"📂 Listed directory: {path} ({len(entries)} items)")
            return entries
            
        except Exception as e:
            logger.error(f"❌ Failed to list directory {path}: {e}")
            raise
    
    async def search_files(self, pattern: str, path: str = ".", max_results: int = 50) -> List[str]:
        """Search for files matching pattern"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            search_path = self._validate_path(path)
            
            if not search_path.exists():
                raise FileNotFoundError(f"Search path not found: {path}")
            
            results = []
            
            # Use glob for pattern matching
            if search_path.is_dir():
                for match in search_path.rglob(pattern):
                    if match.is_file():
                        relative_path = str(match.relative_to(self.base_path))
                        results.append(relative_path)
                        
                        if len(results) >= max_results:
                            break
            
            logger.info(f"🔍 Found {len(results)} files matching '{pattern}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ File search failed: {e}")
            return []
    
    async def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        
        try:
            file_path = self._validate_path(path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    async def directory_exists(self, path: str) -> bool:
        """Check if directory exists"""
        
        try:
            dir_path = self._validate_path(path)
            return dir_path.exists() and dir_path.is_dir()
        except Exception:
            return False
    
    async def create_directory(self, path: str) -> None:
        """Create directory"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            dir_path = self._validate_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📁 Created directory: {path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create directory {path}: {e}")
            raise
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file information"""
        
        if not self.initialized:
            await self.initialize()
        
        try:
            file_path = self._validate_path(path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            stat = file_path.stat()
            
            return {
                "path": path,
                "name": file_path.name,
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get file info for {path}: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown filesystem MCP"""
        logger.info("📴 Filesystem MCP shutdown complete")