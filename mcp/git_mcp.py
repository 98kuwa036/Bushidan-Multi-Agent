"""
Bushidan Multi-Agent System v9.1 - Git MCP

Git operations MCP for version control management.
Provides safe git operations for Ashigaru execution layer.
"""

import asyncio
import logging
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path

from utils.logger import get_logger


logger = get_logger(__name__)


class GitMCP:
    """
    Git MCP - Version control operations
    
    Provides git operations for code management:
    - Status checking
    - File staging
    - Committing changes
    - Branch operations
    - History viewing
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize git MCP"""
        logger.info("📦 Initializing Git MCP...")
        
        # Check if we're in a git repository
        if not (self.repo_path / ".git").exists():
            logger.warning(f"⚠️ Not a git repository: {self.repo_path}")
            # Don't fail - just warn
        
        self.initialized = True
        logger.info(f"✅ Git MCP initialized - Repo: {self.repo_path}")
    
    async def _run_git_command(self, command: List[str]) -> Dict[str, Any]:
        """Run git command safely"""
        
        try:
            full_command = ["git"] + command
            
            result = subprocess.run(
                full_command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    async def status(self) -> str:
        """Get git status"""
        
        if not self.initialized:
            await self.initialize()
        
        result = await self._run_git_command(["status", "--porcelain"])
        
        if result["success"]:
            status_output = result["stdout"]
            if not status_output:
                return "Working tree clean"
            
            # Parse status for summary
            lines = status_output.split("\n")
            modified = len([l for l in lines if l.startswith(" M")])
            added = len([l for l in lines if l.startswith("A ")])
            deleted = len([l for l in lines if l.startswith(" D")])
            untracked = len([l for l in lines if l.startswith("??")])
            
            summary = f"Modified: {modified}, Added: {added}, Deleted: {deleted}, Untracked: {untracked}"
            logger.info(f"📦 Git status: {summary}")
            
            return f"{summary}\n\nDetails:\n{status_output}"
        else:
            logger.error(f"❌ Git status failed: {result['stderr']}")
            return f"Git status failed: {result['stderr']}"
    
    async def add(self, paths: List[str] = None) -> str:
        """Stage files for commit"""
        
        if not self.initialized:
            await self.initialize()
        
        if paths is None:
            command = ["add", "."]
        else:
            command = ["add"] + paths
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            logger.info(f"📦 Staged files: {paths or 'all changes'}")
            return "Files staged successfully"
        else:
            logger.error(f"❌ Git add failed: {result['stderr']}")
            return f"Git add failed: {result['stderr']}"
    
    async def commit(self, message: str, author: str = None) -> str:
        """Create git commit"""
        
        if not self.initialized:
            await self.initialize()
        
        if not message:
            return "Commit message is required"
        
        command = ["commit", "-m", message]
        if author:
            command.extend(["--author", author])
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            logger.info(f"📦 Created commit: {message[:50]}...")
            return f"Commit created: {result['stdout']}"
        else:
            logger.error(f"❌ Git commit failed: {result['stderr']}")
            return f"Git commit failed: {result['stderr']}"
    
    async def diff(self, staged: bool = False) -> str:
        """Show git diff"""
        
        if not self.initialized:
            await self.initialize()
        
        command = ["diff"]
        if staged:
            command.append("--cached")
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            diff_output = result["stdout"]
            if not diff_output:
                return "No changes to show"
            
            logger.info(f"📦 Git diff: {len(diff_output)} characters")
            return diff_output
        else:
            logger.error(f"❌ Git diff failed: {result['stderr']}")
            return f"Git diff failed: {result['stderr']}"
    
    async def log(self, max_count: int = 10) -> str:
        """Show git log"""
        
        if not self.initialized:
            await self.initialize()
        
        command = ["log", "--oneline", f"-{max_count}"]
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            logger.info(f"📦 Git log: {max_count} commits")
            return result["stdout"]
        else:
            logger.error(f"❌ Git log failed: {result['stderr']}")
            return f"Git log failed: {result['stderr']}"
    
    async def branch(self) -> str:
        """Show current branch and list branches"""
        
        if not self.initialized:
            await self.initialize()
        
        result = await self._run_git_command(["branch"])
        
        if result["success"]:
            logger.info("📦 Listed git branches")
            return result["stdout"]
        else:
            logger.error(f"❌ Git branch failed: {result['stderr']}")
            return f"Git branch failed: {result['stderr']}"
    
    async def push(self, remote: str = "origin", branch: str = None) -> str:
        """Push changes to remote"""
        
        if not self.initialized:
            await self.initialize()
        
        command = ["push", remote]
        if branch:
            command.append(branch)
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            logger.info(f"📦 Pushed to {remote}")
            return f"Pushed to {remote}: {result['stdout']}"
        else:
            logger.error(f"❌ Git push failed: {result['stderr']}")
            return f"Git push failed: {result['stderr']}"
    
    async def pull(self, remote: str = "origin", branch: str = None) -> str:
        """Pull changes from remote"""
        
        if not self.initialized:
            await self.initialize()
        
        command = ["pull", remote]
        if branch:
            command.append(branch)
        
        result = await self._run_git_command(command)
        
        if result["success"]:
            logger.info(f"📦 Pulled from {remote}")
            return f"Pulled from {remote}: {result['stdout']}"
        else:
            logger.error(f"❌ Git pull failed: {result['stderr']}")
            return f"Git pull failed: {result['stderr']}"
    
    async def is_clean(self) -> bool:
        """Check if working directory is clean"""
        
        result = await self._run_git_command(["status", "--porcelain"])
        return result["success"] and not result["stdout"]
    
    async def get_current_branch(self) -> str:
        """Get current branch name"""
        
        result = await self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        
        if result["success"]:
            return result["stdout"]
        else:
            return "unknown"
    
    async def shutdown(self) -> None:
        """Shutdown git MCP"""
        logger.info("📴 Git MCP shutdown complete")