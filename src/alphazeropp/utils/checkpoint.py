import pickle
import logging
from pathlib import Path
from typing import Any, Optional, List


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    AGENT_CHECKPOINT_FILE = "agent_checkpoint.pkl"
    NETWORK_CHECKPOINT_FILE = "network_checkpoint.pt"
    
    def __init__(self, checkpoint_dir: str | Path = "checkpoints"):
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Base directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def save_checkpoint(self, save_path: str | Path, agent: Any, net: Any = None) -> Path:
        """
        Save agent and network checkpoint to disk.
        
        Args:
            save_path: Directory path where checkpoint will be saved
            agent: Agent object with game, all_training_examples, and rngs attributes
            net: Network object (optional, will call net.save_checkpoint if provided)
        
        Returns:
            Path to saved checkpoint directory
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save agent state
            agent_file = save_path / self.AGENT_CHECKPOINT_FILE
            with agent_file.open("wb") as f:
                pickle.dump({
                    "game": agent.game,
                    "all_training_examples": agent.all_training_examples,
                    "rngs": agent.rngs
                }, f)
            logger.info(f"Agent checkpoint saved to {agent_file}")
            
            # Save network checkpoint if provided
            if net is not None:
                net.save_checkpoint(save_path)
                logger.info(f"Network checkpoint saved to {save_path}")
            
            logger.info(f"Full checkpoint saved to {save_path}")
            return save_path
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {save_path}: {e}")
            raise
    
    def load_checkpoint(self, load_path: str | Path, agent: Any, net: Any = None, 
                       exclude_keys: Optional[List[str]] = None) -> Path:
        """
        Load agent and network checkpoint from disk.
        
        Args:
            load_path: Directory path where checkpoint is stored
            agent: Agent object to load state into
            net: Network object (optional, will call net.load_checkpoint if provided)
            exclude_keys: List of keys to skip loading (e.g., ["rngs", "game"])
        
        Returns:
            Path to loaded checkpoint directory
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        load_path = Path(load_path)
        exclude_keys = exclude_keys or []
        
        try:
            # Load agent state
            agent_file = load_path / self.AGENT_CHECKPOINT_FILE
            if not agent_file.exists():
                raise FileNotFoundError(f"Agent checkpoint file not found: {agent_file}")
            
            with agent_file.open("rb") as f:
                checkpoint = pickle.load(f)
            
            # Load checkpoint data into agent
            if "rngs" not in exclude_keys:
                agent.rngs = checkpoint["rngs"]
            
            if "game" not in exclude_keys:
                agent.game = checkpoint["game"]
            
            if "all_training_examples" not in exclude_keys:
                # Clear and extend instead of reassign for better multiprocessing performance
                agent.all_training_examples.clear()
                agent.all_training_examples.extend(checkpoint["all_training_examples"])
            
            logger.info(f"Agent checkpoint loaded from {agent_file}")
            
            # Load network checkpoint if provided
            if net is not None:
                net.load_checkpoint(load_path)
                logger.info(f"Network checkpoint loaded from {load_path}")
            
            logger.info(f"Full checkpoint loaded from {load_path}")
            return load_path
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {load_path}: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """
        Validate that a checkpoint exists and contains required files.
        
        Args:
            checkpoint_path: Directory path to validate
        
        Returns:
            True if checkpoint is valid, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        agent_file = checkpoint_path / self.AGENT_CHECKPOINT_FILE
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint directory does not exist: {checkpoint_path}")
            return False
        
        if not agent_file.exists():
            logger.warning(f"Agent checkpoint file not found: {agent_file}")
            return False
        
        try:
            with agent_file.open("rb") as f:
                data = pickle.load(f)
            
            required_keys = {"game", "all_training_examples", "rngs"}
            if not required_keys.issubset(data.keys()):
                logger.warning(f"Checkpoint missing required keys. Expected {required_keys}, got {set(data.keys())}")
                return False
            
            logger.info(f"Checkpoint validation passed: {checkpoint_path}")
            return True
        
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def list_checkpoints(self, base_dir: Optional[str | Path] = None) -> List[Path]:
        """
        List all available checkpoints in the checkpoint directory.
        
        Args:
            base_dir: Optional base directory to search (defaults to checkpoint_dir)
        
        Returns:
            List of checkpoint directory paths
        """
        search_dir = Path(base_dir) if base_dir else self.checkpoint_dir
        
        if not search_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {search_dir}")
            return []
        
        # Find all directories containing agent checkpoint files
        checkpoints = []
        for subdir in search_dir.rglob("."):
            if (subdir / self.AGENT_CHECKPOINT_FILE).exists():
                checkpoints.append(subdir)
        
        checkpoints.sort(reverse=True)  # Most recent first
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """
        Delete a checkpoint directory.
        
        Args:
            checkpoint_path: Path to checkpoint directory to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
