"""Episode saver for human intervention data collection.

Handles saving trajectories to appropriate folders based on success and intervention status.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class EpisodeSaver:
    """Handles saving trajectories to categorized folders."""

    def __init__(self, output_dir: str):
        """Initialize the episode saver.

        Creates three subdirectories:
        - rejection_sample/: Successful autonomous episodes
        - human_intervention/: Episodes with human intervention
        - failed_autonomous/: Failed autonomous episodes

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.rejection_dir = self.output_dir / "rejection_sample"
        self.intervention_dir = self.output_dir / "human_intervention"
        self.failed_dir = self.output_dir / "failed_autonomous"

        # Create directories
        self.rejection_dir.mkdir(parents=True, exist_ok=True)
        self.intervention_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: Dict,
        images: Optional[np.ndarray],
        env_seed: int,
        trial_idx: int,
        success: bool,
        had_intervention: bool,
        save_images: bool = True,
    ) -> Path:
        """Save trajectory to appropriate folder.

        Folder selection logic:
        - Had intervention -> human_intervention/
        - No intervention + success -> rejection_sample/
        - No intervention + fail -> failed_autonomous/

        Args:
            data: Trajectory data dict from TrajectoryRecorder.finalize()
            images: Optional image array (T, H, W, 3)
            env_seed: Environment seed
            trial_idx: Trial index
            success: Whether episode succeeded
            had_intervention: Whether human intervened
            save_images: Whether to save images

        Returns:
            Path to saved state file
        """
        # Determine folder
        if had_intervention:
            folder = self.intervention_dir
        elif success:
            folder = self.rejection_dir
        else:
            folder = self.failed_dir

        # Generate filename with timestamp for uniqueness
        uid = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"env_seed_{env_seed}_trial_{trial_idx}_{uid}"

        # Save state data
        data_to_save = dict(data)
        state_path = folder / f"{base_name}.npz"
        np.savez(state_path, **data_to_save)

        # Save images if requested
        if save_images and images is not None:
            image_path = folder / f"{base_name}_images.npz"
            np.savez_compressed(image_path, images=images)

        return state_path

    def get_counts(self) -> Dict[str, int]:
        """Get counts of saved files in each folder.

        Returns:
            Dict with counts for each folder type
        """
        return {
            "rejection_sample": len(list(self.rejection_dir.glob("*.npz"))) // 2,  # Divide by 2 for images
            "human_intervention": len(list(self.intervention_dir.glob("*.npz"))) // 2,
            "failed_autonomous": len(list(self.failed_dir.glob("*.npz"))) // 2,
        }
