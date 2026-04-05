"""
RollbackManager: transaction-like state management with rollback capabilities.
"""

import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from ...base import ErrorClassification, PipelineError
from ....logging_manager import get_logger
from ._types import PipelineCheckpoint

logger = get_logger(__name__)


class RollbackManager:
    """
    Transaction-like state management with rollback capabilities.

    Provides checkpointing, state restoration, and resource cleanup
    for complex multi-step conversion operations.
    """

    def __init__(
        self,
        max_checkpoints: int = 10,
        enable_disk_persistence: bool = False,
        checkpoint_compression: bool = True,
        checkpoint_storage_path: Optional[str] = None,
        cleanup_on_success: bool = True,
    ):
        """
        Initialize RollbackManager.

        Args:
            max_checkpoints: Maximum number of checkpoints to maintain
            enable_disk_persistence: Store checkpoints on disk for recovery
            checkpoint_compression: Compress checkpoint data to save space
            checkpoint_storage_path: Path for checkpoint storage (None for temp)
            cleanup_on_success: Whether to cleanup checkpoints on successful completion
        """
        self.max_checkpoints = max_checkpoints
        self.enable_disk_persistence = enable_disk_persistence
        self.checkpoint_compression = checkpoint_compression
        self.checkpoint_storage_path = checkpoint_storage_path or tempfile.mkdtemp(
            prefix="localdata_checkpoints_"
        )
        self.cleanup_on_success = cleanup_on_success

        # Ensure storage directory exists
        os.makedirs(self.checkpoint_storage_path, exist_ok=True)

        # Checkpoint tracking
        self.checkpoints: Dict[str, List[PipelineCheckpoint]] = defaultdict(list)
        self.active_transactions: Set[str] = set()

        # Statistics
        self.stats = {
            "checkpoints_created": 0,
            "rollbacks_performed": 0,
            "successful_cleanups": 0,
            "failed_cleanups": 0,
        }

        self._lock = threading.RLock()

        logger.info(
            "RollbackManager initialized",
            storage_path=self.checkpoint_storage_path,
            max_checkpoints=max_checkpoints,
        )

    def begin_transaction(self, pipeline_id: str) -> str:
        """
        Begin a new transaction for pipeline execution.

        Args:
            pipeline_id: Identifier for the pipeline

        Returns:
            Transaction ID
        """
        transaction_id = f"{pipeline_id}_{int(time.time() * 1000)}"

        with self._lock:
            self.active_transactions.add(transaction_id)

        logger.info(
            "Transaction begun", transaction_id=transaction_id, pipeline_id=pipeline_id
        )

        return transaction_id

    def create_checkpoint(
        self,
        transaction_id: str,
        pipeline_id: str,
        step_index: int,
        step_id: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineCheckpoint:
        """
        Create a checkpoint for pipeline state.

        Args:
            transaction_id: Transaction identifier
            pipeline_id: Pipeline identifier
            step_index: Index of current step
            step_id: Identifier of current step
            data: Data state to checkpoint
            metadata: Additional metadata
            execution_context: Execution context to save

        Returns:
            Created checkpoint
        """
        checkpoint = PipelineCheckpoint(
            pipeline_id=pipeline_id,
            step_index=step_index,
            step_id=step_id,
            metadata=metadata or {},
            execution_context=execution_context or {},
        )

        try:
            # Serialize and store data
            if data is not None:
                checkpoint.data_hash = str(hash(str(data)))
                storage_path = os.path.join(
                    self.checkpoint_storage_path,
                    f"checkpoint_{checkpoint.checkpoint_id}.pkl",
                )

                with open(storage_path, "wb") as f:
                    pickle.dump(data, f)

                checkpoint.storage_path = storage_path
                checkpoint.data_snapshot = None  # Don't keep in memory

            # Add to checkpoint history
            with self._lock:
                self.checkpoints[transaction_id].append(checkpoint)
                self.stats["checkpoints_created"] += 1

                # Maintain checkpoint limit
                if len(self.checkpoints[transaction_id]) > self.max_checkpoints:
                    old_checkpoint = self.checkpoints[transaction_id].pop(0)
                    self._cleanup_checkpoint_storage(old_checkpoint)

            logger.debug(
                "Checkpoint created",
                transaction_id=transaction_id,
                checkpoint_id=checkpoint.checkpoint_id,
                step_id=step_id,
            )

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise PipelineError(
                f"Checkpoint creation failed: {str(e)}",
                ErrorClassification.CONFIGURATION_ERROR,
                "checkpoint_creation",
            )

    def rollback_to_checkpoint(
        self, transaction_id: str, checkpoint_id: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Rollback to a specific checkpoint or the latest one.

        Args:
            transaction_id: Transaction to rollback
            checkpoint_id: Specific checkpoint ID (None for latest)

        Returns:
            Tuple of (restored_data, execution_context)
        """
        with self._lock:
            if transaction_id not in self.checkpoints:
                raise ValueError(
                    f"No checkpoints found for transaction: {transaction_id}"
                )

            transaction_checkpoints = self.checkpoints[transaction_id]
            if not transaction_checkpoints:
                raise ValueError(
                    f"No checkpoints available for transaction: {transaction_id}"
                )

            # Find target checkpoint
            if checkpoint_id:
                target_checkpoint = None
                for cp in transaction_checkpoints:
                    if cp.checkpoint_id == checkpoint_id:
                        target_checkpoint = cp
                        break

                if not target_checkpoint:
                    raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            else:
                target_checkpoint = transaction_checkpoints[-1]  # Latest checkpoint

            self.stats["rollbacks_performed"] += 1

        try:
            # Restore data from checkpoint
            restored_data = None
            if target_checkpoint.storage_path and os.path.exists(
                target_checkpoint.storage_path
            ):
                with open(target_checkpoint.storage_path, "rb") as f:
                    restored_data = pickle.load(f)
            elif target_checkpoint.data_snapshot is not None:
                restored_data = target_checkpoint.data_snapshot

            # Cleanup checkpoints after rollback point
            self._cleanup_checkpoints_after(transaction_id, target_checkpoint)

            logger.info(
                "Rollback completed",
                transaction_id=transaction_id,
                checkpoint_id=target_checkpoint.checkpoint_id,
                step_id=target_checkpoint.step_id,
            )

            return restored_data, target_checkpoint.execution_context

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise PipelineError(
                f"Rollback failed: {str(e)}",
                ErrorClassification.EXTERNAL_DEPENDENCY_ERROR,
                "rollback_operation",
            )

    def commit_transaction(self, transaction_id: str) -> None:
        """
        Commit transaction and optionally cleanup checkpoints.

        Args:
            transaction_id: Transaction to commit
        """
        with self._lock:
            if transaction_id not in self.active_transactions:
                logger.warning(f"Transaction not active: {transaction_id}")
                return

            self.active_transactions.remove(transaction_id)

        # Cleanup checkpoints if configured
        if self.cleanup_on_success:
            self._cleanup_transaction_checkpoints(transaction_id)

        logger.info(
            "Transaction committed",
            transaction_id=transaction_id,
            cleanup_performed=self.cleanup_on_success,
        )

    def abort_transaction(self, transaction_id: str) -> None:
        """
        Abort transaction and cleanup all checkpoints.

        Args:
            transaction_id: Transaction to abort
        """
        with self._lock:
            if transaction_id in self.active_transactions:
                self.active_transactions.remove(transaction_id)

        # Always cleanup on abort
        self._cleanup_transaction_checkpoints(transaction_id)

        logger.info("Transaction aborted", transaction_id=transaction_id)

    def get_checkpoint_history(self, transaction_id: str) -> List[PipelineCheckpoint]:
        """Get checkpoint history for a transaction."""
        with self._lock:
            return self.checkpoints.get(transaction_id, []).copy()

    def _cleanup_checkpoints_after(
        self, transaction_id: str, target_checkpoint: PipelineCheckpoint
    ) -> None:
        """Cleanup checkpoints created after the target checkpoint."""
        with self._lock:
            if transaction_id not in self.checkpoints:
                return

            checkpoints = self.checkpoints[transaction_id]
            target_index = -1

            for i, cp in enumerate(checkpoints):
                if cp.checkpoint_id == target_checkpoint.checkpoint_id:
                    target_index = i
                    break

            if target_index >= 0:
                # Cleanup checkpoints after target
                checkpoints_to_cleanup = checkpoints[target_index + 1 :]
                self.checkpoints[transaction_id] = checkpoints[: target_index + 1]

                for cp in checkpoints_to_cleanup:
                    self._cleanup_checkpoint_storage(cp)

    def _cleanup_transaction_checkpoints(self, transaction_id: str) -> None:
        """Cleanup all checkpoints for a transaction."""
        with self._lock:
            checkpoints = self.checkpoints.get(transaction_id, [])
            if transaction_id in self.checkpoints:
                del self.checkpoints[transaction_id]

        # Cleanup storage for all checkpoints
        for checkpoint in checkpoints:
            self._cleanup_checkpoint_storage(checkpoint)

        logger.debug(
            f"Cleaned up {len(checkpoints)} checkpoints for transaction {transaction_id}"
        )

    def _cleanup_checkpoint_storage(self, checkpoint: PipelineCheckpoint) -> None:
        """Cleanup storage for a specific checkpoint."""
        try:
            # Remove storage file
            if checkpoint.storage_path and os.path.exists(checkpoint.storage_path):
                os.remove(checkpoint.storage_path)

            # Remove temporary files
            for temp_file in checkpoint.temporary_files:
                if os.path.exists(temp_file):
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file)
                    else:
                        os.remove(temp_file)

            with self._lock:
                self.stats["successful_cleanups"] += 1

        except Exception as e:
            with self._lock:
                self.stats["failed_cleanups"] += 1
            logger.warning(
                f"Checkpoint cleanup failed: {e}",
                checkpoint_id=checkpoint.checkpoint_id,
            )

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback manager statistics."""
        with self._lock:
            return {
                **self.stats,
                "active_transactions": len(self.active_transactions),
                "total_checkpoints": sum(
                    len(checkpoints) for checkpoints in self.checkpoints.values()
                ),
                "storage_path": self.checkpoint_storage_path,
                "storage_size_mb": self._calculate_storage_size(),
            }

    def _calculate_storage_size(self) -> float:
        """Calculate total storage size in MB."""
        total_size = 0
        try:
            for root, dirs, files in os.walk(self.checkpoint_storage_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
        except Exception:
            pass

        return total_size / (1024 * 1024)

    def cleanup_all_checkpoints(self) -> None:
        """Cleanup all checkpoints and reset state."""
        with self._lock:
            all_checkpoints = []
            for checkpoints in self.checkpoints.values():
                all_checkpoints.extend(checkpoints)

            self.checkpoints.clear()
            self.active_transactions.clear()

        # Cleanup storage for all checkpoints
        for checkpoint in all_checkpoints:
            self._cleanup_checkpoint_storage(checkpoint)

        logger.info(f"Cleaned up all {len(all_checkpoints)} checkpoints")
