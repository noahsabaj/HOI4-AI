# src/ai/ultimate/episodic_memory.py
"""
Neural Episodic Control (NEC) for HOI4
Fast learning by remembering what worked before
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from collections import deque
from typing import Optional, Tuple, List, Dict


class DifferentiableMemory(nn.Module):
    """
    Differentiable Neural Dictionary
    Core component of NEC - stores (key, value) pairs
    """

    def __init__(
            self,
            key_size: int = 128,
            memory_size: int = 50000,
            num_neighbors: int = 50,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.key_size = key_size
        self.memory_size = memory_size
        self.num_neighbors = num_neighbors
        self.device = device

        # Memory storage
        self.keys = torch.zeros(memory_size, key_size, device=device)
        self.values = torch.zeros(memory_size, device=device)
        self.ages = torch.zeros(memory_size, device=device)

        # Current size
        self.size = 0

        # FAISS index for fast k-NN search
        self.index = faiss.IndexFlatL2(key_size)

        # For GPU if available
        if device == 'cuda' and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Write new memory

        Args:
            key: Memory key (encoded state)
            value: Memory value (Q-value or return)
        """
        # Handle batch dimension
        if key.dim() == 2:
            key = key[0]
        if value.dim() == 1:
            value = value[0]

        # Find position to write
        if self.size < self.memory_size:
            idx = self.size
            self.size += 1
        else:
            # Replace oldest memory
            idx = torch.argmin(self.ages).item()

        # Write memory
        self.keys[idx] = key.detach()
        self.values[idx] = value.detach()
        self.ages[idx] = 0

        # Age all memories
        self.ages[:self.size] += 1

        # Update FAISS index
        key_np = key.detach().cpu().numpy().reshape(1, -1)
        if idx < self.index.ntotal:
            # Remove old key
            self.index.remove_ids(np.array([idx]))
        # Add new key
        self.index.add_with_ids(key_np, np.array([idx]))

    def lookup(
            self,
            query: torch.Tensor,
            return_distances: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Lookup k-nearest neighbors

        Args:
            query: Query key
            return_distances: Whether to return distances

        Returns:
            Weighted value based on k-NN
        """
        if self.size == 0:
            value = torch.zeros(1, device=self.device)
            if return_distances:
                return value, None
            return value

        # Handle batch dimension
        if query.dim() == 2:
            query = query[0]

        # Search k-NN
        k = min(self.num_neighbors, self.size)
        query_np = query.detach().cpu().numpy().reshape(1, -1)

        distances, indices = self.index.search(query_np, k)
        distances = distances[0]
        indices = indices[0]

        # Convert to torch
        distances = torch.tensor(distances, device=self.device)
        indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Compute weights (inverse distance weighting)
        weights = 1.0 / (distances + 1e-7)
        weights = weights / weights.sum()

        # Weighted average of values
        values = self.values[indices]
        weighted_value = (weights * values).sum()

        if return_distances:
            return weighted_value, distances
        return weighted_value

    def update_values(self, indices: torch.Tensor, new_values: torch.Tensor) -> None:
        """Update values for specific memories"""
        self.values[indices] = new_values


class NeuralEpisodicControl(nn.Module):
    """
    Complete NEC implementation for HOI4

    Features:
    - Fast episodic memory lookup
    - Learning from single experiences
    - Cross-episode transfer
    """

    def __init__(
            self,
            state_dim: int = 544,  # RSSM state size
            key_dim: int = 128,
            memory_size: int = 50000,
            num_neighbors: int = 50,
            learning_rate: float = 1e-3,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.device = device

        # Key encoder (learned representation)
        self.key_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, key_dim),
            nn.LayerNorm(key_dim)
        ).to(device)

        # Differentiable memory
        self.memory = DifferentiableMemory(
            key_size=key_dim,
            memory_size=memory_size,
            num_neighbors=num_neighbors,
            device=device
        )

        # Value network for bootstrapping
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.key_encoder.parameters()) +
            list(self.value_net.parameters()),
            lr=learning_rate
        )

        # Statistics
        self.write_count = 0
        self.lookup_count = 0

    def encode_key(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state into memory key"""
        return self.key_encoder(state)

    def write(self, state: torch.Tensor, value: torch.Tensor) -> None:
        """
        Store experience in memory

        Args:
            state: Current state (from RSSM)
            value: Value to store (return or Q-value)
        """
        key = self.encode_key(state)
        self.memory.write(key, value)
        self.write_count += 1

    def lookup(self, state: torch.Tensor) -> torch.Tensor:
        """
        Retrieve value for state

        Args:
            state: Query state

        Returns:
            Estimated value
        """
        key = self.encode_key(state)
        value = self.memory.lookup(key)
        self.lookup_count += 1
        return value

    def q_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-value combining memory and value network

        Args:
            state: Current state

        Returns:
            Q-value estimate
        """
        # Memory-based estimate
        memory_value = self.lookup(state)

        # Network-based estimate
        network_value = self.value_net(state).squeeze(-1)

        # Combine (you can tune this)
        # More weight on memory when we have good coverage
        memory_weight = min(0.9, self.write_count / 10000)
        combined_value = (
                memory_weight * memory_value +
                (1 - memory_weight) * network_value
        )

        return combined_value

    def update(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
            gamma: float = 0.99
    ) -> float:
        """
        Update NEC from batch of experiences

        Returns:
            Loss value
        """
        batch_size = states.shape[0]

        # Compute targets
        with torch.no_grad():
            next_values = self.q_value(next_states)
            targets = rewards + gamma * next_values * (1 - dones)

        # Current Q-values
        current_values = self.q_value(states)

        # Loss
        loss = F.mse_loss(current_values, targets)

        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.key_encoder.parameters()) +
            list(self.value_net.parameters()),
            max_norm=5.0
        )
        self.optimizer.step()

        # Write new experiences to memory
        for i in range(batch_size):
            self.write(states[i], targets[i])

        return loss.item()

    def get_statistics(self) -> Dict[str, float]:
        """Get memory statistics"""
        return {
            'memory_size': self.memory.size,
            'write_count': self.write_count,
            'lookup_count': self.lookup_count,
            'lookups_per_write': self.lookup_count / max(1, self.write_count)
        }


class PersistentEpisodicMemory:
    """
    Cross-game persistent memory using ChromaDB
    This is what enables "I remember from 3 games ago..."
    """

    def __init__(self, collection_name: str = "hoi4_experiences"):
        import chromadb

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./hoi4_persistent_memory")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "HOI4 cross-game experiences"}
        )

        # Track game sessions
        self.current_game_id = None
        self.experience_count = 0

    def start_new_game(self, game_id: str) -> None:
        """Start tracking a new game session"""
        self.current_game_id = game_id
        print(f"📝 Started new game session: {game_id}")

    def store_key_experience(
            self,
            state_encoding: np.ndarray,
            action: Dict,
            outcome: str,
            context: Dict
    ) -> None:
        """
        Store important experiences that should persist across games

        Args:
            state_encoding: Encoded state (from NEC key encoder)
            action: Action taken
            outcome: What happened
            context: Game context (date, factories, etc.)
        """
        if self.current_game_id is None:
            return

        # Create document
        doc = f"""
        Game: {self.current_game_id}
        Date: {context.get('game_date', 'Unknown')}
        Factories: Civ={context.get('civilian_factories', 0)}, Mil={context.get('military_factories', 0)}
        Political Power: {context.get('political_power', 0)}
        Action: {action.get('type', 'unknown')} at {action.get('x', 0)}, {action.get('y', 0)}
        Outcome: {outcome}
        """

        # Store with embedding
        self.collection.add(
            embeddings=[state_encoding.tolist()],
            documents=[doc],
            metadatas=[{
                'game_id': self.current_game_id,
                'timestamp': str(np.datetime64('now')),
                'success': 'good' in outcome.lower() or 'success' in outcome.lower(),
                **context
            }],
            ids=[f"{self.current_game_id}_{self.experience_count}"]
        )

        self.experience_count += 1

    def recall_similar_experiences(
            self,
            state_encoding: np.ndarray,
            n_results: int = 5
    ) -> List[Dict]:
        """
        Recall similar experiences from past games

        Args:
            state_encoding: Current state encoding
            n_results: Number of similar experiences to retrieve

        Returns:
            List of similar past experiences
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[state_encoding.tolist()],
            n_results=min(n_results, self.collection.count())
        )

        experiences = []
        for i in range(len(results['documents'][0])):
            experiences.append({
                'description': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return experiences

    def get_successful_strategies(self, context_filter: Dict = None) -> List[Dict]:
        """
        Get strategies that led to success

        Args:
            context_filter: Filter by context (e.g., {'year': 1936})

        Returns:
            List of successful strategies
        """
        where_clause = {'success': True}
        if context_filter:
            where_clause.update(context_filter)

        results = self.collection.get(
            where=where_clause,
            limit=100
        )

        strategies = []
        for i in range(len(results['documents'])):
            strategies.append({
                'description': results['documents'][i],
                'metadata': results['metadatas'][i]
            })

        return strategies