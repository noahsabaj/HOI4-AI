# src/core/memory.py - Long-term strategic memory
import pickle
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime


@dataclass
class GameMemory:
    """Complete memory of a single game"""
    game_id: str
    start_date: datetime
    end_date: datetime
    final_outcome: str  # 'victory', 'defeat', 'ongoing'

    # Key metrics
    peak_factories: Dict[str, int] = field(default_factory=dict)
    territories_gained: List[str] = field(default_factory=list)
    key_decisions: List[Dict] = field(default_factory=list)

    # Strategic trajectory
    factory_curve: List[Tuple[int, int]] = field(default_factory=list)  # (month, count)
    territory_curve: List[Tuple[int, List[str]]] = field(default_factory=list)

    # What worked
    successful_patterns: List[Dict] = field(default_factory=list)

    # Final score
    victory_score: float = 0.0


class StrategicMemory:
    """
    Remembers what leads to victory across many games.
    No hard-coding - discovers patterns through experience.
    """

    def __init__(self, memory_dir: str = "models/memory"):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        # Different types of memory
        self.episodic_memory = deque(maxlen=1000)  # Complete games
        self.semantic_memory = {}  # Learned facts about the game
        self.procedural_memory = {}  # Discovered action sequences

        # Pattern discovery
        self.winning_patterns = {}
        self.losing_patterns = {}

        # Load existing memories
        self.load_memories()

    def remember_game(self, game_memory: GameMemory):
        """Store a complete game memory"""
        self.episodic_memory.append(game_memory)

        # Extract patterns from this game
        if game_memory.final_outcome == 'victory':
            self._extract_winning_patterns(game_memory)
        else:
            self._extract_losing_patterns(game_memory)

        # Update semantic knowledge
        self._update_semantic_memory(game_memory)

        # Save to disk
        self.save_memories()

    def _extract_winning_patterns(self, game: GameMemory):
        """Extract patterns that led to victory"""
        # When did factory building accelerate?
        factory_growth_phases = self._analyze_growth_curve(game.factory_curve)

        # What decisions preceded major gains?
        for i, decision in enumerate(game.key_decisions):
            # Look at outcomes in next few decisions
            outcomes = game.key_decisions[i:i + 5]
            success_rate = sum(1 for o in outcomes if o.get('successful', False)) / len(outcomes)

            if success_rate > 0.7:
                pattern_key = f"{decision['type']}_{decision['context']}"
                if pattern_key not in self.winning_patterns:
                    self.winning_patterns[pattern_key] = {
                        'count': 0,
                        'avg_success': 0.0,
                        'examples': []
                    }

                pattern = self.winning_patterns[pattern_key]
                pattern['count'] += 1
                pattern['avg_success'] = (
                        (pattern['avg_success'] * (pattern['count'] - 1) + success_rate) /
                        pattern['count']
                )
                pattern['examples'].append(decision)

    def _analyze_growth_curve(self, curve: List[Tuple[int, int]]) -> List[Dict]:
        """Analyze when and how growth accelerated"""
        phases = []

        for i in range(1, len(curve)):
            prev_month, prev_count = curve[i - 1]
            curr_month, curr_count = curve[i]

            growth_rate = (curr_count - prev_count) / max(prev_count, 1)

            if growth_rate > 0.1:  # 10% growth
                phases.append({
                    'month': curr_month,
                    'growth_rate': growth_rate,
                    'from_count': prev_count,
                    'to_count': curr_count
                })

        return phases

    def _update_semantic_memory(self, game: GameMemory):
        """Update general knowledge about the game"""
        # Average factory counts at different stages
        for month, count in game.factory_curve:
            year = 1936 + month // 12

            if year not in self.semantic_memory:
                self.semantic_memory[year] = {
                    'avg_civilian_factories': 0,
                    'avg_military_factories': 0,
                    'games_count': 0
                }

            # Update running average
            year_stats = self.semantic_memory[year]
            year_stats['games_count'] += 1

            # This is simplified - in reality you'd track both types
            year_stats['avg_civilian_factories'] = (
                    (year_stats['avg_civilian_factories'] * (year_stats['games_count'] - 1) + count) /
                    year_stats['games_count']
            )

    def recall_best_action(self, current_context: Dict) -> Dict:
        """Recall the best action for current situation"""
        # Look for similar contexts in winning games
        best_match = None
        best_score = -1

        for game in self.episodic_memory:
            if game.final_outcome != 'victory':
                continue

            for decision in game.key_decisions:
                similarity = self._context_similarity(current_context, decision['context'])

                if similarity > best_score:
                    best_score = similarity
                    best_match = decision

        return best_match if best_match else {}

    def _context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Measure how similar two game contexts are"""
        score = 0.0

        # Similar year/month
        if abs(context1.get('year', 1936) - context2.get('year', 1936)) < 1:
            score += 0.3

        # Similar factory counts
        factory_diff = abs(
            context1.get('factories', 0) - context2.get('factories', 0)
        )
        if factory_diff < 10:
            score += 0.3

        # Similar strategic phase
        if context1.get('phase') == context2.get('phase'):
            score += 0.4

        return score

    def get_discovery_insights(self) -> Dict:
        """Get insights from discovered patterns"""
        insights = {
            'total_games': len(self.episodic_memory),
            'victories': sum(1 for g in self.episodic_memory if g.final_outcome == 'victory'),
            'top_winning_patterns': [],
            'top_losing_patterns': [],
            'optimal_factory_targets': {}
        }

        # Find most successful patterns
        winning_sorted = sorted(
            self.winning_patterns.items(),
            key=lambda x: x[1]['avg_success'] * x[1]['count'],
            reverse=True
        )[:10]

        insights['top_winning_patterns'] = [
            {
                'pattern': k,
                'success_rate': v['avg_success'],
                'occurrences': v['count']
            }
            for k, v in winning_sorted
        ]

        # Extract optimal factory targets by year
        for year, stats in self.semantic_memory.items():
            if stats['games_count'] > 5:  # Need enough data
                insights['optimal_factory_targets'][year] = {
                    'civilian': int(stats['avg_civilian_factories']),
                    'based_on_games': stats['games_count']
                }

        return insights

    def save_memories(self):
        """Persist memories to disk"""
        # Save episodic memory
        with open(os.path.join(self.memory_dir, 'episodic.pkl'), 'wb') as f:
            pickle.dump(list(self.episodic_memory), f)

        # Save semantic memory
        with open(os.path.join(self.memory_dir, 'semantic.pkl'), 'wb') as f:
            pickle.dump(self.semantic_memory, f)

        # Save discovered patterns
        with open(os.path.join(self.memory_dir, 'patterns.pkl'), 'wb') as f:
            pickle.dump({
                'winning': self.winning_patterns,
                'losing': self.losing_patterns
            }, f)

    def load_memories(self):
        """Load memories from disk"""
        try:
            # Load episodic
            episodic_path = os.path.join(self.memory_dir, 'episodic.pkl')
            if os.path.exists(episodic_path):
                with open(episodic_path, 'rb') as f:
                    games = pickle.load(f)
                    self.episodic_memory.extend(games)

            # Load semantic
            semantic_path = os.path.join(self.memory_dir, 'semantic.pkl')
            if os.path.exists(semantic_path):
                with open(semantic_path, 'rb') as f:
                    self.semantic_memory = pickle.load(f)

            # Load patterns
            patterns_path = os.path.join(self.memory_dir, 'patterns.pkl')
            if os.path.exists(patterns_path):
                with open(patterns_path, 'rb') as f:
                    patterns = pickle.load(f)
                    self.winning_patterns = patterns['winning']
                    self.losing_patterns = patterns['losing']

            print(f"📚 Loaded memories from {len(self.episodic_memory)} games")

        except Exception as e:
            print(f"⚠️ Could not load memories: {e}")
            print("Starting with fresh memory")