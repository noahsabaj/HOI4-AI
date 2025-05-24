# src/ai/ultimate/hf_strategic_evaluator.py
"""
Hugging Face based strategic evaluation for HOI4 AI
Uses fast, local models for real-time strategic feedback
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from datetime import datetime
import re


class HuggingFaceStrategicEvaluator:
    """
    Fast strategic evaluation using Hugging Face models
    Provides real-time feedback for HOI4 gameplay

    Optimized for Phi-4-mini-reasoning but compatible with other models
    """

    def __init__(
            self,
            model_name: str = "microsoft/phi-4-mini-reasoning",
            device: str = None,
            evaluation_frequency: int = 30
    ):
        self.evaluation_frequency = evaluation_frequency
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🤗 Loading Hugging Face model: {model_name}")
        print(f"   Device: {self.device}")

        # Add special handling for Phi-4 models
        if "phi-4" in model_name.lower():
            print(f"   📊 Using Phi-4 reasoning model for strategic evaluation")

        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Model loaded successfully!")
            self.model_available = True

        except Exception as e:
            print(f"⚠️ Failed to load HF model: {e}")
            print("   Falling back to rule-based evaluation only")
            self.model_available = False

        # History tracking
        self.action_history = deque(maxlen=50)
        self.state_history = deque(maxlen=20)
        self.decision_history = deque(maxlen=100)
        self.strategic_insights = deque(maxlen=20)

        # Performance
        self.evaluation_count = 0
        self.total_reward_given = 0.0

    def record_action(self, action: Dict, game_state: Dict) -> Optional[float]:
        """Record action and evaluate if needed"""
        # Record the action
        self.action_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action.get('description', 'Unknown'),
            'type': action.get('type', 'unknown'),
            'screen': game_state.get('screen_type', 'unknown'),
            'game_date': game_state.get('game_date', 'Unknown'),
        })

        # Track significant decisions
        if self._is_significant_action(action):
            self.decision_history.append({
                'action': action['description'],
                'date': game_state.get('game_date', 'Unknown'),
                'context': game_state.get('screen_type', 'unknown')
            })

        # Update state history
        if self._state_changed_significantly(game_state):
            self.state_history.append(self._extract_key_state(game_state))

        # Evaluate if it's time
        self.evaluation_count += 1
        if self.evaluation_count % self.evaluation_frequency == 0:
            return self._evaluate_strategic_quality()

        return None

    def _evaluate_strategic_quality(self) -> float:
        """Evaluate gameplay using HF model or rules"""
        if len(self.action_history) < 10:
            return 0.0

        context = self._prepare_context()

        # Try model evaluation first
        if self.model_available and self.evaluation_count % (self.evaluation_frequency * 3) == 0:
            # Use model every 3rd evaluation to balance speed/quality
            reward, reasoning, insight = self._model_evaluation(context)
        else:
            # Use fast rules most of the time
            reward, reasoning, insight = self._rule_evaluation(context)

        # Store insights
        if insight:
            self.strategic_insights.append({
                'timestamp': datetime.now().isoformat(),
                'insight': insight,
                'reward': reward
            })

        self.total_reward_given += reward

        # Log significant evaluations
        if abs(reward) >= 2.0:
            self._log_evaluation(reward, reasoning, insight)

        return reward

    def _model_evaluation(self, context: str) -> Tuple[float, str, str]:
        """Use HF model for strategic evaluation"""
        # Create a focused prompt
        prompt = f"""You are evaluating Hearts of Iron 4 gameplay for Germany.

Current situation:
{context[:400]}

Good strategy phases:
- 1936-1937: Build civilian factories (economic focus)
- 1938-1939: Transition to military production
- 1939+: Full war preparation

Based on the current situation, rate the gameplay from -5 (terrible) to +5 (excellent).
Consider: Is the player in the right phase? Are they making progress?

Rating:"""

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # Parse response
            return self._parse_model_response(response, context)

        except Exception as e:
            print(f"⚠️ Model evaluation error: {e}")
            return self._rule_evaluation(context)

    def _parse_model_response(self, response: str, context: str) -> Tuple[float, str, str]:
        """Parse model response for score and insights"""
        # Look for number in response
        numbers = re.findall(r'[+-]?\d+\.?\d*', response)
        score = 0.0

        for num_str in numbers:
            try:
                num = float(num_str)
                if -5 <= num <= 5:
                    score = num
                    break
            except:
                continue

        # Generate insight based on response and score
        if score > 2:
            insight = "Good progress - continue current strategy"
        elif score < -2:
            insight = "Need to change approach - try construction menu (F1)"
        else:
            insight = "Explore production and research menus"

        reasoning = f"Model evaluation: {response[:100]}..."
        return score, reasoning, insight

    def _rule_evaluation(self, context: str) -> Tuple[float, str, str]:
        """Fast rule-based evaluation"""
        reward = 0.0
        reasons = []

        # Get current state
        current_state = self.state_history[-1] if self.state_history else {}
        date = current_state.get('date', 'Unknown')
        civ_factories = current_state.get('civ_factories', 0)
        mil_factories = current_state.get('mil_factories', 0)
        screen = current_state.get('screen', 'unknown')

        # Detect year
        year = None
        for y in range(1936, 1946):
            if str(y) in str(date):
                year = y
                break

        # Phase-based scoring
        if year:
            if year <= 1937:
                # Economic phase
                if civ_factories < 50:
                    if 'construction' in screen:
                        reward += 1.0
                        reasons.append("Good: In construction during eco phase")
                    if self._recent_factory_built('civilian'):
                        reward += 2.0
                        reasons.append("Excellent: Building civilian factories")
                    insight = "Build civilian factories to 50+"
                else:
                    insight = "Good civilian base - prepare for transition"

            elif year == 1938:
                # Transition phase
                if mil_factories < 20:
                    if 'production' in screen:
                        reward += 1.0
                        reasons.append("Good: Setting up military production")
                    insight = "Start military factory construction"
                else:
                    insight = "Set up equipment production lines"

            else:  # 1939+
                # War phase
                if mil_factories < 40:
                    reward -= 1.0
                    reasons.append("Warning: Low military capacity for war")
                insight = "Maximize military production"
        else:
            # Can't detect date
            if screen in ['construction', 'production', 'research']:
                reward += 0.5
                reasons.append(f"Good: Using {screen} screen")
            insight = "Explore game menus using F1-F5 keys"

        # Check for stuck behavior
        if self._is_stuck():
            reward -= 2.0
            reasons.append("Bad: Repetitive actions without progress")

        # Progress check
        if len(self.decision_history) > 2:
            reward += 0.5
            reasons.append("Making strategic decisions")

        reasoning = "Rules: " + "; ".join(reasons) if reasons else "Neutral"
        return reward, reasoning, insight

    def _recent_factory_built(self, factory_type: str) -> bool:
        """Check if a factory was recently built"""
        for decision in list(self.decision_history)[-5:]:
            if factory_type in decision.get('action', '').lower():
                return True
        return False

    def _is_stuck(self) -> bool:
        """Detect if AI is stuck in repetitive behavior"""
        if len(self.action_history) < 20:
            return False

        recent_actions = [a['type'] for a in list(self.action_history)[-20:]]
        unique_actions = len(set(recent_actions))

        return unique_actions < 3  # Very repetitive

    def _prepare_context(self) -> str:
        """Prepare context summary"""
        current_state = self.state_history[-1] if self.state_history else {}

        # Recent actions summary
        action_types = {}
        for act in list(self.action_history)[-20:]:
            action_types[act['type']] = action_types.get(act['type'], 0) + 1

        context = f"""Date: {current_state.get('date', 'Unknown')}
Factories: Civ={current_state.get('civ_factories', 0)}, Mil={current_state.get('mil_factories', 0)}
Screen: {current_state.get('screen', 'unknown')}
Recent actions: {dict(action_types)}
Decisions made: {len(self.decision_history)}"""

        return context

    def _is_significant_action(self, action: Dict) -> bool:
        """Check if action is strategically significant"""
        desc = action.get('description', '').lower()
        significant_words = ['build', 'construct', 'research', 'focus', 'produce', 'factory']
        return any(word in desc for word in significant_words)

    def _state_changed_significantly(self, new_state: Dict) -> bool:
        """Check if state changed significantly"""
        if not self.state_history:
            return True

        last = self.state_history[-1]

        # Check key changes
        if (new_state.get('civilian_factories', 0) != last.get('civ_factories', 0) or
                new_state.get('military_factories', 0) != last.get('mil_factories', 0) or
                new_state.get('screen_type') != last.get('screen')):
            return True

        return False

    def _extract_key_state(self, game_state: Dict) -> Dict:
        """Extract key state information"""
        return {
            'date': game_state.get('game_date', 'Unknown'),
            'civ_factories': game_state.get('civilian_factories', 0),
            'mil_factories': game_state.get('military_factories', 0),
            'screen': game_state.get('screen_type', 'unknown'),
        }

    def _log_evaluation(self, reward: float, reasoning: str, insight: str):
        """Log evaluation results"""
        print(f"\n{'=' * 50}")
        print(f"🤗 STRATEGIC EVALUATION #{self.evaluation_count // self.evaluation_frequency}")
        print(f"Reward: {reward:+.2f}")
        print(f"Reasoning: {reasoning}")
        print(f"💡 Advice: {insight}")
        print(f"{'=' * 50}\n")

    def get_current_strategy(self) -> str:
        """Get current strategic phase"""
        if not self.state_history:
            return "Exploring"

        date = self.state_history[-1].get('date', 'Unknown')

        if 'Unknown' in date:
            return "Detecting game state..."
        elif any(y in str(date) for y in ['1936', '1937']):
            return "Economic Buildup"
        elif '1938' in str(date):
            return "Military Transition"
        elif any(y in str(date) for y in ['1939', '1940', '1941']):
            return "War Preparation"
        else:
            return "Strategic Planning"

    def get_strategic_summary(self) -> Dict:
        """Get summary of strategic performance"""
        avg_reward = self.total_reward_given / max(1, self.evaluation_count // self.evaluation_frequency)

        return {
            'evaluations': self.evaluation_count // self.evaluation_frequency,
            'average_reward': avg_reward,
            'current_strategy': self.get_current_strategy(),
            'total_decisions': len(self.decision_history),
            'recent_insights': list(self.strategic_insights)[-5:],
        }