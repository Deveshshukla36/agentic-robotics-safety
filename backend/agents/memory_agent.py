from collections import defaultdict, deque
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

class MemoryAgent:
    """Manages memory, context, and pattern detection"""
    
    def __init__(self, memory_size: int = 1000):
        self.scenarios = deque(maxlen=memory_size)
        self.safety_violations = deque(maxlen=memory_size)
        self.decisions = deque(maxlen=memory_size)
        self.patterns = {}
        self.similarity_threshold = 0.7
        
    def store_scenario(self, scenario: Dict) -> str:
        """Store a scenario with metadata"""
        scenario_id = hashlib.md5(
            json.dumps(scenario, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        memory_entry = {
            'id': scenario_id,
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'outcome': None
        }
        
        self.scenarios.append(memory_entry)
        return scenario_id
    
    def store_safety_incident(self, incident: Dict):
        """Store safety incident for learning"""
        self.safety_violations.append({
            'timestamp': datetime.now().isoformat(),
            'incident': incident
        })
    
    def store_decision(self, scenario_id: str, decision: Dict):
        """Store agent decision for analysis"""
        self.decisions.append({
            'scenario_id': scenario_id,
            'timestamp': datetime.now().isoformat(),
            'decision': decision
        })
    
    def find_similar_scenarios(self, current_scenario: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar past scenarios"""
        similarities = []
        
        for past in self.scenarios:
            sim = self._calculate_similarity(current_scenario, past['scenario'])
            if sim > self.similarity_threshold:
                similarities.append({
                    'scenario': past,
                    'similarity': sim
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _calculate_similarity(self, scenario1: Dict, scenario2: Dict) -> float:
        """Calculate similarity between two scenarios"""
        features1 = self._extract_features(scenario1)
        features2 = self._extract_features(scenario2)
        
        if not features1 or not features2:
            return 0.0
        
        dot_product = sum(f1 * f2 for f1, f2 in zip(features1, features2))
        norm1 = np.sqrt(sum(f**2 for f in features1))
        norm2 = np.sqrt(sum(f**2 for f in features2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _extract_features(self, scenario: Dict) -> List[float]:
        """Extract numerical features from scenario"""
        features = []
        
        robot = scenario.get('robot', {})
        features.append(robot.get('speed', 0) / 5.0)
        
        humans = scenario.get('environment', {}).get('humans', [])
        features.append(min(1.0, len(humans) / 10))
        
        if humans and 'position' in robot:
            robot_pos = np.array(robot['position'])
            min_dist = min(np.linalg.norm(robot_pos - np.array(h.get('position', [0,0]))) 
                          for h in humans)
            features.append(min(1.0, min_dist / 5.0))
        else:
            features.append(1.0)
        
        restricted_zones = scenario.get('environment', {}).get('restricted_zones', [])
        features.append(min(1.0, len(restricted_zones) / 5))
        
        hazard_zones = scenario.get('environment', {}).get('hazard_zones', [])
        features.append(min(1.0, len(hazard_zones) / 5))
        
        return features
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_scenarios': len(self.scenarios),
            'total_violations': len(self.safety_violations),
            'total_decisions': len(self.decisions),
            'patterns_detected': len(self.patterns),
            'memory_usage_percent': (len(self.scenarios) / self.scenarios.maxlen) * 100 if self.scenarios.maxlen > 0 else 0,
            'unique_patterns': list(self.patterns.keys())
        }
    
    def clear_memory(self):
        """Clear all memory (for testing)"""
        self.scenarios.clear()
        self.safety_violations.clear()
        self.decisions.clear()
        self.patterns.clear()