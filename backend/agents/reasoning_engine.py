"""
Local reasoning engine using lightweight NLP + rule-based reasoning
No external APIs - fully local
"""

import re
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import hashlib

class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    CRITICAL = "critical"

@dataclass
class ReasoningTrace:
    step: int
    observation: str
    reasoning: str
    action: str
    confidence: float
    
class LocalReasoningEngine:
    """Local reasoning without external APIs"""
    
    def __init__(self):
        self.safety_rules = self._load_safety_rules()
        self.context_memory = deque(maxlen=100)
        self.reasoning_traces = []
        
    def _load_safety_rules(self) -> Dict:
        return {
            "distance_rules": {
                "critical_distance": 0.5,
                "warning_distance": 1.0,
                "safe_distance": 2.0
            },
            "speed_rules": {
                "max_speed": 2.0,
                "speed_buffer": 0.3
            },
            "zone_rules": {
                "restricted_penalty": 10,
                "hazard_penalty": 20
            }
        }
    
    def reason_about_safety(self, robot_state: Dict, environment: Dict, tool_outputs: Dict) -> Dict:
        """Main reasoning entry point"""
        
        trace = []
        
        # Step 1: Observe current state
        observation = self._observe_state(robot_state, environment)
        trace.append(ReasoningTrace(1, observation, "Analyzing initial state", "Observe", 0.95))
        
        # Step 2: Apply safety rules
        violations = self._check_safety_violations(robot_state, environment)
        trace.append(ReasoningTrace(2, f"Found {len(violations)} violations", 
                                   self._explain_violations(violations), 
                                   "Evaluate", 0.88))
        
        # Step 3: Contextual reasoning
        context_risk = self._assess_contextual_risk(robot_state, environment, tool_outputs)
        trace.append(ReasoningTrace(3, f"Context risk score: {context_risk:.2f}",
                                   self._explain_context_risk(context_risk, tool_outputs),
                                   "Analyze", 0.82))
        
        # Step 4: Determine safety level
        safety_level, confidence = self._determine_safety_level(violations, context_risk)
        trace.append(ReasoningTrace(4, f"Safety level: {safety_level.value}",
                                   self._explain_safety_decision(safety_level, violations, context_risk),
                                   "Decide", confidence))
        
        # Step 5: Generate action recommendation
        action = self._recommend_action(safety_level, violations, robot_state)
        trace.append(ReasoningTrace(5, f"Recommended action: {action['action']}",
                                   action['reasoning'],
                                   "Recommend", 0.85))
        
        self.reasoning_traces.extend(trace)
        
        return {
            "safety_level": safety_level.value,
            "confidence": confidence,
            "violations": violations,
            "risk_score": context_risk,
            "recommended_action": action,
            "reasoning_traces": [(t.step, t.observation, t.reasoning, t.action, t.confidence) 
                               for t in trace]
        }
    
    def _observe_state(self, robot: Dict, env: Dict) -> str:
        pos = robot.get('position', [0,0])
        humans = len(env.get('humans', []))
        obstacles = len(env.get('obstacles', []))
        return f"Robot at ({pos[0]:.1f}, {pos[1]:.1f}) with {humans} humans, {obstacles} obstacles"
    
    def _check_safety_violations(self, robot: Dict, env: Dict) -> List[Dict]:
        violations = []
        robot_pos = np.array(robot.get('position', [0,0]))
        robot_speed = robot.get('speed', 0)
        
        # Check distance to humans
        for human in env.get('humans', []):
            human_pos = np.array(human.get('position', [0,0]))
            distance = np.linalg.norm(robot_pos - human_pos)
            
            if distance < self.safety_rules['distance_rules']['critical_distance']:
                violations.append({
                    'type': 'critical_distance',
                    'severity': 10,
                    'details': f'Robot too close to human: {distance:.2f}m',
                    'location': human_pos.tolist()
                })
            elif distance < self.safety_rules['distance_rules']['safe_distance']:
                violations.append({
                    'type': 'warning_distance',
                    'severity': 5,
                    'details': f'Robot approaching human: {distance:.2f}m',
                    'location': human_pos.tolist()
                })
        
        # Check speed limit
        if robot_speed > self.safety_rules['speed_rules']['max_speed']:
            violations.append({
                'type': 'speed_violation',
                'severity': 7,
                'details': f'Robot speed {robot_speed:.2f}m/s exceeds limit',
                'location': robot_pos.tolist()
            })
        
        # Check zone violations
        for zone in env.get('restricted_zones', []):
            if self._point_in_zone(robot_pos, zone):
                violations.append({
                    'type': 'restricted_zone',
                    'severity': 8,
                    'details': f'Robot entered restricted zone: {zone.get("name", "unknown")}',
                    'location': robot_pos.tolist()
                })
        
        return violations
    
    def _point_in_zone(self, point: np.ndarray, zone: Dict) -> bool:
        if zone.get('type') == 'circle':
            center = np.array(zone['center'])
            radius = zone.get('radius', 1.0)
            return np.linalg.norm(point - center) < radius
        elif zone.get('type') == 'rectangle':
            x, y = point
            return (zone.get('x_min', -np.inf) <= x <= zone.get('x_max', np.inf) and 
                   zone.get('y_min', -np.inf) <= y <= zone.get('y_max', np.inf))
        return False
    
    def _assess_contextual_risk(self, robot: Dict, env: Dict, tool_outputs: Dict) -> float:
        """Calculate contextual risk score (0-100)"""
        risk = 0.0
        
        # Distance risk
        min_distance = float('inf')
        for human in env.get('humans', []):
            dist = np.linalg.norm(np.array(robot['position']) - np.array(human['position']))
            min_distance = min(min_distance, dist)
        
        if min_distance < 0.5:
            risk += 40
        elif min_distance < 1.0:
            risk += 20
        elif min_distance < 2.0:
            risk += 10
        
        # Speed risk
        speed = robot.get('speed', 0)
        risk += (speed / self.safety_rules['speed_rules']['max_speed']) * 20
        
        # Zone risk
        for zone in env.get('hazard_zones', []):
            if self._point_in_zone(np.array(robot['position']), zone):
                risk += 30
                break
        
        # Sensor noise risk
        if tool_outputs.get('sensor_noise', 0) > 0.3:
            risk += tool_outputs['sensor_noise'] * 20
        
        return min(100, risk)
    
    def _explain_violations(self, violations: List[Dict]) -> str:
        if not violations:
            return "No safety violations detected"
        return f"Detected {len(violations)} violations: {', '.join([v['type'] for v in violations])}"
    
    def _explain_context_risk(self, risk: float, tool_outputs: Dict) -> str:
        explanations = []
        if risk > 70:
            explanations.append("HIGH RISK: Multiple safety violations detected")
        elif risk > 40:
            explanations.append("MEDIUM RISK: Safety margins approaching limits")
        else:
            explanations.append("LOW RISK: Operating within safe parameters")
        
        if tool_outputs.get('sensor_noise', 0) > 0.3:
            explanations.append(f"Sensor noise ({tool_outputs['sensor_noise']:.2f}) affecting perception")
        
        return " | ".join(explanations)
    
    def _determine_safety_level(self, violations: List[Dict], risk: float) -> Tuple[SafetyLevel, float]:
        if any(v['severity'] >= 8 for v in violations) or risk > 80:
            return SafetyLevel.CRITICAL, 0.95
        elif len(violations) > 0 or risk > 50:
            return SafetyLevel.UNSAFE, 0.85
        elif risk > 25:
            return SafetyLevel.CAUTION, 0.75
        else:
            return SafetyLevel.SAFE, 0.90
    
    def _explain_safety_decision(self, level: SafetyLevel, violations: List[Dict], risk: float) -> str:
        if level == SafetyLevel.CRITICAL:
            return f"CRITICAL: {len(violations)} severe violations with {risk:.1f}% risk - immediate action required"
        elif level == SafetyLevel.UNSAFE:
            return f"UNSAFE: {len(violations)} safety violations detected"
        elif level == SafetyLevel.CAUTION:
            return f"CAUTION: Elevated risk level ({risk:.1f}%), monitoring required"
        else:
            return f"SAFE: Operating within all safety parameters"
    
    def _recommend_action(self, level: SafetyLevel, violations: List[Dict], robot: Dict) -> Dict:
        if level == SafetyLevel.CRITICAL:
            return {
                'action': 'EMERGENCY_STOP',
                'priority': 1,
                'reasoning': 'Immediate stop required to prevent collision or injury',
                'parameters': {'deceleration': 'maximum', 'alert': True}
            }
        elif level == SafetyLevel.UNSAFE:
            return {
                'action': 'REDUCE_SPEED',
                'priority': 2,
                'reasoning': f'Reduce speed from {robot.get("speed", 0):.2f} to safe level due to {len(violations)} violations',
                'parameters': {'target_speed': 0.5, 'ramp_time': 1.0}
            }
        elif level == SafetyLevel.CAUTION:
            return {
                'action': 'CONTINUE_MONITORING',
                'priority': 3,
                'reasoning': 'Maintain current operation with enhanced monitoring',
                'parameters': {'monitoring_rate': 2.0}
            }
        else:
            return {
                'action': 'NORMAL_OPERATION',
                'priority': 4,
                'reasoning': 'All systems nominal, continue normal operation',
                'parameters': {}
            }
    
    def explain_decision(self, decision: Dict) -> str:
        """Generate human-readable explanation"""
        explanation = f"""
SAFETY ASSESSMENT: {decision['safety_level'].upper()}
Confidence: {decision['confidence']*100:.1f}%

RISK ANALYSIS:
- Risk Score: {decision['risk_score']:.1f}/100
- Violations Found: {len(decision['violations'])}

RECOMMENDED ACTION:
- Action: {decision['recommended_action']['action']}
- Priority: {decision['recommended_action']['priority']}
- Reasoning: {decision['recommended_action']['reasoning']}

DETAILED REASONING:
"""
        
        for step, obs, reason, action, conf in decision['reasoning_traces']:
            explanation += f"\n{step}. {obs}\n   → Reasoning: {reason}\n   → Action: {action} (conf: {conf:.2f})"
        
        if decision['violations']:
            explanation += "\n\nVIOLATIONS:"
            for v in decision['violations']:
                explanation += f"\n- {v['type']}: {v['details']} (severity: {v['severity']})"
        
        return explanation
