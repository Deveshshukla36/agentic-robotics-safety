from typing import Dict, List, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from .reasoning_engine import LocalReasoningEngine
from .tool_executor import ToolExecutor
from .memory_agent import MemoryAgent

class AgentOrchestrator:
    """Orchestrates all agents in the system"""
    
    def __init__(self):
        self.reasoning_engine = LocalReasoningEngine()
        self.tool_executor = ToolExecutor()
        self.memory_agent = MemoryAgent()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.planning_history = []
        
    async def process_scenario(self, scenario: Dict) -> Dict:
        """Main entry point for scenario processing"""
        start_time = time.time()
        
        # Validate scenario
        if not self._validate_scenario(scenario):
            return self._error_response("Invalid scenario format")
        
        # Check for similar past scenarios
        similar = self.memory_agent.find_similar_scenarios(scenario)
        
        # Step 1: Planning
        plan = self._create_plan(scenario)
        
        # Step 2: Execute tools
        tool_outputs = await self._execute_tools(plan['required_tools'], scenario)
        
        # Step 3: Reason about safety
        reasoning_result = self.reasoning_engine.reason_about_safety(
            scenario.get('robot', {}),
            scenario.get('environment', {}),
            tool_outputs
        )
        
        # Step 4: Evaluate decision
        evaluation = self._evaluate_decision(reasoning_result, tool_outputs)
        
        # Step 5: Store in memory
        scenario_id = self.memory_agent.store_scenario(scenario)
        self.memory_agent.store_decision(scenario_id, reasoning_result)
        
        if reasoning_result['safety_level'] in ['unsafe', 'critical']:
            self.memory_agent.store_safety_incident({
                'severity': 10 if reasoning_result['safety_level'] == 'critical' else 5,
                'violations': reasoning_result['violations'],
                'risk_score': reasoning_result['risk_score']
            })
        
        # Generate explanation
        explanation = self.reasoning_engine.explain_decision(reasoning_result)
        
        processing_time = time.time() - start_time
        
        return {
            'scenario_id': scenario_id,
            'safety_assessment': reasoning_result,
            'evaluation': evaluation,
            'similar_scenarios': similar,
            'explanation': explanation,
            'tool_outputs': tool_outputs,
            'processing_time_ms': processing_time * 1000,
            'plan': plan
        }
    
    def _validate_scenario(self, scenario: Dict) -> bool:
        """Validate scenario structure"""
        if 'robot' not in scenario or 'environment' not in scenario:
            return False
        
        if 'position' not in scenario.get('robot', {}):
            return False
        
        return True
    
    def _create_plan(self, scenario: Dict) -> Dict:
        """Create execution plan"""
        required_tools = ['distance_calculator', 'zone_intrusion']
        
        # Add tools based on scenario
        if scenario['robot'].get('speed', 0) > 0:
            required_tools.append('speed_compliance')
            required_tools.append('collision_risk')
        
        if scenario.get('has_sensor_noise', False):
            required_tools.append('sensor_noise')
        
        if scenario.get('environment', {}).get('obstacles'):
            required_tools.append('path_risk')
        
        plan = {
            'required_tools': required_tools,
            'execution_order': self._determine_execution_order(required_tools),
            'parallel_tools': self._get_parallel_tools(required_tools)
        }
        
        self.planning_history.append({
            'timestamp': time.time(),
            'scenario': scenario,
            'plan': plan
        })
        
        return plan
    
    def _determine_execution_order(self, tools: List[str]) -> List[str]:
        """Determine optimal tool execution order"""
        dependencies = {
            'distance_calculator': [],
            'zone_intrusion': ['distance_calculator'],
            'collision_risk': ['distance_calculator', 'speed_compliance'],
            'speed_compliance': [],
            'sensor_noise': [],
            'anomaly_detector': ['distance_calculator', 'speed_compliance'],
            'path_risk': ['distance_calculator', 'zone_intrusion']
        }
        
        ordered = []
        remaining = set(tools)
        
        while remaining:
            for tool in list(remaining):
                deps = dependencies.get(tool, [])
                if all(dep in ordered for dep in deps):
                    ordered.append(tool)
                    remaining.remove(tool)
                    break
        
        return ordered
    
    def _get_parallel_tools(self, tools: List[str]) -> List[List[str]]:
        """Group tools that can run in parallel"""
        parallel_groups = []
        independent_tools = ['distance_calculator', 'speed_compliance', 'sensor_noise']
        
        group = [t for t in tools if t in independent_tools]
        if group:
            parallel_groups.append(group)
        
        dependent = [t for t in tools if t not in independent_tools]
        for tool in dependent:
            parallel_groups.append([tool])
        
        return parallel_groups
    
    async def _execute_tools(self, tools: List[str], scenario: Dict) -> Dict:
        """Execute required tools"""
        results = {}
        robot = scenario.get('robot', {})
        env = scenario.get('environment', {})
        
        for tool_name in tools:
            if tool_name == 'distance_calculator':
                entities = env.get('humans', []) + env.get('obstacles', [])
                result = self.tool_executor.execute_tool(
                    'distance_calculator',
                    robot_pos=robot.get('position', [0,0]),
                    entities=entities
                )
                
            elif tool_name == 'collision_risk':
                result = self.tool_executor.execute_tool(
                    'collision_risk',
                    robot_pos=robot.get('position', [0,0]),
                    robot_vel=robot.get('velocity', [0,0]),
                    humans=env.get('humans', []),
                    obstacles=env.get('obstacles', [])
                )
                
            elif tool_name == 'zone_intrusion':
                result = self.tool_executor.execute_tool(
                    'zone_intrusion',
                    robot_pos=robot.get('position', [0,0]),
                    restricted_zones=env.get('restricted_zones', []),
                    hazard_zones=env.get('hazard_zones', [])
                )
                
            elif tool_name == 'speed_compliance':
                result = self.tool_executor.execute_tool(
                    'speed_compliance',
                    robot_speed=robot.get('speed', 0),
                    max_speed=env.get('max_speed', 2.0),
                    speed_limit_zones=env.get('speed_limit_zones', [])
                )
                
            elif tool_name == 'sensor_noise':
                readings = [robot.get('speed', 0) + np.random.normal(0, 0.1) for _ in range(10)]
                result = self.tool_executor.execute_tool(
                    'sensor_noise',
                    sensor_readings=readings,
                    expected_range=(0, env.get('max_speed', 2.0))
                )
                
            elif tool_name == 'anomaly_detector':
                result = self.tool_executor.execute_tool(
                    'anomaly_detector',
                    current_state=robot,
                    historical_states=list(self.memory_agent.decisions)
                )
                
            elif tool_name == 'path_risk':
                start = robot.get('position', [0,0])
                end = env.get('goal', [10,10])
                path = [start, end]
                result = self.tool_executor.execute_tool(
                    'path_risk',
                    path=path,
                    environment=env
                )
                
            else:
                result = self.tool_executor.execute_tool(tool_name)
            
            if result.success:
                results[tool_name] = result.result
            else:
                results[tool_name] = {'error': result.error}
        
        return results
    
    def _evaluate_decision(self, reasoning_result: Dict, tool_outputs: Dict) -> Dict:
        """Evaluate the quality of the decision"""
        score = 0
        max_score = 0
        
        # Correctness (30%)
        max_score += 30
        if reasoning_result['safety_level'] in ['safe', 'caution', 'unsafe', 'critical']:
            score += 30
        
        # Reasoning quality (25%)
        max_score += 25
        if len(reasoning_result.get('reasoning_traces', [])) > 0:
            score += 15
        if reasoning_result.get('confidence', 0) > 0.7:
            score += 10
        
        # Tool usage (20%)
        max_score += 20
        successful_tools = sum(1 for v in tool_outputs.values() if 'error' not in v)
        score += (successful_tools / max(1, len(tool_outputs))) * 20
        
        # Consistency (25%)
        max_score += 25
        if len(reasoning_result.get('violations', [])) > 0 and reasoning_result['risk_score'] > 50:
            score += 25
        elif len(reasoning_result.get('violations', [])) == 0 and reasoning_result['risk_score'] < 30:
            score += 25
        
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': percentage,
            'grade': self._calculate_grade(percentage)
        }
    
    def _calculate_grade(self, percentage: float) -> str:
        if percentage >= 90:
            return 'A+'
        elif percentage >= 80:
            return 'A'
        elif percentage >= 70:
            return 'B'
        elif percentage >= 60:
            return 'C'
        else:
            return 'F'
    
    def _error_response(self, message: str) -> Dict:
        return {
            'error': True,
            'message': message,
            'safety_assessment': {
                'safety_level': 'error',
                'confidence': 0,
                'violations': [],
                'risk_score': 100,
                'recommended_action': {'action': 'ERROR', 'reasoning': message},
                'reasoning_traces': []
            },
            'evaluation': {'score': 0, 'percentage': 0, 'grade': 'F'},
            'explanation': f"Error: {message}"
        }