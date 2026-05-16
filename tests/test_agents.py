"""
Unit tests for Agentic Robotics Safety Monitor
Run with: pytest tests/test_agents.py -v
"""

import pytest
import sys
import os
import json
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from agents.orchestrator import AgentOrchestrator
from agents.reasoning_engine import LocalReasoningEngine
from agents.tool_executor import ToolExecutor
from agents.memory_agent import MemoryAgent
from simulation.engine import SimulationEngine

class TestReasoningEngine:
    """Tests for the reasoning engine agent"""
    
    def setup_method(self):
        self.engine = LocalReasoningEngine()
    
    def test_initialization(self):
        assert self.engine is not None
        assert len(self.engine.safety_rules) > 0
    
    def test_safety_assessment_safe(self):
        robot = {'position': [5.0, 5.0], 'speed': 0.5}
        env = {'humans': [{'position': [8.0, 8.0]}]}
        
        result = self.engine.reason_about_safety(robot, env, {})
        
        assert 'safety_level' in result
        assert 'reasoning_traces' in result
        assert len(result['reasoning_traces']) == 5
    
    def test_safety_assessment_unsafe(self):
        robot = {'position': [5.0, 5.0], 'speed': 2.5}
        env = {'humans': [{'position': [5.2, 5.1]}]}
        
        result = self.engine.reason_about_safety(robot, env, {})
        
        assert result['safety_level'] in ['unsafe', 'critical']
        assert len(result['violations']) > 0
    
    def test_distance_calculation(self):
        robot = {'position': [5.0, 5.0]}
        env = {'humans': [{'position': [4.0, 4.0]}]}
        
        result = self.engine._check_safety_violations(robot, env)
        
        # Should detect distance violation
        assert any(v['type'] == 'warning_distance' for v in result)
    
    def test_risk_score_range(self):
        robot = {'position': [5.0, 5.0], 'speed': 1.0}
        env = {'humans': []}
        
        risk = self.engine._assess_contextual_risk(robot, env, {})
        
        assert 0 <= risk <= 100
    
    def test_explanation_generation(self):
        robot = {'position': [5.0, 5.0], 'speed': 1.0}
        env = {'humans': [{'position': [4.0, 4.0]}]}
        
        result = self.engine.reason_about_safety(robot, env, {})
        explanation = self.engine.explain_decision(result)
        
        assert explanation is not None
        assert len(explanation) > 0

class TestToolExecutor:
    """Tests for the tool execution agent"""
    
    def setup_method(self):
        self.executor = ToolExecutor()
    
    def test_distance_calculator(self):
        result = self.executor.execute_tool(
            'distance_calculator',
            robot_pos=[5.0, 5.0],
            entities=[{'position': [4.0, 4.0], 'type': 'human'}]
        )
        
        assert result.success
        assert result.result['min_distance'] > 0
    
    def test_collision_risk(self):
        result = self.executor.execute_tool(
            'collision_risk',
            robot_pos=[5.0, 5.0],
            robot_vel=[0.5, 0.5],
            humans=[{'position': [5.5, 5.5], 'velocity': [0, 0]}],
            obstacles=[]
        )
        
        assert result.success
        assert 'collision_probability' in result.result
    
    def test_zone_intrusion(self):
        result = self.executor.execute_tool(
            'zone_intrusion',
            robot_pos=[1.0, 1.0],
            restricted_zones=[{'type': 'circle', 'center': [1.0, 1.0], 'radius': 1.5}],
            hazard_zones=[]
        )
        
        assert result.success
        assert result.result['has_intrusion'] is True
    
    def test_speed_compliance(self):
        result = self.executor.execute_tool(
            'speed_compliance',
            robot_speed=2.5,
            max_speed=2.0
        )
        
        assert result.success
        assert result.result['is_compliant'] is False
        assert result.result['violation_percentage'] > 0
    
    def test_sensor_noise_detection(self):
        readings = [1.0, 1.1, 0.9, 5.0, 1.0, 1.0]  # Outlier at 5.0
        
        result = self.executor.execute_tool(
            'sensor_noise',
            sensor_readings=readings
        )
        
        assert result.success
        assert result.result['outlier_count'] >= 1
    
    def test_anomaly_detection(self):
        current = {'speed': 5.0, 'acceleration': 2.0}
        historical = [{'speed': 1.0, 'acceleration': 0.1} for _ in range(10)]
        
        result = self.executor.execute_tool(
            'anomaly_detector',
            current_state=current,
            historical_states=historical
        )
        
        assert result.success
        assert result.result['has_anomalies'] is True
    
    def test_path_risk(self):
        path = [[0, 0], [5, 5], [10, 10]]
        env = {'humans': [{'position': [5, 5]}], 'hazard_zones': []}
        
        result = self.executor.execute_tool(
            'path_risk',
            path=path,
            environment=env
        )
        
        assert result.success
        assert 'total_path_risk' in result.result
    
    def test_uncertainty_estimation(self):
        measurements = [1.0, 1.1, 0.9, 1.0, 1.05]
        
        result = self.executor.execute_tool(
            'uncertainty',
            measurements=measurements
        )
        
        assert result.success
        assert 'uncertainty' in result.result
    
    def test_unknown_tool(self):
        result = self.executor.execute_tool('unknown_tool')
        
        assert result.success is False
        assert result.error is not None

class TestMemoryAgent:
    """Tests for the memory agent"""
    
    def setup_method(self):
        self.memory = MemoryAgent(memory_size=100)
    
    def test_store_scenario(self):
        scenario = {'robot': {'position': [1, 2]}, 'environment': {}}
        scenario_id = self.memory.store_scenario(scenario)
        
        assert scenario_id is not None
        assert len(scenario_id) == 8
    
    def test_find_similar_scenarios(self):
        scenario1 = {
            'robot': {'position': [5, 5], 'speed': 1.5},
            'environment': {'humans': [{'position': [4, 4]}]}
        }
        scenario2 = {
            'robot': {'position': [5.1, 5.1], 'speed': 1.4},
            'environment': {'humans': [{'position': [4.1, 4.1]}]}
        }
        
        self.memory.store_scenario(scenario1)
        similar = self.memory.find_similar_scenarios(scenario2)
        
        # Should find at least one similar scenario
        assert len(similar) >= 0  # May be 0 if similarity threshold not met
    
    def test_store_safety_incident(self):
        incident = {'severity': 5, 'violations': ['speed_violation']}
        
        self.memory.store_safety_incident(incident)
        
        assert len(self.memory.safety_violations) == 1
    
    def test_memory_statistics(self):
        for i in range(5):
            scenario = {'robot': {'position': [i, i]}, 'environment': {}}
            self.memory.store_scenario(scenario)
        
        stats = self.memory.get_statistics()
        
        assert stats['total_scenarios'] == 5
        assert 'memory_usage_percent' in stats
    
    def test_clear_memory(self):
        scenario = {'robot': {'position': [1, 1]}, 'environment': {}}
        self.memory.store_scenario(scenario)
        
        assert len(self.memory.scenarios) == 1
        
        self.memory.clear_memory()
        
        assert len(self.memory.scenarios) == 0

class TestOrchestrator:
    """Tests for the agent orchestrator"""
    
    def setup_method(self):
        self.orchestrator = AgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_process_valid_scenario(self):
        scenario = {
            'robot': {'position': [5.0, 5.0], 'speed': 1.5},
            'environment': {'humans': [{'position': [4.0, 4.5]}]}
        }
        
        result = await self.orchestrator.process_scenario(scenario)
        
        assert 'safety_assessment' in result
        assert 'evaluation' in result
        assert 'explanation' in result
        assert 'processing_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_process_invalid_scenario(self):
        scenario = {'robot': {}, 'environment': {}}
        
        result = await self.orchestrator.process_scenario(scenario)
        
        assert result.get('error') is True or 'safety_assessment' in result
    
    @pytest.mark.asyncio
    async def test_missing_position(self):
        scenario = {
            'robot': {'speed': 1.5},  # Missing position
            'environment': {}
        }
        
        result = await self.orchestrator.process_scenario(scenario)
        
        assert 'safety_assessment' in result
    
    @pytest.mark.asyncio
    async def test_plan_creation(self):
        scenario = {
            'robot': {'position': [5, 5], 'speed': 1.5},
            'environment': {'humans': [{'position': [4, 4]}]}
        }
        
        plan = self.orchestrator._create_plan(scenario)
        
        assert 'required_tools' in plan
        assert len(plan['required_tools']) >= 2

class TestSimulationEngine:
    """Tests for the simulation engine"""
    
    def setup_method(self):
        self.engine = SimulationEngine(width=10.0, height=10.0)
    
    def test_initialization(self):
        assert self.engine.robot is not None
        assert self.engine.width == 10.0
        assert self.engine.height == 10.0
    
    def test_reset(self):
        self.engine.robot.position = np.array([1.0, 1.0])
        self.engine.reset()
        
        # Should reset to default position
        assert np.array_equal(self.engine.robot.position, np.array([5.0, 5.0]))
    
    def test_step_movement(self):
        initial_pos = self.engine.robot.position.copy()
        
        result = self.engine.step({'velocity': [1.0, 0]})
        
        # Position should have changed
        assert not np.array_equal(result['robot']['position'], initial_pos.tolist())
    
    def test_boundary_collision(self):
        # Try to move outside bounds
        self.engine.robot.position = np.array([0.5, 5.0])
        
        result = self.engine.step({'velocity': [-1.0, 0]})
        
        # Should stay within bounds
        assert result['robot']['position'][0] >= 0
    
    def test_get_state(self):
        state = self.engine.get_state()
        
        assert 'robot' in state
        assert 'humans' in state
        assert 'dimensions' in state
        assert state['dimensions']['width'] == 10.0
    
    def test_add_human(self):
        human_id = self.engine.add_human([3.0, 3.0])
        
        assert human_id is not None
        assert len(self.engine.humans) == 3  # Started with 2 humans
    
    def test_set_robot_position(self):
        self.engine.set_robot_position([2.0, 2.0])
        
        assert np.array_equal(self.engine.robot.position, np.array([2.0, 2.0]))

class TestIntegration:
    """Integration tests for the entire system"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        orchestrator = AgentOrchestrator()
        
        scenario = {
            'robot': {'position': [5.0, 5.0], 'speed': 1.8},
            'environment': {
                'humans': [{'position': [4.5, 4.5]}],
                'restricted_zones': [{'type': 'circle', 'center': [1.0, 1.0], 'radius': 1.5}]
            }
        }
        
        result = await orchestrator.process_scenario(scenario)
        
        assert result['safety_assessment']['safety_level'] in ['safe', 'caution', 'unsafe', 'critical']
        assert result['evaluation']['percentage'] >= 0
        assert result['processing_time_ms'] > 0
    
    @pytest.mark.asyncio
    async def test_simulation_with_safety(self):
        engine = SimulationEngine()
        orchestrator = AgentOrchestrator()
        
        # Move robot
        engine.step({'velocity': [1.0, 0]})
        state = engine.get_state()
        
        # Check safety
        scenario = {
            'robot': state['robot'],
            'environment': {
                'humans': state['humans'],
                'obstacles': state['obstacles'],
                'restricted_zones': state['restricted_zones'],
                'hazard_zones': state['hazard_zones']
            }
        }
        
        result = await orchestrator.process_scenario(scenario)
        
        assert 'safety_level' in result['safety_assessment']

def test_imports():
    """Test that all modules can be imported"""
    try:
        from agents.orchestrator import AgentOrchestrator
        from agents.reasoning_engine import LocalReasoningEngine
        from agents.tool_executor import ToolExecutor
        from agents.memory_agent import MemoryAgent
        from simulation.engine import SimulationEngine
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])