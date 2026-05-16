import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class Entity:
    id: str
    position: np.ndarray
    velocity: np.ndarray
    type: str
    radius: float = 0.3

class SimulationEngine:
    """2D physics simulation for robotics testing"""
    
    def __init__(self, width: float = 10.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.robot = None
        self.humans = []
        self.obstacles = []
        self.restricted_zones = []
        self.hazard_zones = []
        self.dt = 0.1
        self.time = 0
        
        self.reset()
    
    def reset(self, config: Optional[Dict] = None):
        """Reset simulation to initial state"""
        if config:
            self.width = config.get('width', 10.0)
            self.height = config.get('height', 10.0)
            
            self.robot = Entity(
                id='robot',
                position=np.array(config.get('robot_position', [5.0, 5.0])),
                velocity=np.array([0, 0]),
                type='robot',
                radius=0.4
            )
            
            self.humans = []
            for h in config.get('humans', []):
                self.humans.append(Entity(
                    id=h.get('id', f"human_{len(self.humans)}"),
                    position=np.array(h['position']),
                    velocity=np.array(h.get('velocity', [0, 0])),
                    type='human',
                    radius=0.3
                ))
            
            self.obstacles = config.get('obstacles', [])
            self.restricted_zones = config.get('restricted_zones', [])
            self.hazard_zones = config.get('hazard_zones', [])
        else:
            # Default configuration
            self.robot = Entity(
                id='robot',
                position=np.array([5.0, 5.0]),
                velocity=np.array([0, 0]),
                type='robot',
                radius=0.4
            )
            
            self.humans = [
                Entity(
                    id='human_1',
                    position=np.array([3.0, 4.0]),
                    velocity=np.array([0.2, 0.1]),
                    type='human',
                    radius=0.3
                ),
                Entity(
                    id='human_2',
                    position=np.array([7.0, 6.0]),
                    velocity=np.array([-0.1, -0.2]),
                    type='human',
                    radius=0.3
                )
            ]
            
            self.obstacles = [
                {'position': [2.0, 2.0], 'radius': 0.5, 'type': 'pillar'},
                {'position': [8.0, 8.0], 'radius': 0.5, 'type': 'pillar'}
            ]
            
            self.restricted_zones = [
                {'type': 'circle', 'center': [1.0, 1.0], 'radius': 1.5, 'name': 'maintenance_area'}
            ]
            
            self.hazard_zones = [
                {'type': 'rectangle', 'x_min': 8.0, 'x_max': 10.0, 'y_min': 0.0, 'y_max': 2.0, 
                 'name': 'high_voltage', 'hazard_level': 3}
            ]
        
        self.time = 0
    
    def step(self, action: Dict) -> Dict:
        """Execute one simulation step"""
        # Update robot based on action
        if 'velocity' in action:
            self.robot.velocity = np.array(action['velocity'])
        elif 'acceleration' in action:
            self.robot.velocity += np.array(action['acceleration']) * self.dt
        
        # Apply speed limit
        speed = np.linalg.norm(self.robot.velocity)
        max_speed = 2.0
        if speed > max_speed:
            self.robot.velocity = self.robot.velocity / speed * max_speed
        
        # Update position
        new_position = self.robot.position + self.robot.velocity * self.dt
        
        # Boundary collision
        new_position[0] = np.clip(new_position[0], self.robot.radius, self.width - self.robot.radius)
        new_position[1] = np.clip(new_position[1], self.robot.radius, self.height - self.robot.radius)
        
        # Obstacle collision
        for obstacle in self.obstacles:
            obs_pos = np.array(obstacle['position'])
            obs_radius = obstacle.get('radius', 0.5)
            if np.linalg.norm(new_position - obs_pos) < self.robot.radius + obs_radius:
                # Bounce off obstacle
                normal = new_position - obs_pos
                normal = normal / np.linalg.norm(normal)
                new_position = obs_pos + normal * (self.robot.radius + obs_radius)
                self.robot.velocity = self.robot.velocity - 2 * np.dot(self.robot.velocity, normal) * normal
        
        self.robot.position = new_position
        
        # Update humans with random walk
        for human in self.humans:
            if np.random.random() < 0.1:
                human.velocity = np.random.randn(2) * 0.2
            
            new_human_pos = human.position + human.velocity * self.dt
            new_human_pos = np.clip(new_human_pos, human.radius, self.width - human.radius)
            human.position = new_human_pos
        
        self.time += self.dt
        
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current simulation state"""
        return {
            'time': self.time,
            'robot': {
                'id': self.robot.id,
                'position': self.robot.position.tolist(),
                'velocity': self.robot.velocity.tolist(),
                'speed': float(np.linalg.norm(self.robot.velocity))
            },
            'humans': [
                {
                    'id': h.id,
                    'position': h.position.tolist(),
                    'velocity': h.velocity.tolist()
                }
                for h in self.humans
            ],
            'obstacles': self.obstacles,
            'restricted_zones': self.restricted_zones,
            'hazard_zones': self.hazard_zones,
            'dimensions': {'width': self.width, 'height': self.height}
        }
    
    def get_config(self) -> Dict:
        """Get simulation configuration"""
        return {
            'width': self.width,
            'height': self.height,
            'robot_position': self.robot.position.tolist(),
            'humans': [{'position': h.position.tolist(), 'velocity': h.velocity.tolist()} for h in self.humans],
            'obstacles': self.obstacles,
            'restricted_zones': self.restricted_zones,
            'hazard_zones': self.hazard_zones
        }
    
    def set_robot_position(self, position: List[float]):
        """Manually set robot position"""
        self.robot.position = np.array(position)
    
    def add_human(self, position: List[float], velocity: List[float] = None):
        """Add a new human entity"""
        new_id = f"human_{len(self.humans) + 1}"
        self.humans.append(Entity(
            id=new_id,
            position=np.array(position),
            velocity=np.array(velocity if velocity else [0, 0]),
            type='human',
            radius=0.3
        ))
        return new_id