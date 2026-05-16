import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any
    error: str
    execution_time: float

class ToolExecutor:
    """Executes safety analysis tools"""
    
    def __init__(self):
        self.tools = {
            'distance_calculator': self.calculate_distances,
            'collision_risk': self.estimate_collision_risk,
            'zone_intrusion': self.check_zone_intrusion,
            'speed_compliance': self.check_speed_compliance,
            'sensor_noise': self.detect_sensor_noise,
            'anomaly_detector': self.detect_anomalies,
            'path_risk': self.evaluate_path_risk,
            'uncertainty': self.estimate_uncertainty
        }
        
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a specific tool"""
        import time
        start = time.time()
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            result = self.tools[tool_name](**kwargs)
            execution_time = time.time() - start
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                error=None,
                execution_time=execution_time
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def calculate_distances(self, robot_pos: List[float], entities: List[Dict]) -> Dict:
        """Calculate distances to all entities"""
        robot = np.array(robot_pos)
        distances = []
        
        for entity in entities:
            entity_pos = np.array(entity.get('position', [0,0]))
            dist = np.linalg.norm(robot - entity_pos)
            distances.append({
                'entity_id': entity.get('id', 'unknown'),
                'entity_type': entity.get('type', 'unknown'),
                'distance': float(dist),
                'position': entity_pos.tolist()
            })
        
        distances.sort(key=lambda x: x['distance'])
        
        return {
            'min_distance': distances[0]['distance'] if distances else float('inf'),
            'closest_entity': distances[0] if distances else None,
            'all_distances': distances,
            'within_safe_zone': all(d['distance'] > 0.5 for d in distances)
        }
    
    def estimate_collision_risk(self, robot_pos: List[float], robot_vel: List[float],
                                humans: List[Dict], obstacles: List[Dict]) -> Dict:
        """Estimate collision risk using predictive modeling"""
        robot = np.array(robot_pos)
        robot_v = np.array(robot_vel)
        
        collision_probability = 0.0
        time_to_collision = float('inf')
        potential_collisions = []
        
        # Check humans
        for human in humans:
            human_pos = np.array(human.get('position', [0,0]))
            human_vel = np.array(human.get('velocity', [0,0]))
            
            rel_pos = human_pos - robot
            rel_vel = robot_v - human_vel
            
            # Time to closest approach
            if np.linalg.norm(rel_vel) > 1e-6:
                tca = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
                if tca > 0:
                    closest_dist = np.linalg.norm(rel_pos + rel_vel * tca)
                    
                    if closest_dist < 0.5:
                        prob = max(0, 1 - closest_dist / 0.5)
                        collision_probability += prob
                        time_to_collision = min(time_to_collision, tca)
                        potential_collisions.append({
                            'entity': human,
                            'probability': prob,
                            'time': tca,
                            'distance_at_approach': closest_dist
                        })
        
        # Check static obstacles
        for obstacle in obstacles:
            obs_pos = np.array(obstacle.get('position', [0,0]))
            obs_radius = obstacle.get('radius', 0.3)
            
            dist = np.linalg.norm(robot - obs_pos)
            if dist < obs_radius + 0.2:
                prob = max(0, 1 - dist / (obs_radius + 0.2))
                collision_probability += prob
                potential_collisions.append({
                    'entity': obstacle,
                    'probability': prob,
                    'time': 0,
                    'distance_at_approach': dist
                })
        
        return {
            'collision_probability': min(1.0, collision_probability),
            'time_to_collision': time_to_collision if time_to_collision != float('inf') else None,
            'potential_collisions': potential_collisions,
            'risk_level': self._risk_level_from_probability(collision_probability)
        }
    
    def _risk_level_from_probability(self, prob: float) -> str:
        if prob > 0.7:
            return 'CRITICAL'
        elif prob > 0.4:
            return 'HIGH'
        elif prob > 0.1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def check_zone_intrusion(self, robot_pos: List[float], 
                             restricted_zones: List[Dict],
                             hazard_zones: List[Dict]) -> Dict:
        """Check if robot entered restricted/hazard zones"""
        robot = np.array(robot_pos)
        
        intrusions = {
            'restricted': [],
            'hazard': []
        }
        
        for zone in restricted_zones:
            if self._point_in_zone(robot, zone):
                intrusions['restricted'].append({
                    'zone_name': zone.get('name', 'unknown'),
                    'zone_type': zone.get('type', 'unknown'),
                    'distance_to_boundary': self._distance_to_zone_boundary(robot, zone)
                })
        
        for zone in hazard_zones:
            if self._point_in_zone(robot, zone):
                intrusions['hazard'].append({
                    'zone_name': zone.get('name', 'unknown'),
                    'zone_type': zone.get('type', 'unknown'),
                    'hazard_level': zone.get('hazard_level', 1),
                    'distance_to_boundary': self._distance_to_zone_boundary(robot, zone)
                })
        
        return {
            'has_intrusion': len(intrusions['restricted']) > 0 or len(intrusions['hazard']) > 0,
            'intrusions': intrusions,
            'severity': max([z.get('hazard_level', 0) for z in intrusions['hazard']] + [0])
        }
    
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
    
    def _distance_to_zone_boundary(self, point: np.ndarray, zone: Dict) -> float:
        if zone.get('type') == 'circle':
            center = np.array(zone['center'])
            radius = zone.get('radius', 1.0)
            return abs(np.linalg.norm(point - center) - radius)
        elif zone.get('type') == 'rectangle':
            x, y = point
            dx = max(zone.get('x_min', -np.inf) - x, 0, x - zone.get('x_max', np.inf))
            dy = max(zone.get('y_min', -np.inf) - y, 0, y - zone.get('y_max', np.inf))
            return np.sqrt(dx*dx + dy*dy)
        return float('inf')
    
    def check_speed_compliance(self, robot_speed: float, max_speed: float = 2.0,
                               speed_limit_zones: List[Dict] = None,
                               robot_pos: List[float] = None) -> Dict:
        """Check if robot speed complies with limits"""
        compliance = {
            'is_compliant': robot_speed <= max_speed,
            'current_speed': robot_speed,
            'max_allowed': max_speed,
            'violation_percentage': max(0, (robot_speed - max_speed) / max_speed * 100) if robot_speed > max_speed else 0
        }
        
        # Check zone-specific speed limits
        if speed_limit_zones and robot_pos:
            robot = np.array(robot_pos)
            for zone in speed_limit_zones:
                if self._point_in_zone(robot, zone):
                    zone_limit = zone.get('speed_limit', max_speed)
                    compliance['zone_specific'] = {
                        'zone_name': zone.get('name', 'unknown'),
                        'speed_limit': zone_limit,
                        'is_compliant_in_zone': robot_speed <= zone_limit,
                        'violation_in_zone': max(0, robot_speed - zone_limit)
                    }
                    compliance['is_compliant'] = compliance['is_compliant'] and compliance['zone_specific']['is_compliant_in_zone']
                    break
        
        return compliance
    
    def detect_sensor_noise(self, sensor_readings: List[float], 
                            expected_range: Tuple[float, float] = None,
                            historical_readings: List[List[float]] = None) -> Dict:
        """Detect and quantify sensor noise"""
        readings = np.array(sensor_readings)
        
        # Basic statistical analysis
        noise_metrics = {
            'mean': float(np.mean(readings)),
            'std': float(np.std(readings)),
            'variance': float(np.var(readings)),
            'min': float(np.min(readings)),
            'max': float(np.max(readings)),
            'range': float(np.max(readings) - np.min(readings))
        }
        
        # Detect outliers. Use z-score plus a median absolute deviation check so
        # a single spike cannot inflate the standard deviation enough to hide.
        z_scores = np.abs((readings - noise_metrics['mean']) / (noise_metrics['std'] + 1e-6))
        median = np.median(readings)
        mad = np.median(np.abs(readings - median))
        if mad > 1e-6:
            modified_z_scores = 0.6745 * np.abs(readings - median) / mad
            outlier_mask = (z_scores > 3) | (modified_z_scores > 3.5)
        else:
            outlier_mask = z_scores > 3

        outlier_count = int(np.sum(outlier_mask))
        noise_metrics['outlier_count'] = outlier_count
        noise_metrics['outlier_percentage'] = outlier_count / len(readings) * 100
        
        # Determine noise level
        noise_level = min(1.0, noise_metrics['std'] / (noise_metrics['range'] + 1e-6) * 2)
        noise_metrics['noise_level'] = float(noise_level)
        noise_metrics['noise_severity'] = self._noise_severity(noise_level)
        
        # Check expected range
        if expected_range:
            out_of_range = np.sum((readings < expected_range[0]) | (readings > expected_range[1]))
            noise_metrics['out_of_range_percentage'] = out_of_range / len(readings) * 100
            noise_metrics['is_reliable'] = noise_metrics['out_of_range_percentage'] < 10
        
        return noise_metrics
    
    def _noise_severity(self, noise_level: float) -> str:
        if noise_level > 0.7:
            return 'SEVERE'
        elif noise_level > 0.4:
            return 'MODERATE'
        elif noise_level > 0.1:
            return 'MILD'
        else:
            return 'MINIMAL'
    
    def detect_anomalies(self, current_state: Dict, 
                         historical_states: List[Dict],
                         threshold: float = 2.0) -> Dict:
        """Detect anomalies in robot behavior"""
        anomalies = []
        
        # Extract relevant features
        current_features = np.array([
            current_state.get('speed', 0),
            current_state.get('acceleration', 0),
            current_state.get('distance_to_nearest_human', 10),
            current_state.get('distance_to_nearest_obstacle', 10)
        ])
        
        if historical_states:
            # Build historical feature matrix
            historical_features = np.array([[
                s.get('speed', 0),
                s.get('acceleration', 0),
                s.get('distance_to_nearest_human', 10),
                s.get('distance_to_nearest_obstacle', 10)
            ] for s in historical_states[-100:]])
            
            if len(historical_features) > 0:
                # Calculate z-scores
                means = np.mean(historical_features, axis=0)
                stds = np.std(historical_features, axis=0) + 1e-6
                z_scores = np.abs((current_features - means) / stds)
                
                for i, (feature, z) in enumerate(zip(['speed', 'acceleration', 'human_distance', 'obstacle_distance'], z_scores)):
                    if z > threshold:
                        anomalies.append({
                            'feature': feature,
                            'z_score': float(z),
                            'current_value': float(current_features[i]),
                            'expected_mean': float(means[i]),
                            'severity': min(1.0, (z - threshold) / threshold)
                        })
        
        return {
            'has_anomalies': len(anomalies) > 0,
            'anomalies': anomalies,
            'anomaly_score': float(np.mean([a['severity'] for a in anomalies])) if anomalies else 0
        }
    
    def evaluate_path_risk(self, path: List[List[float]], 
                          environment: Dict,
                          resolution: float = 0.1) -> Dict:
        """Evaluate risk along a planned path"""
        if not path or len(path) < 2:
            return {'error': 'Invalid path', 'risk_score': 1.0}
        
        path = np.array(path)
        total_risk = 0
        segments = []
        
        for i in range(len(path) - 1):
            segment = path[i:i+2]
            segment_length = np.linalg.norm(segment[1] - segment[0])
            
            # Sample points along segment
            num_samples = max(2, int(segment_length / resolution))
            for t in np.linspace(0, 1, num_samples):
                point = segment[0] + t * (segment[1] - segment[0])
                
                # Calculate risk at point
                point_risk = self._calculate_point_risk(point, environment)
                total_risk += point_risk * (segment_length / num_samples)
                
                segments.append({
                    'start': segment[0].tolist(),
                    'end': segment[1].tolist(),
                    'length': float(segment_length),
                    'max_risk': float(np.max([self._calculate_point_risk(point, environment) 
                                             for point in [segment[0], segment[1]]])),
                    'mean_risk': float(point_risk)
                })
        
        return {
            'total_path_risk': float(total_risk),
            'normalized_risk': float(total_risk / max(1, len(path))),
            'segments': segments,
            'max_segment_risk': float(max(s.get('max_risk', 0) for s in segments)),
            'is_safe': total_risk < 10.0
        }
    
    def _calculate_point_risk(self, point: np.ndarray, environment: Dict) -> float:
        """Calculate risk at a specific point"""
        risk = 0.0
        
        # Distance to humans
        for human in environment.get('humans', []):
            human_pos = np.array(human.get('position', [0,0]))
            dist = np.linalg.norm(point - human_pos)
            if dist < 1.0:
                risk += (1.0 - dist) * 10
        
        # Zone risks
        for zone in environment.get('hazard_zones', []):
            if self._point_in_zone(point, zone):
                risk += zone.get('hazard_level', 1) * 5
        
        for zone in environment.get('restricted_zones', []):
            if self._point_in_zone(point, zone):
                risk += 10
        
        return min(100, risk)
    
    def estimate_uncertainty(self, measurements: List[float],
                            prediction_model: str = 'statistical') -> Dict:
        """Estimate uncertainty in measurements"""
        measurements = np.array(measurements)
        
        if len(measurements) < 2:
            return {'uncertainty': 0.5, 'confidence': 0.5, 'error': 'Insufficient data'}
        
        uncertainty_metrics = {
            'statistical_uncertainty': float(np.std(measurements) / (np.mean(measurements) + 1e-6)),
            'measurement_variance': float(np.var(measurements)),
            'confidence_interval_95': {
                'lower': float(np.percentile(measurements, 2.5)),
                'upper': float(np.percentile(measurements, 97.5))
            },
            'data_quality': min(1.0, len(measurements) / 100)
        }
        
        # Calculate overall uncertainty
        overall_uncertainty = min(1.0, uncertainty_metrics['statistical_uncertainty'])
        
        return {
            'uncertainty': overall_uncertainty,
            'confidence': 1.0 - overall_uncertainty,
            'metrics': uncertainty_metrics,
            'recommendation': self._uncertainty_recommendation(overall_uncertainty)
        }
    
    def _uncertainty_recommendation(self, uncertainty: float) -> str:
        if uncertainty > 0.7:
            return "CRITICAL: High uncertainty - immediate sensor calibration required"
        elif uncertainty > 0.4:
            return "WARNING: Moderate uncertainty - verify measurements"
        elif uncertainty > 0.1:
            return "INFO: Low uncertainty - acceptable for operation"
        else:
            return "GOOD: Very low uncertainty - high confidence in measurements"
