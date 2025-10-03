# tracker_counter_with_posture.py

import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

class PostureAnalyzer:
    """Analyze human posture (sitting or standing)"""
    
    # COCO keypoint indices
    KEYPOINT_INDICES = {
        'nose': 0,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    def __init__(self, sitting_threshold=0.4):
        """
        Args:
            sitting_threshold: Sitting posture threshold (hip to knee distance / hip to ankle distance)
                              Higher values make it easier to classify as sitting
        """
        self.sitting_threshold = sitting_threshold
        self.confidence_threshold = 0.5  # Lower confidence threshold to get more valid keypoints
        self.min_required_confidence = 0.3  # Minimum acceptable confidence
    
    def get_keypoint(self, keypoints, index, min_confidence=None):
        """Get keypoint coordinates at specified index with dynamic confidence threshold"""
        if index < len(keypoints):
            x, y, conf = keypoints[index]
            threshold = min_confidence if min_confidence is not None else self.confidence_threshold
            if conf > threshold:
                return x, y, conf  # Return confidence for subsequent use
        return None
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return None
        # Only use coordinates, ignore confidence
        p1 = point1[:2] if len(point1) > 2 else point1
        p2 = point2[:2] if len(point2) > 2 else point2
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_weighted_center(self, points_with_conf):
        """Calculate weighted center based on confidence scores"""
        if not points_with_conf:
            return None
        
        valid_points = [p for p in points_with_conf if p is not None]
        if not valid_points:
            return None
        
        if len(valid_points) == 1:
            return valid_points[0][:2]  # Only return coordinates
        
        # Weighted average
        total_weight = sum(p[2] for p in valid_points)
        if total_weight == 0:
            return None
        
        weighted_x = sum(p[0] * p[2] for p in valid_points) / total_weight
        weighted_y = sum(p[1] * p[2] for p in valid_points) / total_weight
        
        return (weighted_x, weighted_y)
    
    def calculate_angle(self, point1, point2, point3):
        """
        Calculate angle between three points
        Args:
            point1, point2, point3: (x, y) coordinates
            point2 is the vertex
        Returns:
            float: angle in degrees
        """
        if None in [point1, point2, point3]:
            return None
        
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def analyze_posture(self, keypoints):
        """
        Enhanced posture analysis with multiple criteria
        
        Args:
            keypoints: Keypoint array with shape (17, 3) [x, y, confidence]
        
        Returns:
            str: 'sitting', 'standing', or 'unknown'
        """
        # Get keypoints with confidence scores
        left_shoulder = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_shoulder'])
        right_shoulder = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_shoulder'])
        left_hip = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_hip'])
        right_hip = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_hip'])
        left_knee = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_knee'])
        right_knee = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_knee'])
        left_ankle = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_ankle'])
        right_ankle = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_ankle'])
        
        # Calculate weighted center points
        shoulder_center = self.calculate_weighted_center([left_shoulder, right_shoulder])
        hip_center = self.calculate_weighted_center([left_hip, right_hip])
        knee_center = self.calculate_weighted_center([left_knee, right_knee])
        ankle_center = self.calculate_weighted_center([left_ankle, right_ankle])
        
        # If missing keypoints, try using single-side points
        if hip_center is None:
            hip_center = (left_hip[:2] if left_hip else None) or (right_hip[:2] if right_hip else None)
        if knee_center is None:
            knee_center = (left_knee[:2] if left_knee else None) or (right_knee[:2] if right_knee else None)
        if ankle_center is None:
            ankle_center = (left_ankle[:2] if left_ankle else None) or (right_ankle[:2] if right_ankle else None)
        
        # Must have hip, knee, ankle to make judgment
        if not (hip_center and knee_center and ankle_center):
            return 'unknown'
        
        # Calculate various distances and ratios
        hip_to_knee = self.calculate_distance(hip_center, knee_center)
        knee_to_ankle = self.calculate_distance(knee_center, ankle_center)
        hip_to_ankle = self.calculate_distance(hip_center, ankle_center)
        
        if hip_to_ankle is None or hip_to_ankle < 15:  # Adjust minimum distance threshold
            return 'unknown'
        
        # Multiple judgment criteria
        scores = {'sitting': 0, 'standing': 0}
        
        # 1. Distance ratio judgment
        hip_knee_ratio = hip_to_knee / hip_to_ankle if hip_to_ankle > 0 else 0
        if hip_knee_ratio < 0.45:  # Adjust threshold
            scores['sitting'] += 1
        elif hip_knee_ratio > 0.65:
            scores['standing'] += 1
        
        # 2. Vertical position relationship
        hip_knee_vertical = hip_center[1] - knee_center[1]  # Positive value means knee is below hip
        knee_ankle_vertical = knee_center[1] - ankle_center[1]  # Positive value means ankle is below knee
        
        # Sitting: knees are usually slightly below hips, but not too far
        if 0 < hip_knee_vertical < 80:  # Knee slightly lower than hip
            scores['sitting'] += 1
        elif hip_knee_vertical > 120:  # Knee far below hip
            scores['standing'] += 1
        
        # 3. Body compactness (sitting posture is more compact)
        body_compactness = hip_to_ankle / (hip_to_knee + knee_to_ankle) if (hip_to_knee + knee_to_ankle) > 0 else 0
        if body_compactness < 0.85:  # Body is more compact
            scores['sitting'] += 1
        elif body_compactness > 0.95:  # Body is more extended
            scores['standing'] += 1
        
        # 4. If shoulder information is available, add torso angle judgment
        if shoulder_center:
            torso_length = self.calculate_distance(shoulder_center, hip_center)
            leg_length = hip_to_ankle
            
            if torso_length and leg_length and leg_length > 0:
                torso_leg_ratio = torso_length / leg_length
                # When sitting, torso to leg ratio is larger
                if torso_leg_ratio > 0.7:
                    scores['sitting'] += 1
                elif torso_leg_ratio < 0.5:
                    scores['standing'] += 1
        
        # 5. Knee height relative judgment
        knee_height_ratio = knee_ankle_vertical / hip_to_ankle if hip_to_ankle > 0 else 0
        if knee_height_ratio < 0.3:  # Knee is relatively low
            scores['sitting'] += 1
        elif knee_height_ratio > 0.6:  # Knee is relatively high
            scores['standing'] += 1
        
        # Determine posture based on scores
        if scores['sitting'] > scores['standing']:
            return 'sitting'
        elif scores['standing'] > scores['sitting']:
            return 'standing'
        else:
            return 'unknown'
    
    def analyze_posture_with_angle(self, keypoints):
        """Enhanced angle-based posture analysis"""
        # Get keypoints with confidence
        left_shoulder = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_shoulder'])
        right_shoulder = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_shoulder'])
        left_hip = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_hip'])
        right_hip = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_hip'])
        left_knee = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_knee'])
        right_knee = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_knee'])
        left_ankle = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['left_ankle'])
        right_ankle = self.get_keypoint(keypoints, self.KEYPOINT_INDICES['right_ankle'])
        
        # Calculate weighted center points
        shoulder_center = self.calculate_weighted_center([left_shoulder, right_shoulder])
        hip_center = self.calculate_weighted_center([left_hip, right_hip])
        knee_center = self.calculate_weighted_center([left_knee, right_knee])
        ankle_center = self.calculate_weighted_center([left_ankle, right_ankle])
        
        # Fallback to single-side points
        if hip_center is None:
            hip_center = (left_hip[:2] if left_hip else None) or (right_hip[:2] if right_hip else None)
        if knee_center is None:
            knee_center = (left_knee[:2] if left_knee else None) or (right_knee[:2] if right_knee else None)
        if ankle_center is None:
            ankle_center = (left_ankle[:2] if left_ankle else None) or (right_ankle[:2] if right_ankle else None)
        
        if not (hip_center and knee_center and ankle_center):
            return 'unknown'
        
        scores = {'sitting': 0, 'standing': 0}
        
        # 1. Hip-knee-ankle angle
        hip_knee_ankle_angle = self.calculate_angle(hip_center, knee_center, ankle_center)
        if hip_knee_ankle_angle is not None:
            if 80 <= hip_knee_ankle_angle <= 140:  # More precise sitting angle range
                scores['sitting'] += 2
            elif hip_knee_ankle_angle >= 150:  # Standing angle
                scores['standing'] += 2
            elif 140 < hip_knee_ankle_angle < 150:  # Middle range, slightly toward standing
                scores['standing'] += 1
        
        # 2. If shoulders are available, calculate torso angle
        if shoulder_center and hip_center and knee_center:
            # Shoulder-hip-knee angle
            shoulder_hip_knee_angle = self.calculate_angle(shoulder_center, hip_center, knee_center)
            if shoulder_hip_knee_angle is not None:
                if 60 <= shoulder_hip_knee_angle <= 110:  # Sitting torso bent
                    scores['sitting'] += 1
                elif shoulder_hip_knee_angle >= 140:  # Standing torso upright
                    scores['standing'] += 1
        
        # 3. Leg bending degree (calculate both legs separately)
        angles = []
        for hip, knee, ankle in [(left_hip, left_knee, left_ankle), (right_hip, right_knee, right_ankle)]:
            if all(p is not None for p in [hip, knee, ankle]):
                angle = self.calculate_angle(hip[:2], knee[:2], ankle[:2])
                if angle is not None:
                    angles.append(angle)
        
        if angles:
            avg_angle = np.mean(angles)
            if avg_angle < 135:  # Legs are more bent
                scores['sitting'] += 1
            elif avg_angle > 155:  # Legs are straighter
                scores['standing'] += 1
        
        # Determine based on scores
        if scores['sitting'] > scores['standing']:
            return 'sitting'
        elif scores['standing'] > scores['sitting']:
            return 'standing'
        else:
            return 'unknown'
    
    def analyze_posture_hybrid(self, keypoints):
        """
        Enhanced hybrid posture analysis with confidence weighting
        
        Args:
            keypoints: Keypoint array with shape (17, 3) [x, y, confidence]
        
        Returns:
            str: 'sitting', 'standing', or 'unknown'
        """
        # Calculate average confidence of keypoints
        key_indices = [self.KEYPOINT_INDICES[key] for key in ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']]
        confidences = []
        for idx in key_indices:
            if idx < len(keypoints) and keypoints[idx][2] > self.min_required_confidence:
                confidences.append(keypoints[idx][2])
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Execute two analysis methods
        distance_result = self.analyze_posture(keypoints)
        angle_result = self.analyze_posture_with_angle(keypoints)
        
        # If both methods get the same result, return directly
        if distance_result == angle_result and distance_result != 'unknown':
            return distance_result
        
        # If methods are inconsistent, use voting mechanism
        results = [distance_result, angle_result]
        valid_results = [r for r in results if r != 'unknown']
        
        if not valid_results:
            return 'unknown'
        
        # Adjust strategy based on confidence
        if avg_confidence > 0.7:  # High confidence, trust distance method more
            if distance_result != 'unknown':
                return distance_result
            else:
                return angle_result
        elif avg_confidence > 0.5:  # Medium confidence, comprehensive judgment
            if len(valid_results) == 1:
                return valid_results[0]
            # Both results are valid but different, prefer distance method
            return distance_result if distance_result != 'unknown' else angle_result
        else:  # Low confidence, more conservative
            if distance_result == angle_result:
                return distance_result
            else:
                return 'unknown'  # Return unknown when low confidence and results are inconsistent

model_path = Path(__file__).parent.parent / 'models' / 'yolov8n-pose.pt'

# Load YOLO Pose model
model = YOLO(model_path)  # Use pose model

# Open camera
cap = cv2.VideoCapture(0)

# For storing tracking trajectories and postures
track_history = defaultdict(lambda: [])
posture_history = defaultdict(lambda: [])

# Create posture analyzer with optimized threshold
posture_analyzer = PostureAnalyzer(sitting_threshold=0.42)

# Statistics data
stats = {
    'sitting': 0,
    'standing': 0,
    'unknown': 0
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Use pose model for tracking
    results = model.track(frame, conf=0.5, persist=True, verbose=False)
    
    # Get detection results
    if results[0].boxes.id is not None and results[0].keypoints is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.data.cpu().numpy()  # shape: (N, 17, 3)
        
        # Reset statistics data, prepare to recalculate current frame statistics
        current_frame_stats = {
            'sitting': 0,
            'standing': 0,
            'unknown': 0
        }
        
        for box, track_id, kpts in zip(boxes, track_ids, keypoints):
            x1, y1, x2, y2 = box
            
            # Calculate center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Analyze posture using hybrid method (primary: distance-based, auxiliary: angle-based)
            posture = posture_analyzer.analyze_posture_hybrid(kpts)
            
            # Store posture history (for stable judgment)
            posture_history[track_id].append(posture)
            if len(posture_history[track_id]) > 20:  # Increase history length to 20
                posture_history[track_id].pop(0)
            
            # Use weighted voting and time decay to determine final posture
            recent_postures = posture_history[track_id]
            if len(recent_postures) >= 3:  # At least 3 samples are needed
                # Give higher weight to recent judgments
                weights = np.exp(np.linspace(-1, 0, len(recent_postures)))  # Exponential decay weights
                
                posture_scores = {'sitting': 0, 'standing': 0, 'unknown': 0}
                for i, p in enumerate(recent_postures):
                    posture_scores[p] += weights[i]
                
                # Only change judgment when there is clear advantage
                max_score = max(posture_scores.values())
                if max_score > 0:
                    final_posture = max(posture_scores, key=posture_scores.get)
                    
                    # If the difference between highest and second highest score is small, keep unknown
                    sorted_scores = sorted(posture_scores.values(), reverse=True)
                    if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.3:
                        final_posture = 'unknown'
                else:
                    final_posture = 'unknown'
            else:
                final_posture = posture  # Use current judgment when samples are insufficient
            
            # Accumulate current frame statistics
            current_frame_stats[final_posture] += 1
            
            # Draw trajectory
            track = track_history[track_id]
            track.append((center_x, center_y))
            if len(track) > 30:
                track.pop(0)
            
            points = [(int(x), int(y)) for x, y in track]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
            
            # Choose color based on posture
            color = (0, 255, 0) if final_posture == 'standing' else (0, 0, 255)
            if final_posture == 'unknown':
                color = (128, 128, 128)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Display ID and posture
            label = f'ID:{track_id} {final_posture.upper()}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update global statistics
        stats.update(current_frame_stats)
        
        # Clean up history records of tracking IDs that no longer exist (avoid memory leaks)
        current_track_ids = set(track_ids)
        old_track_ids = set(posture_history.keys()) - current_track_ids
        for old_id in old_track_ids:
            if old_id in posture_history:
                del posture_history[old_id]
            if old_id in track_history:
                del track_history[old_id]
        
        person_count = len(track_ids)
    else:
        # When no personnel are detected, reset statistics
        person_count = 0
        stats = {
            'sitting': 0,
            'standing': 0,
            'unknown': 0
        }
    
    # Draw keypoint skeleton
    annotated_frame = results[0].plot()
    
    # Display statistics information
    y_offset = 30
    cv2.putText(annotated_frame, f'Total: {person_count}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    y_offset += 40
    cv2.putText(annotated_frame, f'Standing: {stats["standing"]}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_offset += 35
    cv2.putText(annotated_frame, f'Sitting: {stats["sitting"]}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_offset += 35
    cv2.putText(annotated_frame, f'Unknown: {stats["unknown"]}', 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    
    # Verify statistics consistency (for debugging)
    total_classified = stats["sitting"] + stats["standing"] + stats["unknown"]
    if person_count > 0 and total_classified != person_count:
        y_offset += 35
        cv2.putText(annotated_frame, f'WARNING: Count mismatch!', 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow('People Tracker with Posture Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
