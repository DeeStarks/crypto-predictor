import logging
import json
from datetime import datetime
from utils.type_conversion import convert_to_native_types

logger = logging.getLogger(__name__)


class Predictor:
    """Generates and formats predictions."""
    
    def __init__(self, model_trainer, confidence_threshold=0.6, output_format='text'):
        """
        Initialize predictor.
        
        Args:
            model_trainer: Trained model trainer instance
            confidence_threshold: Minimum confidence for valid predictions
            output_format: Output format (text, json, csv)
        """
        self.model_trainer = model_trainer
        self.confidence_threshold = confidence_threshold
        self.output_format = output_format
        self.prediction_history = []
        
        logger.info(f"Initialized predictor with {confidence_threshold} confidence threshold")
    
    def predict(self, features, window_info=None):
        """
        Make a prediction with confidence filtering.
        
        Args:
            features: Feature dictionary
            window_info: Optional window information for context
            
        Returns:
            dict: Prediction result or None if below confidence threshold
        """
        prediction = self.model_trainer.predict(features)
        
        if prediction is None:
            logger.warning("Model prediction failed")
            return None
        
        prediction['timestamp'] = datetime.now().isoformat()
        if window_info:
            prediction['window_info'] = window_info
        
        if prediction['confidence'] < self.confidence_threshold:
            logger.info(
                f"Prediction confidence {prediction['confidence']:.3f} below threshold "
                f"{self.confidence_threshold} - skipping"
            )
            prediction['action'] = 'skip'
            prediction['reason'] = 'low_confidence'
        else:
            prediction['action'] = 'trade'
            logger.info(
                f"Prediction: {prediction['direction'].upper()} "
                f"(confidence: {prediction['confidence']:.3f})"
            )
        
        self.prediction_history.append(prediction)
        
        return prediction
    
    def format_output(self, prediction):
        """
        Format prediction for output.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            str: Formatted output
        """
        if self.output_format == 'json':
            return self._format_json(prediction)
        elif self.output_format == 'csv':
            return self._format_csv(prediction)
        else:
            return self._format_text(prediction)
    
    def _format_text(self, prediction):
        """Format as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"PREDICTION - {prediction['timestamp']}")
        lines.append("=" * 60)
        
        if 'window_info' in prediction:
            info = prediction['window_info']
            lines.append(f"Window: {info.get('window_id', 'N/A')}")
            lines.append(f"Current Price: ${info.get('current_price', 'N/A'):.2f}")
        
        lines.append("")
        lines.append(f"Direction: {prediction['direction'].upper()}")
        lines.append(f"Confidence: {prediction['confidence']:.1%}")
        lines.append(f"Probability Up: {prediction['probability_up']:.1%}")
        
        if prediction.get('action') == 'skip':
            lines.append("")
            lines.append(f"⚠️  Action: SKIP - {prediction.get('reason', 'unknown reason')}")
        else:
            lines.append("")
            lines.append(f"✓ Action: {prediction['action'].upper()}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_json(self, prediction):
        return json.dumps(convert_to_native_types(prediction), indent=2)
    
    def _format_csv(self, prediction):
        fields = [
            prediction['timestamp'],
            prediction.get('window_info', {}).get('window_id', ''),
            prediction['direction'],
            f"{prediction['confidence']:.4f}",
            f"{prediction['probability_up']:.4f}",
            prediction.get('action', ''),
            prediction.get('reason', '')
        ]
        return ",".join(str(f) for f in fields)
    
    def get_csv_header(self):
        return "timestamp,window_id,direction,confidence,probability_up,action,reason"
    
    def evaluate_predictions(self, actual_outcomes):
        """
        Evaluate prediction accuracy.
        
        Args:
            actual_outcomes: List of tuples (prediction_index, actual_direction)
            
        Returns:
            dict: Evaluation metrics
        """
        if not actual_outcomes:
            return {}
        
        correct = 0
        total = 0
        
        for pred_idx, actual_dir in actual_outcomes:
            if pred_idx < len(self.prediction_history):
                pred = self.prediction_history[pred_idx]
                if pred.get('action') == 'trade':
                    total += 1
                    if pred['direction'] == actual_dir:
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            'total_predictions': len(self.prediction_history),
            'traded_predictions': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'avg_confidence': sum(
                p['confidence'] for p in self.prediction_history 
                if p.get('action') == 'trade'
            ) / total if total > 0 else 0
        }
        
        logger.info(f"Prediction accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return metrics
    
    def get_recent_performance(self, n=20):
        """Get performance stats for recent predictions."""
        if len(self.prediction_history) < 2:
            return None
        
        recent = self.prediction_history[-n:]
        
        return {
            'total': len(recent),
            'traded': sum(1 for p in recent if p.get('action') == 'trade'),
            'skipped': sum(1 for p in recent if p.get('action') == 'skip'),
            'avg_confidence': sum(p['confidence'] for p in recent) / len(recent),
            'direction_distribution': {
                'up': sum(1 for p in recent if p['direction'] == 'up'),
                'down': sum(1 for p in recent if p['direction'] == 'down')
            }
        }
    
    def save_predictions(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(convert_to_native_types(self.prediction_history), f, indent=2)
        
        logger.info(f"Saved {len(self.prediction_history)} predictions to {filepath}")
    
    def load_predictions(self, filepath):
        with open(filepath, 'r') as f:
            self.prediction_history = json.load(f)
        
        logger.info(f"Loaded {len(self.prediction_history)} predictions from {filepath}")
