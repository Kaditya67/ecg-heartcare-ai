import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
import joblib
import os

class ECGXGBoostWrapper:
    """
    Wrapper so XGBoost model behaves similarly to PyTorch model for inference
    """

    def __init__(self, model_path=None, label_encoder_path=None, **kwargs):
        self.model = None
        if model_path:
            self.load_model(model_path)

        self.label_encoder = None
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")

        # XGBoost JSON/UBJ/Binary Model
        if model_path.endswith((".json", ".ubj", ".pth", ".model")):
            self.model = xgb.Booster()
            self.model.load_model(model_path)
        else:
            # Try loading anyway if it exists
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            except:
                raise ValueError("Unsupported XGBoost format or corrupted file.")

    def __call__(self, input_tensor):
        """
        Mimic PyTorch model call
        input_tensor: (batch, length) or (batch, 1, length)
        """
        # Convert to numpy and flatten for XGBoost
        x = input_tensor.cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        
        dtest = xgb.DMatrix(x)
        preds = self.model.predict(dtest)
        
        # Convert back to torch tensor logits
        return torch.tensor(preds)

class AdaBoostECGWrapper:
    """
    Wrapper class to make AdaBoost behave similar to PyTorch/XGBoost inference interface.
    """

    def __init__(self, model_path=None, label_encoder_path=None, **kwargs):
        self.model = None

        if model_path:
            self.load_model(model_path)

        self.label_encoder = None
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AdaBoost model not found: {model_path}")

        if model_path.endswith(".pkl"):
            self.model = joblib.load(model_path)
        else:
            raise ValueError("Unsupported AdaBoost format. Use .pkl")

    def __call__(self, input_tensor):
        """
        Mimic PyTorch model call
        """
        x = input_tensor.cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        # AdaBoost predict usually returns classes, but we need probabilities/logits for compatibility
        try:
            # Try to get probabilities
            probs = self.model.predict_proba(x)
            return torch.tensor(probs)
        except:
            # Fallback to one-hot if no probabilities
            preds = self.model.predict(x)
            # This is a bit hacky but works for argmax compatibility
            return torch.tensor(np.eye(10)[preds]) # Assuming 10 classes max


class RandomForestECGWrapper:
    """
    Wrapper for sklearn RandomForestClassifier, following the same inference
    interface as XGBoost and AdaBoost wrappers.

    The RF model predicts binary classes: 0 = Normal, 1 = Abnormal.
    Returns a 2-column probability tensor [P(Normal), P(Abnormal)] compatible
    with torch.argmax() for class selection.
    """

    def __init__(self, model_path=None, **kwargs):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        self._num_classes = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Random Forest model not found: {model_path}")

        if model_path.endswith(".pkl"):
            self.model = joblib.load(model_path)
            # Discover class count from loaded model
            self._num_classes = len(self.model.classes_) if hasattr(self.model, 'classes_') else 2
        else:
            raise ValueError("Unsupported format. Random Forest requires a .pkl file.")

    def eval(self):
        """No-op — sklearn models don't have training/eval modes."""
        return self

    def load_state_dict(self, *args, **kwargs):
        """No-op — sklearn models don't use state dicts."""
        pass

    def __call__(self, input_tensor):
        """
        Args:
            input_tensor: torch.Tensor of shape (batch, length) or (batch, 1, length)
        Returns:
            torch.Tensor of shape (batch, num_classes) — probability distributions
        """
        x = input_tensor.cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)  # Flatten 3D to 2D

        try:
            probs = self.model.predict_proba(x)  # (batch, num_classes)
            return torch.tensor(probs, dtype=torch.float32)
        except Exception as e:
            # Fallback: use hard predictions and scatter to one-hot
            preds = self.model.predict(x)
            n_classes = self._num_classes or 2
            one_hot = np.eye(n_classes)[preds]
            return torch.tensor(one_hot, dtype=torch.float32)
