"""
Factory for creating calibration instances with auto-fallback.
"""

from typing import Optional
import numpy as np

from var_offside.calibration.base import BaseCalibration, CalibrationResult
from var_offside.calibration.pnl_calibration import PnLCalibration
from var_offside.calibration.vanishing_point import VanishingPointCalibration
from var_offside.calibration.manual_calibration import ManualCalibration


class AutoCalibration(BaseCalibration):
    """
    Auto calibration with fallback chain:
    1. Try PnLCalib (deep learning, best accuracy)
    2. Fall back to Vanishing Point (classical CV)
    3. Fall back to Manual (user-assisted)
    """
    
    def __init__(self, min_quality: float = 0.3):
        """
        Initialize auto calibration.
        
        Args:
            min_quality: Minimum quality score to accept a calibration
        """
        self.min_quality = min_quality
        self.pnl = PnLCalibration()
        self.vp = VanishingPointCalibration()
        self._last_method = None
    
    def get_name(self) -> str:
        return "auto"
    
    def calibrate(self, image: np.ndarray) -> CalibrationResult:
        """
        Calibrate with auto fallback.
        
        Args:
            image: BGR input image
            
        Returns:
            CalibrationResult from best available method
        """
        # Try PnLCalib first
        if self.pnl._check_pnl_available():
            print("[AutoCalib] Trying PnLCalib...")
            result = self.pnl.calibrate(image)
            
            if result.is_valid and result.quality_score >= self.min_quality:
                print(f"[AutoCalib] PnLCalib succeeded (quality: {result.quality_score:.2f})")
                self._last_method = "pnl"
                return result
            else:
                print(f"[AutoCalib] PnLCalib failed or low quality, trying fallback...")
        else:
            print("[AutoCalib] PnLCalib not available, using classical method...")
        
        # Fall back to vanishing point
        print("[AutoCalib] Trying Vanishing Point method...")
        result = self.vp.calibrate(image)
        
        if result.is_valid:
            print(f"[AutoCalib] Vanishing Point succeeded (quality: {result.quality_score:.2f})")
            self._last_method = "vanishing_point"
            return result
        
        # Return empty result if all methods fail
        print("[AutoCalib] All automatic methods failed")
        self._last_method = None
        return CalibrationResult(method="auto_failed")


def get_calibrator(method: str = "auto") -> BaseCalibration:
    """
    Get a calibration instance by method name.
    
    Args:
        method: Calibration method name
            - "pnl" or "pnlcalib": PnLCalib deep learning method
            - "vanishing" or "vanishing_point" or "vp": Classical CV method
            - "manual": Interactive manual calibration
            - "auto": Try PnLCalib, fall back to vanishing point
            
    Returns:
        Calibration instance
        
    Raises:
        ValueError: If method is unknown
    """
    method = method.lower().strip()
    
    if method in ["pnl", "pnlcalib", "deep"]:
        return PnLCalibration()
    
    elif method in ["vanishing", "vanishing_point", "vp", "classical", "cv"]:
        return VanishingPointCalibration()
    
    elif method in ["manual", "interactive"]:
        return ManualCalibration()
    
    elif method == "auto":
        return AutoCalibration()
    
    else:
        raise ValueError(
            f"Unknown calibration method: {method}. "
            f"Valid options: pnl, vanishing_point, manual, auto"
        )


def get_available_methods() -> list:
    """Get list of available calibration methods."""
    methods = ["vanishing_point", "manual", "auto"]
    
    # Check if PnLCalib is available
    pnl = PnLCalibration()
    if pnl._check_pnl_available():
        methods.insert(0, "pnl")
    
    return methods
