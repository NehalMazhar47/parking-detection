"""
parking_rules.py
----------------
Parking restriction engine.
Given a detected vehicle label and the configured allowed vehicle type,
determines whether the vehicle is permitted or unauthorized.
"""


class ParkingRuleEngine:
    """
    Simple rule engine: one allowed vehicle type, everything else is blocked.

    The allowed_vehicle is read from config.yaml at startup.
    Changing config.yaml and restarting the app updates the rule instantly —
    no code change needed.
    """

    def __init__(self, allowed_vehicle: str):
        """
        Args:
            allowed_vehicle: The vehicle label that IS allowed to park
                             (e.g. "car", "motorcycle", "bus", "truck", "bicycle").
        """
        self.allowed_vehicle = allowed_vehicle.lower().strip()
        print(f"[ParkingRules] Allowed vehicle type: '{self.allowed_vehicle}'")

    def check(self, detection: dict) -> dict:
        """
        Evaluate a single detection against the parking rule.

        Args:
            detection: A detection dict from VehicleDetector.detect().

        Returns:
            The same dict augmented with:
                - status   "allowed" | "unauthorized"
                - allowed  bool
        """
        label = detection["label"].lower().strip()
        is_allowed = (label == self.allowed_vehicle)

        return {
            **detection,
            "status":  "allowed" if is_allowed else "unauthorized",
            "allowed": is_allowed,
        }

    def evaluate_frame(self, detections: list[dict]) -> list[dict]:
        """
        Evaluate all detections in a frame.

        Returns:
            List of evaluated detection dicts.
        """
        return [self.check(d) for d in detections]

    def has_unauthorized(self, evaluated: list[dict]) -> bool:
        """Return True if any detection in the list is unauthorized."""
        return any(not d["allowed"] for d in evaluated)
