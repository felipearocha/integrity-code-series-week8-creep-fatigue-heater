"""
Cybersecurity architecture for fired heater tube integrity monitoring.

STRIDE threat model:
  S - Spoofing: Tube metal temperature (TMT) sensor identity
  T - Tampering: TMT readings, creep model parameters, oxide data
  R - Repudiation: Unsigned inspection decisions, unlogged overrides
  I - Information Disclosure: Proprietary tube metallurgy data leakage
  D - Denial of Service: DCS/historian communication disruption
  E - Elevation of Privilege: Unauthorized modification of alarm limits

Mitigation:
  - SHA-256 hash chain for all simulation inputs/outputs
  - Sensor redundancy validation (TMT cross-check)
  - Tamper-evident audit log
  - Role-based access for alarm limit changes
"""

import hashlib
import json
import time as time_module
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class AuditEntry:
    """Single entry in the hash-chain audit log."""
    timestamp: float
    event_type: str  # "simulation", "input_validation", "alarm", "override"
    description: str
    data_hash: str
    previous_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry (excluding entry_hash)."""
        payload = (
            f"{self.timestamp}|{self.event_type}|{self.description}|"
            f"{self.data_hash}|{self.previous_hash}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditChain:
    """SHA-256 hash-chain audit log for integrity verification."""

    def __init__(self):
        self.entries: List[AuditEntry] = []
        self._genesis_hash = hashlib.sha256(b"ICS2_WEEK8_GENESIS").hexdigest()

    def add_entry(self, event_type: str, description: str,
                  data: dict) -> AuditEntry:
        """
        Add a new entry to the chain.

        Parameters
        ----------
        event_type : str
        description : str
        data : dict
            Data to hash.

        Returns
        -------
        AuditEntry
        """
        now = time_module.time()
        data_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        prev_hash = (self.entries[-1].entry_hash
                     if self.entries else self._genesis_hash)

        entry = AuditEntry(
            timestamp=now,
            event_type=event_type,
            description=description,
            data_hash=data_hash,
            previous_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire chain.

        Returns
        -------
        bool
            True if chain is intact.
        """
        for i, entry in enumerate(self.entries):
            # Verify entry hash
            if entry.entry_hash != entry.compute_hash():
                return False
            # Verify chain linkage
            expected_prev = (self.entries[i - 1].entry_hash
                             if i > 0 else self._genesis_hash)
            if entry.previous_hash != expected_prev:
                return False
        return True

    def export_log(self) -> List[dict]:
        """Export chain as list of dicts."""
        return [asdict(e) for e in self.entries]


@dataclass
class ThreatAssessment:
    """STRIDE threat assessment for a single threat."""
    category: str  # S, T, R, I, D, E
    threat: str
    likelihood: str  # "low", "medium", "high"
    impact: str  # "low", "medium", "high", "critical"
    mitigation: str
    residual_risk: str


def build_stride_model() -> List[ThreatAssessment]:
    """
    Build STRIDE threat model for fired heater tube monitoring system.

    Returns
    -------
    list of ThreatAssessment
    """
    return [
        ThreatAssessment(
            category="Spoofing",
            threat="Attacker replaces TMT thermocouple signal with "
                   "synthetic data showing normal temperatures during overheat",
            likelihood="medium",
            impact="critical",
            mitigation="Triple-redundant TMT sensors with 2-of-3 voting logic; "
                       "cross-validate TMT against process outlet temperature "
                       "and firebox heat duty calculation",
            residual_risk="low",
        ),
        ThreatAssessment(
            category="Tampering",
            threat="Modification of Omega parameter or LMP coefficients in "
                   "creep life model to extend apparent remaining life",
            likelihood="medium",
            impact="critical",
            mitigation="SHA-256 hash chain on all model parameters; "
                       "read-only model configuration with change management; "
                       "parameter drift detection against API 530 reference values",
            residual_risk="low",
        ),
        ThreatAssessment(
            category="Tampering",
            threat="Manipulation of oxide thickness measurement data to "
                   "understate wall thinning rate",
            likelihood="low",
            impact="high",
            mitigation="Automated cross-check: measured wall thickness from UT "
                       "vs. predicted wall from oxidation model; flag deviations > 10%",
            residual_risk="low",
        ),
        ThreatAssessment(
            category="Repudiation",
            threat="Operator overrides high-TMT alarm without logging; "
                   "no record of who authorized continued operation at "
                   "temperatures above API 530 design",
            likelihood="high",
            impact="critical",
            mitigation="Immutable audit log (hash chain) for all alarm "
                       "acknowledgments and overrides; dual authorization for "
                       "TMT alarm suppression; automatic escalation after 3 overrides",
            residual_risk="medium",
        ),
        ThreatAssessment(
            category="Information Disclosure",
            threat="Proprietary tube metallurgy data (Omega parameters from "
                   "ex-service testing) exfiltrated from DCS historian",
            likelihood="low",
            impact="medium",
            mitigation="Network segmentation between OT and IT; "
                       "encrypted data at rest and in transit; "
                       "role-based access to creep model database",
            residual_risk="low",
        ),
        ThreatAssessment(
            category="Denial of Service",
            threat="DCS-historian communication disruption prevents "
                   "real-time TMT logging during startup transient "
                   "(Marathon Martinez scenario)",
            likelihood="medium",
            impact="critical",
            mitigation="Local TMT data buffer on field instruments (72hr); "
                       "independent safety-rated TMT trip system on SIS; "
                       "manual TMT logging protocol activated on comms loss",
            residual_risk="medium",
        ),
        ThreatAssessment(
            category="Elevation of Privilege",
            threat="Unauthorized modification of TMT alarm high-high setpoint "
                   "to suppress safety trips during startup",
            likelihood="medium",
            impact="critical",
            mitigation="SIS alarm limits in read-only safety controller; "
                       "Management of Change (MOC) required for any alarm "
                       "limit modification; weekly automated audit of "
                       "alarm setpoints against approved values",
            residual_risk="low",
        ),
    ]


def validate_sensor_inputs(tmt_readings: list,
                            process_outlet_T: float,
                            heat_duty_kW: float) -> dict:
    """
    Validate TMT sensor readings against process cross-checks.

    Parameters
    ----------
    tmt_readings : list of float
        TMT readings from redundant sensors [K].
    process_outlet_T : float
        Process outlet temperature [K].
    heat_duty_kW : float
        Firebox heat duty [kW].

    Returns
    -------
    dict with keys:
        'valid': bool
        'median_tmt': float
        'spread': float
        'alerts': list of str
    """
    readings = [r for r in tmt_readings if r is not None and r > 0]
    if len(readings) < 2:
        return {
            "valid": False,
            "median_tmt": None,
            "spread": None,
            "alerts": ["Insufficient TMT sensor redundancy (< 2 valid readings)"],
        }

    median_tmt = float(np.median(readings)) if len(readings) > 0 else 0.0
    spread = max(readings) - min(readings)

    alerts = []
    # Check sensor agreement
    if spread > 25.0:
        alerts.append(
            f"TMT sensor spread {spread:.1f} K exceeds 25 K threshold; "
            "possible sensor drift or spoofing"
        )

    # Cross-check: TMT should be higher than process outlet
    if median_tmt < process_outlet_T:
        alerts.append(
            "TMT below process outlet temperature; physically inconsistent"
        )

    # Cross-check: TMT vs heat duty (simplified)
    expected_tmt_rise = heat_duty_kW * 0.08  # [ASSUMED] K per kW, simplified
    if median_tmt - process_outlet_T > expected_tmt_rise * 2.0:
        alerts.append(
            "TMT significantly above expected for current heat duty; "
            "possible hot spot or flow maldistribution"
        )

    return {
        "valid": len(alerts) == 0,
        "median_tmt": median_tmt,
        "spread": spread,
        "alerts": alerts,
    }


import numpy as np  # noqa: E402 — needed for sensor validation below
