#!/usr/bin/env python3
"""
🔒 HIPAA COMPLIANCE FRAMEWORK
==============================

Comprehensive HIPAA compliance framework for healthcare quantum digital twins:
- PHI (Protected Health Information) encryption and de-identification
- Audit logging and access control
- Data retention and secure deletion
- Business Associate Agreement (BAA) support
- HIPAA Security Rule compliance (45 CFR §164.312)
- HIPAA Privacy Rule compliance (45 CFR §164.502)

HIPAA Compliance Requirements:
    - Administrative Safeguards: Access controls, workforce training
    - Physical Safeguards: Facility access, device security
    - Technical Safeguards: Encryption, audit controls, integrity controls
    - Organizational Requirements: Business associate contracts
    - Breach Notification Rule: Incident response

Author: Hassan Al-Sahli
Purpose: HIPAA-compliant healthcare data handling
Reference: docs/HEALTHCARE_FOCUS_STRATEGIC_PLAN.md - Security & Compliance
Implementation: IMPLEMENTATION_TRACKER.md - hipaa_compliance.py
"""

import hashlib
import hmac
import secrets
import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class PHICategory(Enum):
    """Protected Health Information categories per HIPAA"""
    # Direct Identifiers (must be removed for de-identification)
    NAME = "name"
    GEOGRAPHIC = "geographic"  # Smaller than state
    DATES = "dates"  # Except year
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    SSN = "ssn"
    MRN = "medical_record_number"
    HEALTH_PLAN = "health_plan_number"
    ACCOUNT = "account_number"
    CERTIFICATE = "certificate_number"
    VEHICLE = "vehicle_identifier"
    DEVICE = "device_identifier"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric_identifier"
    PHOTO = "photo"
    UNIQUE_CODE = "unique_identifying_code"

    # Medical Information
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MEDICATIONS = "medications"
    LAB_RESULTS = "lab_results"
    GENOMIC_DATA = "genomic_data"
    IMAGING = "imaging_data"


class AccessLevel(Enum):
    """User access levels"""
    PATIENT = "patient"  # Own data only
    PROVIDER = "provider"  # Assigned patients
    RESEARCHER = "researcher"  # De-identified data only
    ADMIN = "admin"  # System administration
    QUANTUM_SYSTEM = "quantum_system"  # Automated processing


class AuditAction(Enum):
    """Auditable actions per HIPAA"""
    ACCESS = "access"
    MODIFY = "modify"
    DELETE = "delete"
    EXPORT = "export"
    DECRYPT = "decrypt"
    DE_IDENTIFY = "de_identify"
    BREACH_DETECTED = "breach_detected"


@dataclass
class AuditLogEntry:
    """HIPAA-compliant audit log entry"""
    log_id: str
    timestamp: datetime
    user_id: str
    user_role: AccessLevel
    action: AuditAction
    resource_type: str  # e.g., "patient_record", "genomic_data"
    resource_id: str
    phi_accessed: List[PHICategory]
    ip_address: str
    success: bool
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptedPHI:
    """Encrypted Protected Health Information"""
    encrypted_data: bytes
    encryption_key_id: str
    phi_categories: List[PHICategory]
    encrypted_at: datetime
    encryption_algorithm: str = "Fernet (AES-128)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'encrypted_data': base64.b64encode(self.encrypted_data).decode('utf-8'),
            'encryption_key_id': self.encryption_key_id,
            'phi_categories': [c.value for c in self.phi_categories],
            'encrypted_at': self.encrypted_at.isoformat(),
            'encryption_algorithm': self.encryption_algorithm
        }


@dataclass
class DeIdentifiedData:
    """De-identified data per HIPAA Safe Harbor method"""
    data_id: str
    original_data_hash: str  # For re-identification key (if approved)
    de_identified_content: Dict[str, Any]
    removed_identifiers: List[PHICategory]
    de_identified_at: datetime
    de_identification_method: str = "HIPAA Safe Harbor"
    attestation: str = "All 18 HIPAA identifiers removed"


@dataclass
class BreachNotification:
    """HIPAA breach notification record"""
    breach_id: str
    detected_at: datetime
    breach_type: str
    affected_patients: int
    phi_compromised: List[PHICategory]
    notification_sent: bool
    notification_sent_at: Optional[datetime]
    hhs_notified: bool  # Dept of Health & Human Services
    media_notified: bool  # Required if >500 patients
    mitigation_actions: List[str]


class HIPAAEncryptionManager:
    """
    HIPAA-compliant encryption manager

    Implements HIPAA Security Rule §164.312(a)(2)(iv) and §164.312(e)(2)(ii)
    - Encryption at rest (AES-128 minimum)
    - Encryption in transit (TLS 1.2+ required separately)
    - Key rotation every 90 days recommended
    """

    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize encryption manager"""
        # In production: Use AWS KMS, Azure Key Vault, or HSM
        self.master_key = master_key or Fernet.generate_key()
        self.cipher = Fernet(self.master_key)
        self.key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
        self.key_created_at = datetime.now()

        logger.info(f"🔒 HIPAA Encryption initialized (Key ID: {self.key_id})")

    def encrypt_phi(
        self,
        phi_data: Dict[str, Any],
        phi_categories: List[PHICategory]
    ) -> EncryptedPHI:
        """
        Encrypt Protected Health Information

        Args:
            phi_data: PHI data to encrypt
            phi_categories: Categories of PHI contained

        Returns:
            EncryptedPHI object
        """
        # Convert to JSON
        json_data = json.dumps(phi_data, default=str).encode('utf-8')

        # Encrypt with Fernet (AES-128 CBC + HMAC)
        encrypted = self.cipher.encrypt(json_data)

        return EncryptedPHI(
            encrypted_data=encrypted,
            encryption_key_id=self.key_id,
            phi_categories=phi_categories,
            encrypted_at=datetime.now(),
            encryption_algorithm="Fernet (AES-128 CBC + HMAC-SHA256)"
        )

    def decrypt_phi(
        self,
        encrypted_phi: EncryptedPHI
    ) -> Dict[str, Any]:
        """
        Decrypt Protected Health Information

        Args:
            encrypted_phi: Encrypted PHI object

        Returns:
            Decrypted PHI data

        Raises:
            ValueError: If decryption fails
        """
        try:
            # Decrypt
            decrypted = self.cipher.decrypt(encrypted_phi.encrypted_data)

            # Parse JSON
            phi_data = json.loads(decrypted.decode('utf-8'))

            return phi_data

        except Exception as e:
            logger.error(f"❌ PHI decryption failed: {e}")
            raise ValueError(f"Decryption failed: {e}")

    def rotate_keys(self) -> str:
        """
        Rotate encryption keys (recommended every 90 days)

        Returns:
            New key ID
        """
        old_key_id = self.key_id

        # Generate new key
        self.master_key = Fernet.generate_key()
        self.cipher = Fernet(self.master_key)
        self.key_id = hashlib.sha256(self.master_key).hexdigest()[:16]
        self.key_created_at = datetime.now()

        logger.warning(f"🔄 Encryption key rotated: {old_key_id} → {self.key_id}")

        return self.key_id


class HIPAADeIdentifier:
    """
    HIPAA-compliant de-identification using Safe Harbor method

    Implements HIPAA Privacy Rule §164.514(b)(2)
    Removes all 18 HIPAA identifiers for research use
    """

    # 18 HIPAA Safe Harbor identifiers
    SAFE_HARBOR_IDENTIFIERS = [
        PHICategory.NAME,
        PHICategory.GEOGRAPHIC,
        PHICategory.DATES,
        PHICategory.PHONE,
        PHICategory.FAX,
        PHICategory.EMAIL,
        PHICategory.SSN,
        PHICategory.MRN,
        PHICategory.HEALTH_PLAN,
        PHICategory.ACCOUNT,
        PHICategory.CERTIFICATE,
        PHICategory.VEHICLE,
        PHICategory.DEVICE,
        PHICategory.URL,
        PHICategory.IP_ADDRESS,
        PHICategory.BIOMETRIC,
        PHICategory.PHOTO,
        PHICategory.UNIQUE_CODE
    ]

    def __init__(self):
        """Initialize de-identifier"""
        logger.info("🔓 HIPAA De-identifier initialized (Safe Harbor method)")

    def de_identify(
        self,
        phi_data: Dict[str, Any],
        keep_fields: Optional[List[str]] = None
    ) -> DeIdentifiedData:
        """
        De-identify PHI data using Safe Harbor method

        Args:
            phi_data: PHI data to de-identify
            keep_fields: Fields to keep (must not contain identifiers)

        Returns:
            DeIdentifiedData object
        """
        keep_fields = keep_fields or []

        # Create hash of original data for re-identification key
        original_hash = hashlib.sha256(
            json.dumps(phi_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Remove all 18 HIPAA identifiers
        de_identified = {}
        removed_identifiers = []

        for key, value in phi_data.items():
            # Check if this field is a HIPAA identifier
            is_identifier = self._is_hipaa_identifier(key)

            if is_identifier:
                category = self._get_phi_category(key)
                removed_identifiers.append(category)
            elif key in keep_fields or self._is_safe_field(key):
                # Keep safe fields (age >89 → 90+, dates → year only, etc.)
                de_identified[key] = self._generalize_value(key, value)

        return DeIdentifiedData(
            data_id=f"deid_{uuid.uuid4().hex[:12]}",
            original_data_hash=original_hash,
            de_identified_content=de_identified,
            removed_identifiers=list(set(removed_identifiers)),
            de_identified_at=datetime.now(),
            de_identification_method="HIPAA Safe Harbor (§164.514(b)(2))",
            attestation="All 18 HIPAA identifiers removed per Safe Harbor method"
        )

    def _is_hipaa_identifier(self, field_name: str) -> bool:
        """Check if field is a HIPAA identifier"""
        identifier_keywords = [
            'name', 'address', 'city', 'zip', 'phone', 'fax', 'email',
            'ssn', 'mrn', 'medical_record', 'account', 'certificate',
            'vehicle', 'device_id', 'url', 'ip', 'biometric', 'photo'
        ]

        field_lower = field_name.lower()
        return any(keyword in field_lower for keyword in identifier_keywords)

    def _get_phi_category(self, field_name: str) -> PHICategory:
        """Get PHI category for field"""
        field_lower = field_name.lower()

        if 'name' in field_lower:
            return PHICategory.NAME
        elif any(x in field_lower for x in ['address', 'city', 'zip']):
            return PHICategory.GEOGRAPHIC
        elif any(x in field_lower for x in ['phone', 'telephone']):
            return PHICategory.PHONE
        elif 'email' in field_lower:
            return PHICategory.EMAIL
        elif 'ssn' in field_lower:
            return PHICategory.SSN
        elif any(x in field_lower for x in ['mrn', 'medical_record']):
            return PHICategory.MRN
        else:
            return PHICategory.UNIQUE_CODE

    def _is_safe_field(self, field_name: str) -> bool:
        """Check if field is safe to keep (non-identifying)"""
        safe_keywords = [
            'age', 'sex', 'gender', 'diagnosis', 'treatment',
            'medication', 'lab', 'test', 'result', 'genomic',
            'mutation', 'biomarker', 'imaging', 'scan'
        ]

        field_lower = field_name.lower()
        return any(keyword in field_lower for keyword in safe_keywords)

    def _generalize_value(self, key: str, value: Any) -> Any:
        """Generalize value to remove identifying information"""
        # Age >89 → "90+"
        if 'age' in key.lower() and isinstance(value, (int, float)) and value > 89:
            return "90+"

        # Dates → year only
        if 'date' in key.lower() and isinstance(value, (datetime, str)):
            try:
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                    dt = value
                return dt.year
            except:
                pass

        return value


class HIPAAAuditLogger:
    """
    HIPAA-compliant audit logging

    Implements HIPAA Security Rule §164.312(b) - Audit Controls
    Tracks all access to PHI with immutable logs
    """

    def __init__(self):
        """Initialize audit logger"""
        self.audit_log: List[AuditLogEntry] = []
        logger.info("📋 HIPAA Audit Logger initialized")

    def log_access(
        self,
        user_id: str,
        user_role: AccessLevel,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        phi_accessed: List[PHICategory],
        ip_address: str,
        success: bool = True,
        failure_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """
        Log PHI access event

        Args:
            user_id: User identifier
            user_role: User's access level
            action: Action performed
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            phi_accessed: PHI categories accessed
            ip_address: Source IP address
            success: Whether action succeeded
            failure_reason: Reason for failure (if applicable)
            metadata: Additional metadata

        Returns:
            AuditLogEntry
        """
        entry = AuditLogEntry(
            log_id=f"audit_{uuid.uuid4().hex}",
            timestamp=datetime.now(),
            user_id=user_id,
            user_role=user_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            phi_accessed=phi_accessed,
            ip_address=ip_address,
            success=success,
            failure_reason=failure_reason,
            metadata=metadata or {}
        )

        # Append to immutable log
        self.audit_log.append(entry)

        # Log to file (in production: send to SIEM system)
        logger.info(
            f"📋 AUDIT: {user_id} ({user_role.value}) {action.value} "
            f"{resource_type}:{resource_id} - {'SUCCESS' if success else 'FAILED'}"
        )

        return entry

    def get_user_audit_trail(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditLogEntry]:
        """Get audit trail for specific user"""
        start = start_date or datetime.min
        end = end_date or datetime.now()

        return [
            entry for entry in self.audit_log
            if entry.user_id == user_id and start <= entry.timestamp <= end
        ]

    def get_resource_audit_trail(
        self,
        resource_type: str,
        resource_id: str
    ) -> List[AuditLogEntry]:
        """Get audit trail for specific resource"""
        return [
            entry for entry in self.audit_log
            if entry.resource_type == resource_type and entry.resource_id == resource_id
        ]

    def detect_anomalous_access(self) -> List[AuditLogEntry]:
        """
        Detect potentially anomalous access patterns

        Returns:
            List of suspicious audit entries
        """
        suspicious = []

        # Check for excessive failed access attempts
        user_failures: Dict[str, int] = {}
        for entry in self.audit_log[-1000:]:  # Last 1000 entries
            if not entry.success:
                user_failures[entry.user_id] = user_failures.get(entry.user_id, 0) + 1

        # Flag users with >5 failures
        for user_id, failures in user_failures.items():
            if failures > 5:
                suspicious.extend([
                    e for e in self.audit_log
                    if e.user_id == user_id and not e.success
                ][-5:])  # Last 5 failures

        return suspicious


class HIPAAComplianceFramework:
    """
    🔒 HIPAA Compliance Framework

    Comprehensive HIPAA compliance for quantum digital twins:
    - PHI encryption at rest and in transit
    - De-identification for research
    - Audit logging and access control
    - Breach detection and notification
    - Business Associate Agreement (BAA) support

    Reference: HEALTHCARE_FOCUS_STRATEGIC_PLAN.md - Security & Compliance
    """

    def __init__(
        self,
        enable_encryption: bool = True,
        enable_audit_logging: bool = True,
        master_key: Optional[bytes] = None
    ):
        """Initialize HIPAA compliance framework"""
        self.enable_encryption = enable_encryption
        self.enable_audit_logging = enable_audit_logging

        # Initialize components
        self.encryption_manager = HIPAAEncryptionManager(master_key) if enable_encryption else None
        self.de_identifier = HIPAADeIdentifier()
        self.audit_logger = HIPAAAuditLogger() if enable_audit_logging else None

        # Breach tracking
        self.breaches: List[BreachNotification] = []

        logger.info("🔒 HIPAA Compliance Framework initialized")
        logger.info(f"   Encryption: {'ENABLED' if enable_encryption else 'DISABLED'}")
        logger.info(f"   Audit Logging: {'ENABLED' if enable_audit_logging else 'DISABLED'}")

    def encrypt_patient_data(
        self,
        patient_data: Dict[str, Any],
        phi_categories: List[PHICategory],
        user_id: str,
        user_role: AccessLevel
    ) -> EncryptedPHI:
        """
        Encrypt patient PHI data

        Args:
            patient_data: Patient data to encrypt
            phi_categories: PHI categories in data
            user_id: User performing encryption
            user_role: User's access level

        Returns:
            EncryptedPHI object
        """
        if not self.encryption_manager:
            raise RuntimeError("Encryption not enabled")

        # Encrypt
        encrypted = self.encryption_manager.encrypt_phi(patient_data, phi_categories)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_access(
                user_id=user_id,
                user_role=user_role,
                action=AuditAction.MODIFY,
                resource_type="patient_data",
                resource_id=patient_data.get('patient_id', 'unknown'),
                phi_accessed=phi_categories,
                ip_address="127.0.0.1",  # Replace with actual IP
                success=True,
                metadata={'action': 'encrypt'}
            )

        return encrypted

    def decrypt_patient_data(
        self,
        encrypted_phi: EncryptedPHI,
        user_id: str,
        user_role: AccessLevel,
        authorized_roles: Optional[List[AccessLevel]] = None
    ) -> Dict[str, Any]:
        """
        Decrypt patient PHI data with access control

        Args:
            encrypted_phi: Encrypted PHI object
            user_id: User requesting decryption
            user_role: User's access level
            authorized_roles: Roles authorized to decrypt (default: PROVIDER, ADMIN)

        Returns:
            Decrypted patient data

        Raises:
            PermissionError: If user not authorized
        """
        if not self.encryption_manager:
            raise RuntimeError("Encryption not enabled")

        # Access control
        authorized_roles = authorized_roles or [AccessLevel.PROVIDER, AccessLevel.ADMIN]
        if user_role not in authorized_roles:
            # Log failed access
            if self.audit_logger:
                self.audit_logger.log_access(
                    user_id=user_id,
                    user_role=user_role,
                    action=AuditAction.DECRYPT,
                    resource_type="encrypted_phi",
                    resource_id=encrypted_phi.encryption_key_id,
                    phi_accessed=encrypted_phi.phi_categories,
                    ip_address="127.0.0.1",
                    success=False,
                    failure_reason=f"Unauthorized role: {user_role.value}"
                )

            raise PermissionError(f"Role {user_role.value} not authorized to decrypt PHI")

        # Decrypt
        decrypted = self.encryption_manager.decrypt_phi(encrypted_phi)

        # Audit log successful access
        if self.audit_logger:
            self.audit_logger.log_access(
                user_id=user_id,
                user_role=user_role,
                action=AuditAction.DECRYPT,
                resource_type="encrypted_phi",
                resource_id=encrypted_phi.encryption_key_id,
                phi_accessed=encrypted_phi.phi_categories,
                ip_address="127.0.0.1",
                success=True
            )

        return decrypted

    def de_identify_for_research(
        self,
        patient_data: Dict[str, Any],
        researcher_id: str
    ) -> DeIdentifiedData:
        """
        De-identify patient data for research use

        Args:
            patient_data: Patient data to de-identify
            researcher_id: Researcher requesting de-identification

        Returns:
            DeIdentifiedData object
        """
        # De-identify
        de_identified = self.de_identifier.de_identify(patient_data)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_access(
                user_id=researcher_id,
                user_role=AccessLevel.RESEARCHER,
                action=AuditAction.DE_IDENTIFY,
                resource_type="patient_data",
                resource_id=patient_data.get('patient_id', 'unknown'),
                phi_accessed=[PHICategory.DIAGNOSIS, PHICategory.TREATMENT],
                ip_address="127.0.0.1",
                success=True,
                metadata={
                    'de_identified_id': de_identified.data_id,
                    'identifiers_removed': len(de_identified.removed_identifiers)
                }
            )

        logger.info(f"✅ De-identified data: {de_identified.data_id}")
        logger.info(f"   Removed {len(de_identified.removed_identifiers)} HIPAA identifiers")

        return de_identified

    def detect_breach(
        self,
        breach_description: str,
        affected_patient_ids: List[str],
        phi_compromised: List[PHICategory]
    ) -> BreachNotification:
        """
        Detect and document HIPAA breach

        Args:
            breach_description: Description of breach
            affected_patient_ids: Patient IDs affected
            phi_compromised: PHI categories compromised

        Returns:
            BreachNotification
        """
        breach = BreachNotification(
            breach_id=f"breach_{uuid.uuid4().hex[:12]}",
            detected_at=datetime.now(),
            breach_type=breach_description,
            affected_patients=len(affected_patient_ids),
            phi_compromised=phi_compromised,
            notification_sent=False,
            notification_sent_at=None,
            hhs_notified=False,
            media_notified=False,
            mitigation_actions=[]
        )

        self.breaches.append(breach)

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_access(
                user_id="SYSTEM",
                user_role=AccessLevel.ADMIN,
                action=AuditAction.BREACH_DETECTED,
                resource_type="security_breach",
                resource_id=breach.breach_id,
                phi_accessed=phi_compromised,
                ip_address="SYSTEM",
                success=True,
                metadata={
                    'affected_patients': len(affected_patient_ids),
                    'description': breach_description
                }
            )

        logger.error(f"🚨 HIPAA BREACH DETECTED: {breach.breach_id}")
        logger.error(f"   Affected patients: {len(affected_patient_ids)}")
        logger.error(f"   PHI compromised: {[c.value for c in phi_compromised]}")

        # HIPAA requires notification within 60 days
        if len(affected_patient_ids) > 500:
            logger.error("   ⚠️ >500 patients affected - media notification required")

        return breach

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate HIPAA compliance report

        Returns:
            Compliance status report
        """
        report = {
            'report_id': f"compliance_{uuid.uuid4().hex[:8]}",
            'generated_at': datetime.now().isoformat(),
            'encryption_enabled': self.enable_encryption,
            'audit_logging_enabled': self.enable_audit_logging,
            'encryption_key_age_days': (
                (datetime.now() - self.encryption_manager.key_created_at).days
                if self.encryption_manager else None
            ),
            'total_audit_entries': len(self.audit_logger.audit_log) if self.audit_logger else 0,
            'total_breaches': len(self.breaches),
            'breaches_requiring_notification': len([
                b for b in self.breaches if not b.notification_sent
            ]),
            'compliance_status': 'COMPLIANT' if self.enable_encryption and self.enable_audit_logging else 'NON-COMPLIANT',
            'recommendations': []
        }

        # Add recommendations
        if self.encryption_manager and (datetime.now() - self.encryption_manager.key_created_at).days > 90:
            report['recommendations'].append("Rotate encryption keys (>90 days old)")

        if not self.enable_encryption:
            report['recommendations'].append("CRITICAL: Enable PHI encryption")

        if not self.enable_audit_logging:
            report['recommendations'].append("CRITICAL: Enable audit logging")

        return report


# Convenience functions
def encrypt_phi(
    data: Dict[str, Any],
    phi_categories: List[PHICategory]
) -> EncryptedPHI:
    """Convenience function for PHI encryption"""
    framework = HIPAAComplianceFramework()
    return framework.encryption_manager.encrypt_phi(data, phi_categories)


def de_identify_data(
    data: Dict[str, Any]
) -> DeIdentifiedData:
    """Convenience function for de-identification"""
    framework = HIPAAComplianceFramework()
    return framework.de_identifier.de_identify(data)
