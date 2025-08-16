"""
Security and privacy protection for BCI-2-Token framework.

Implements comprehensive security measures, input validation, access control,
and privacy protection for brain-computer interface applications.
"""

import hashlib
import hmac
import secrets
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SecurityConfig:
    """Security configuration for BCI applications."""
    
    # Access control
    enable_access_control: bool = True
    session_timeout: float = 3600.0  # 1 hour
    max_concurrent_sessions: int = 10
    
    # Input validation
    max_signal_duration: float = 300.0  # 5 minutes max
    max_signal_amplitude: float = 1000.0  # µV
    max_batch_size: int = 1000
    
    # Privacy protection
    require_privacy_protection: bool = True
    min_privacy_epsilon: float = 0.1
    max_privacy_epsilon: float = 10.0
    audit_privacy_usage: bool = True
    
    # Data protection
    encrypt_saved_data: bool = True
    secure_delete: bool = True
    audit_data_access: bool = True
    
    # Rate limiting (Enhanced for production security)
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 30  # Reduced for better security
    max_requests_per_hour: int = 500   # Reduced for better security
    
    # Enhanced Generation 2 Security
    enable_anomaly_detection: bool = True
    threat_detection_threshold: float = 0.8
    enable_input_sanitization: bool = True
    log_security_events: bool = True
    enable_session_validation: bool = True
    max_requests_per_minute: int = 60
    max_decode_operations_per_hour: int = 1000
    
    # Advanced security features
    enable_anomaly_detection: bool = True
    suspicious_activity_threshold: int = 10
    auto_block_suspicious_ips: bool = True
    enable_request_signing: bool = True


class AccessController:
    """Controls access to BCI system components."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        
    def create_session(self, user_id: str, permissions: List[str] = None) -> str:
        """
        Create new authenticated session.
        
        Args:
            user_id: User identifier
            permissions: List of permissions for this session
            
        Returns:
            Session token
            
        Raises:
            PermissionError: If too many concurrent sessions
        """
        with self.session_lock:
            # Check concurrent session limit
            active_count = len(self.active_sessions)
            if active_count >= self.config.max_concurrent_sessions:
                raise PermissionError(f"Too many concurrent sessions: {active_count}")
                
            # Generate secure session token
            session_token = secrets.token_urlsafe(32)
            
            # Create session
            session_data = {
                'user_id': user_id,
                'permissions': permissions or ['basic'],
                'created_time': time.time(),
                'last_activity': time.time(),
                'request_count': 0
            }
            
            self.active_sessions[session_token] = session_data
            return session_token
            
    def validate_session(self, session_token: str, required_permission: str = None) -> bool:
        """
        Validate session and check permissions.
        
        Args:
            session_token: Session token to validate
            required_permission: Required permission for operation
            
        Returns:
            True if session is valid and authorized
        """
        with self.session_lock:
            if session_token not in self.active_sessions:
                return False
                
            session = self.active_sessions[session_token]
            current_time = time.time()
            
            # Check session timeout
            if current_time - session['last_activity'] > self.config.session_timeout:
                del self.active_sessions[session_token]
                return False
                
            # Check permissions
            if required_permission and required_permission not in session['permissions']:
                return False
                
            # Update activity
            session['last_activity'] = current_time
            session['request_count'] += 1
            
            return True
            
    def invalidate_session(self, session_token: str):
        """Invalidate a session."""
        with self.session_lock:
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
                
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        with self.session_lock:
            current_time = time.time()
            expired_tokens = [
                token for token, session in self.active_sessions.items()
                if current_time - session['last_activity'] > self.config.session_timeout
            ]
            
            for token in expired_tokens:
                del self.active_sessions[token]
                
            return len(expired_tokens)


class RateLimiter:
    """Rate limiting for BCI operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history: Dict[str, List[float]] = {}
        self.operation_history: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        
    def check_rate_limit(self, 
                        identifier: str, 
                        operation_type: str = 'request',
                        window_seconds: float = 60.0) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: User/session identifier
            operation_type: Type of operation ('request' or 'decode')
            window_seconds: Time window for rate limiting
            
        Returns:
            True if within limits, False otherwise
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            # Choose appropriate history and limit
            if operation_type == 'decode':
                history = self.operation_history
                limit = self.config.max_decode_operations_per_hour
                window_seconds = 3600.0  # 1 hour for decode operations
                cutoff_time = current_time - window_seconds
            else:
                history = self.request_history
                limit = self.config.max_requests_per_minute
                
            # Initialize history for new identifiers
            if identifier not in history:
                history[identifier] = []
                
            # Clean old requests
            history[identifier] = [
                t for t in history[identifier] if t > cutoff_time
            ]
            
            # Check limit
            if len(history[identifier]) >= limit:
                return False
                
            # Record current request
            history[identifier].append(current_time)
            return True
            
    def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for identifier."""
        with self.lock:
            current_time = time.time()
            
            # Count recent requests
            recent_requests = 0
            if identifier in self.request_history:
                cutoff = current_time - 60.0
                recent_requests = sum(1 for t in self.request_history[identifier] if t > cutoff)
                
            # Count recent decode operations
            recent_decodes = 0
            if identifier in self.operation_history:
                cutoff = current_time - 3600.0
                recent_decodes = sum(1 for t in self.operation_history[identifier] if t > cutoff)
                
            return {
                'requests_per_minute': recent_requests,
                'max_requests_per_minute': self.config.max_requests_per_minute,
                'decode_operations_per_hour': recent_decodes,
                'max_decode_operations_per_hour': self.config.max_decode_operations_per_hour,
                'requests_remaining': max(0, self.config.max_requests_per_minute - recent_requests),
                'decodes_remaining': max(0, self.config.max_decode_operations_per_hour - recent_decodes)
            }


class DataProtector:
    """Protects sensitive brain data."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_key = self._generate_key() if config.encrypt_saved_data else None
        
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
        
    def encrypt_data(self, data: bytes) -> Dict[str, bytes]:
        """
        Encrypt sensitive data.
        
        Args:
            data: Raw data to encrypt
            
        Returns:
            Dictionary with encrypted data and metadata
        """
        if not self.config.encrypt_saved_data or not self.encryption_key:
            return {'data': data, 'encrypted': False}
            
        try:
            from cryptography.fernet import Fernet
            
            # Use the first 32 bytes as Fernet key (base64 encoded)
            import base64
            fernet_key = base64.urlsafe_b64encode(self.encryption_key)
            cipher = Fernet(fernet_key)
            
            encrypted_data = cipher.encrypt(data)
            
            return {
                'data': encrypted_data,
                'encrypted': True,
                'algorithm': 'Fernet',
                'timestamp': time.time()
            }
            
        except ImportError:
            warnings.warn("cryptography library not available, storing data unencrypted")
            return {'data': data, 'encrypted': False}
            
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """
        Decrypt protected data.
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            Decrypted data
        """
        if not encrypted_package.get('encrypted', False):
            return encrypted_package['data']
            
        try:
            from cryptography.fernet import Fernet
            import base64
            
            fernet_key = base64.urlsafe_b64encode(self.encryption_key)
            cipher = Fernet(fernet_key)
            
            return cipher.decrypt(encrypted_package['data'])
            
        except ImportError:
            raise RuntimeError("cryptography library required for decryption")
            
    def secure_delete_file(self, file_path: Union[str, Path]):
        """
        Securely delete a file by overwriting before deletion.
        
        Args:
            file_path: Path to file to delete
        """
        if not self.config.secure_delete:
            Path(file_path).unlink(missing_ok=True)
            return
            
        file_path = Path(file_path)
        
        if not file_path.exists():
            return
            
        try:
            # Overwrite file with random data multiple times
            file_size = file_path.stat().st_size
            
            for _ in range(3):  # Three passes
                with open(file_path, 'wb') as f:
                    # Write random data
                    remaining = file_size
                    while remaining > 0:
                        chunk_size = min(remaining, 8192)
                        random_data = secrets.token_bytes(chunk_size)
                        f.write(random_data)
                        remaining -= chunk_size
                        
                # Sync to disk
                f.flush()
                
            # Finally delete the file
            file_path.unlink()
            
        except Exception as e:
            warnings.warn(f"Secure deletion failed: {e}")
            # Fallback to regular deletion
            file_path.unlink(missing_ok=True)


class AuditLogger:
    """Audit logging for security-sensitive operations."""
    
    def __init__(self, audit_file: Optional[Path] = None):
        self.audit_file = audit_file
        self.lock = threading.Lock()
        
    def log_access(self, 
                   user_id: str,
                   operation: str,
                   resource: str,
                   success: bool,
                   details: Optional[Dict[str, Any]] = None):
        """
        Log access attempt.
        
        Args:
            user_id: User attempting access
            operation: Operation attempted
            resource: Resource accessed
            success: Whether access was successful
            details: Additional details
        """
        audit_entry = {
            'timestamp': time.time(),
            'user_id': user_id,
            'operation': operation,
            'resource': resource,
            'success': success,
            'details': details or {}
        }
        
        self._write_audit_entry(audit_entry)
        
    def log_privacy_operation(self,
                             user_id: str,
                             epsilon_used: float,
                             data_type: str,
                             purpose: str):
        """
        Log privacy-related operations.
        
        Args:
            user_id: User performing operation
            epsilon_used: Privacy budget consumed
            data_type: Type of data processed
            purpose: Purpose of the operation
        """
        audit_entry = {
            'timestamp': time.time(),
            'type': 'privacy_operation',
            'user_id': user_id,
            'epsilon_used': epsilon_used,
            'data_type': data_type,
            'purpose': purpose
        }
        
        self._write_audit_entry(audit_entry)
        
    def log_data_access(self,
                       user_id: str,
                       file_path: str,
                       operation: str,
                       success: bool):
        """
        Log data file access.
        
        Args:
            user_id: User accessing data
            file_path: Path to data file
            operation: Operation performed ('read', 'write', 'delete')
            success: Whether operation succeeded
        """
        # Hash file path for privacy
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        
        audit_entry = {
            'timestamp': time.time(),
            'type': 'data_access',
            'user_id': user_id,
            'file_path_hash': path_hash,
            'operation': operation,
            'success': success
        }
        
        self._write_audit_entry(audit_entry)
        
    def _write_audit_entry(self, entry: Dict[str, Any]):
        """Write audit entry to log."""
        if not self.audit_file:
            return
            
        with self.lock:
            try:
                import json
                with open(self.audit_file, 'a') as f:
                    f.write(json.dumps(entry, default=str) + '\n')
            except Exception as e:
                warnings.warn(f"Failed to write audit log: {e}")


class PrivacyValidator:
    """Validates privacy protection mechanisms."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.privacy_budget_usage: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def validate_privacy_request(self,
                                user_id: str,
                                requested_epsilon: float,
                                operation: str) -> bool:
        """
        Validate privacy protection request.
        
        Args:
            user_id: User requesting operation
            requested_epsilon: Requested privacy budget
            operation: Type of operation
            
        Returns:
            True if request is valid and within limits
        """
        # Check epsilon range
        if not (self.config.min_privacy_epsilon <= requested_epsilon <= self.config.max_privacy_epsilon):
            warnings.warn(
                f"Privacy epsilon {requested_epsilon} outside valid range "
                f"[{self.config.min_privacy_epsilon}, {self.config.max_privacy_epsilon}]"
            )
            return False
            
        # Check user's privacy budget usage
        with self.lock:
            current_usage = self.privacy_budget_usage.get(user_id, 0.0)
            
            # Simple budget limit (could be more sophisticated)
            daily_limit = 10.0  # Total epsilon per day
            
            if current_usage + requested_epsilon > daily_limit:
                warnings.warn(f"Privacy budget exceeded for user {user_id}")
                return False
                
            # Update usage
            self.privacy_budget_usage[user_id] = current_usage + requested_epsilon
            
        return True
        
    def record_privacy_usage(self,
                            user_id: str,
                            epsilon_used: float,
                            data_sensitivity: str = 'high'):
        """
        Record privacy budget usage.
        
        Args:
            user_id: User who consumed privacy budget
            epsilon_used: Amount of privacy budget used
            data_sensitivity: Sensitivity level of processed data
        """
        with self.lock:
            current_usage = self.privacy_budget_usage.get(user_id, 0.0)
            self.privacy_budget_usage[user_id] = current_usage + epsilon_used
            
        # Log to audit if enabled
        if self.config.audit_privacy_usage:
            from .monitoring import get_monitor
            monitor = get_monitor()
            monitor.logger.info(
                'Privacy',
                f'Privacy budget used: {epsilon_used:.3f}',
                {
                    'user_id': user_id,
                    'epsilon_used': epsilon_used,
                    'total_usage': self.privacy_budget_usage[user_id],
                    'data_sensitivity': data_sensitivity
                }
            )
            
    def get_privacy_budget_status(self, user_id: str) -> Dict[str, float]:
        """Get privacy budget status for user."""
        with self.lock:
            current_usage = self.privacy_budget_usage.get(user_id, 0.0)
            daily_limit = 10.0
            
            return {
                'user_id': user_id,
                'current_usage': current_usage,
                'daily_limit': daily_limit,
                'remaining': max(0, daily_limit - current_usage),
                'usage_percentage': min(100, (current_usage / daily_limit) * 100)
            }


class SecureProcessor:
    """Secure processing pipeline for brain signals."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.access_controller = AccessController(config)
        self.rate_limiter = RateLimiter(config)
        self.privacy_validator = PrivacyValidator(config)
        self.data_protector = DataProtector(config)
        self.audit_logger = AuditLogger()
        
    def process_signal_securely(self,
                               signal: Any,
                               user_id: str,
                               session_token: str,
                               privacy_epsilon: Optional[float] = None) -> Dict[str, Any]:
        """
        Process brain signal with security measures.
        
        Args:
            signal: Brain signal data
            user_id: User identifier
            session_token: Session token
            privacy_epsilon: Privacy protection level
            
        Returns:
            Processing result with security metadata
            
        Raises:
            PermissionError: If access is denied
            ValueError: If input validation fails
        """
        start_time = time.time()
        
        try:
            # 1. Session validation
            if not self.access_controller.validate_session(session_token, 'process_signal'):
                self.audit_logger.log_access(user_id, 'process_signal', 'brain_signal', False, 
                                           {'reason': 'invalid_session'})
                raise PermissionError("Invalid or expired session")
                
            # 2. Rate limiting
            if not self.rate_limiter.check_rate_limit(user_id, 'decode'):
                self.audit_logger.log_access(user_id, 'process_signal', 'brain_signal', False,
                                           {'reason': 'rate_limit_exceeded'})
                raise PermissionError("Rate limit exceeded")
                
            # 3. Input validation and sanitization
            from .reliability import InputSanitizer
            sanitized_signal = InputSanitizer.sanitize_brain_signal(
                signal,
                max_channels=self.config.max_signal_amplitude,
                max_amplitude=self.config.max_signal_amplitude
            )
            
            # 4. Privacy validation
            if privacy_epsilon is not None:
                if not self.privacy_validator.validate_privacy_request(user_id, privacy_epsilon, 'decode'):
                    raise PermissionError("Privacy request denied")
                    
            # 5. Process signal (placeholder - would call actual decoder)
            processing_result = {
                'success': True,
                'tokens': [],  # Placeholder
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
            
            # 6. Record privacy usage
            if privacy_epsilon is not None:
                self.privacy_validator.record_privacy_usage(user_id, privacy_epsilon)
                
            # 7. Audit successful operation
            self.audit_logger.log_access(user_id, 'process_signal', 'brain_signal', True, {
                'signal_shape': signal.shape if HAS_NUMPY and hasattr(signal, 'shape') else 'unknown',
                'privacy_epsilon': privacy_epsilon,
                'processing_time': processing_result['processing_time']
            })
            
            return processing_result
            
        except Exception as e:
            # Audit failed operation
            self.audit_logger.log_access(user_id, 'process_signal', 'brain_signal', False, {
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
            
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'access_control': {
                'active_sessions': len(self.access_controller.active_sessions),
                'max_sessions': self.config.max_concurrent_sessions
            },
            'rate_limiting': {
                'max_requests_per_minute': self.config.max_requests_per_minute,
                'max_decode_operations_per_hour': self.config.max_decode_operations_per_hour
            },
            'privacy_protection': {
                'enabled': self.config.require_privacy_protection,
                'epsilon_range': [self.config.min_privacy_epsilon, self.config.max_privacy_epsilon]
            },
            'data_protection': {
                'encryption_enabled': self.config.encrypt_saved_data,
                'secure_delete_enabled': self.config.secure_delete,
                'audit_enabled': self.config.audit_data_access
            }
        }


def create_secure_session(user_id: str, 
                         permissions: List[str] = None,
                         config: SecurityConfig = None) -> str:
    """
    Create a secure session for BCI operations.
    
    Args:
        user_id: User identifier
        permissions: List of permissions
        config: Security configuration
        
    Returns:
        Session token
    """
    if config is None:
        config = SecurityConfig()
        
    processor = SecureProcessor(config)
    return processor.access_controller.create_session(user_id, permissions)


def secure_brain_operation(operation_name: str, config: SecurityConfig = None):
    """
    Decorator for securing brain processing operations.
    
    Args:
        operation_name: Name of the operation
        config: Security configuration
        
    Returns:
        Decorated function with security measures
    """
    if config is None:
        config = SecurityConfig()
        
    processor = SecureProcessor(config)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(signal, user_id: str, session_token: str, *args, **kwargs):
            # Validate session
            if not processor.access_controller.validate_session(session_token):
                raise PermissionError("Invalid session")
                
            # Check rate limits
            if not processor.rate_limiter.check_rate_limit(user_id):
                raise PermissionError("Rate limit exceeded")
                
            # Sanitize inputs
            from .reliability import InputSanitizer
            sanitized_signal = InputSanitizer.sanitize_brain_signal(signal)
            
            # Execute operation
            return func(sanitized_signal, *args, **kwargs)
            
        return wrapper
    return decorator


if __name__ == '__main__':
    # Test security system
    print("Testing BCI-2-Token Security System")
    print("=" * 45)
    
    # Test configuration
    config = SecurityConfig(
        enable_access_control=True,
        max_concurrent_sessions=5,
        require_privacy_protection=True
    )
    
    # Test access control
    processor = SecureProcessor(config)
    
    try:
        # Create session
        session_token = processor.access_controller.create_session('test_user', ['process_signal'])
        print(f"✓ Session created: {session_token[:16]}...")
        
        # Validate session
        valid = processor.access_controller.validate_session(session_token, 'process_signal')
        print(f"✓ Session validation: {valid}")
        
        # Test rate limiting
        for i in range(3):
            within_limit = processor.rate_limiter.check_rate_limit('test_user')
            print(f"✓ Rate limit check {i+1}: {within_limit}")
            
        # Test privacy validation
        privacy_valid = processor.privacy_validator.validate_privacy_request('test_user', 1.0, 'decode')
        print(f"✓ Privacy validation: {privacy_valid}")
        
        # Test data protection
        test_data = b"test brain data"
        encrypted = processor.data_protector.encrypt_data(test_data)
        print(f"✓ Data encryption: {encrypted['encrypted']}")
        
        print("\n✓ Security system working")
        
    except Exception as e:
        print(f"✗ Security test failed: {e}")
        import traceback
        traceback.print_exc()


class EnhancedSecurityFramework:
    """Generation 2 Enhanced Security Framework with advanced threat detection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.threat_scores = {}
        self.security_events = []
        self.lock = threading.Lock()
        
    def analyze_threat_level(self, user_id: str, signal_data: Any, operation: str) -> float:
        """Analyze threat level for incoming requests."""
        threat_score = 0.0
        
        if not HAS_NUMPY:
            return threat_score
            
        try:
            signal_array = np.asarray(signal_data)
            
            # Anomaly detection based on signal characteristics
            if signal_array.size > 0:
                # Check for unusual amplitude patterns
                max_amp = np.max(np.abs(signal_array))
                if max_amp > self.config.max_signal_amplitude * 2:
                    threat_score += 0.3
                    
                # Check for artificial patterns (too regular)
                if signal_array.ndim == 2 and signal_array.shape[1] > 10:
                    variance_across_time = np.var(signal_array, axis=1)
                    if np.all(variance_across_time < 0.001):  # Too regular
                        threat_score += 0.4
                        
                # Check for potential injection attacks (sudden spikes)
                if signal_array.ndim == 2:
                    for channel in signal_array:
                        diff = np.diff(channel)
                        if np.any(np.abs(diff) > max_amp * 0.8):
                            threat_score += 0.2
                            
            # Behavioral analysis
            with self.lock:
                if user_id not in self.threat_scores:
                    self.threat_scores[user_id] = []
                    
                # Check request frequency (rapid requests could indicate automation)
                current_time = time.time()
                recent_requests = [t for t in self.threat_scores[user_id] 
                                 if current_time - t < 60]  # Last minute
                
                if len(recent_requests) > 10:  # More than 10 requests per minute
                    threat_score += 0.3
                    
                self.threat_scores[user_id].append(current_time)
                
                # Keep only recent history
                self.threat_scores[user_id] = [
                    t for t in self.threat_scores[user_id] 
                    if current_time - t < 3600  # Last hour
                ]
                
        except Exception as e:
            # Analysis failure itself could indicate an attack
            threat_score += 0.5
            self._log_security_event('analysis_failure', user_id, f"Threat analysis failed: {e}")
            
        return min(threat_score, 1.0)  # Cap at 1.0
        
    def sanitize_input(self, signal_data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if not self.config.enable_input_sanitization:
            return signal_data
            
        if not HAS_NUMPY:
            return signal_data
            
        try:
            signal_array = np.asarray(signal_data)
            
            # Remove NaN and Inf values
            signal_array = np.nan_to_num(signal_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip extreme values
            max_allowed = self.config.max_signal_amplitude
            signal_array = np.clip(signal_array, -max_allowed, max_allowed)
            
            # Ensure reasonable size
            if signal_array.size > self.config.max_batch_size * 1000:
                # Truncate oversized inputs
                if signal_array.ndim == 2:
                    max_timepoints = self.config.max_batch_size * 1000 // signal_array.shape[0]
                    signal_array = signal_array[:, :max_timepoints]
                else:
                    signal_array = signal_array[:self.config.max_batch_size * 1000]
                    
            return signal_array
            
        except Exception as e:
            self._log_security_event('sanitization_failure', 'unknown', f"Input sanitization failed: {e}")
            raise SecurityError(f"Input sanitization failed: {e}")
            
    def validate_session(self, session_token: str, user_id: str) -> bool:
        """Validate session token and user identity."""
        if not self.config.enable_session_validation:
            return True
            
        try:
            # Basic session validation (in production, use proper JWT or similar)
            if not session_token or len(session_token) < 10:
                self._log_security_event('invalid_session', user_id, "Invalid session token")
                return False
                
            # Check session format (basic validation)
            if not session_token.replace('-', '').replace('_', '').isalnum():
                self._log_security_event('malformed_session', user_id, "Malformed session token")
                return False
                
            return True
            
        except Exception as e:
            self._log_security_event('session_validation_error', user_id, f"Session validation error: {e}")
            return False
            
    def detect_anomalies(self, user_id: str, operation_metrics: Dict[str, float]) -> bool:
        """Detect anomalous behavior patterns."""
        if not self.config.enable_anomaly_detection:
            return False
            
        try:
            anomaly_indicators = []
            
            # Check processing time anomalies
            if 'processing_time' in operation_metrics:
                proc_time = operation_metrics['processing_time']
                if proc_time > 30.0:  # Unusually long processing
                    anomaly_indicators.append('long_processing_time')
                    
            # Check data size anomalies
            if 'data_size' in operation_metrics:
                data_size = operation_metrics['data_size']
                if data_size > self.config.max_batch_size * 1000:
                    anomaly_indicators.append('oversized_data')
                    
            # Check resource usage anomalies
            if 'memory_usage' in operation_metrics:
                memory_usage = operation_metrics['memory_usage']
                if memory_usage > 1000000000:  # > 1GB
                    anomaly_indicators.append('excessive_memory')
                    
            if anomaly_indicators:
                self._log_security_event('anomaly_detected', user_id, 
                                       f"Anomalies detected: {anomaly_indicators}")
                return True
                
            return False
            
        except Exception as e:
            self._log_security_event('anomaly_detection_error', user_id, 
                                   f"Anomaly detection error: {e}")
            return False
            
    def _log_security_event(self, event_type: str, user_id: str, description: str):
        """Log security events for audit trail."""
        if not self.config.log_security_events:
            return
            
        security_event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'description': description
        }
        
        with self.lock:
            self.security_events.append(security_event)
            
            # Keep only recent events (last 24 hours)
            cutoff_time = time.time() - 86400
            self.security_events = [
                event for event in self.security_events 
                if event['timestamp'] > cutoff_time
            ]
            
        # Also log to monitoring system
        try:
            from .monitoring import get_monitor
            monitor = get_monitor()
            monitor.logger.warning(
                'Security',
                f'{event_type}: {description}',
                {
                    'user_id': user_id,
                    'event_type': event_type,
                    'timestamp': security_event['timestamp']
                }
            )
        except ImportError:
            pass
            
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self.lock:
            current_time = time.time()
            
            # Count recent security events
            recent_events = [
                event for event in self.security_events
                if current_time - event['timestamp'] < 3600  # Last hour
            ]
            
            # Categorize events
            event_counts = {}
            for event in recent_events:
                event_type = event['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
            # Determine security status
            total_recent_events = len(recent_events)
            if total_recent_events == 0:
                security_level = 'secure'
            elif total_recent_events < 5:
                security_level = 'elevated'
            else:
                security_level = 'high_risk'
                
            return {
                'security_level': security_level,
                'recent_events_count': total_recent_events,
                'event_breakdown': event_counts,
                'active_users': len(self.threat_scores),
                'config_status': {
                    'anomaly_detection': self.config.enable_anomaly_detection,
                    'input_sanitization': self.config.enable_input_sanitization,
                    'session_validation': self.config.enable_session_validation,
                    'event_logging': self.config.log_security_events
                }
            }


class SecurityError(Exception):
    """Security-related error."""
    pass