"""
Enterprise Features: Monitoring, Logging, Security, and High Availability
"""

import os
import json
import time
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import threading
import queue

import torch
import numpy as np

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise features."""
    
    # Monitoring
    monitoring_enabled: bool = True
    prometheus_port: int = 9090
    metrics_interval: int = 60
    
    # Security
    encryption_enabled: bool = True
    audit_logging: bool = True
    access_control: bool = True
    api_key_required: bool = True
    
    # High Availability
    checkpoint_replication: int = 3
    auto_recovery: bool = True
    health_check_interval: int = 30
    
    # Resource Management
    gpu_memory_fraction: float = 0.9
    cpu_memory_limit: str = "256GB"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Audit
    audit_log_path: str = "logs/audit.log"
    audit_retention_days: int = 90


class MetricsCollector:
    """Collect and expose metrics for monitoring."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        if PROMETHEUS_AVAILABLE and config.monitoring_enabled:
            self._setup_prometheus()
        
        if OPENTELEMETRY_AVAILABLE and config.monitoring_enabled:
            self._setup_opentelemetry()
        
        # Internal metrics storage
        self._metrics_store: Dict[str, List[float]] = {}
        self._metrics_lock = threading.Lock()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics."""
        # Training metrics
        self.train_loss_gauge = Gauge(
            'nexus_train_loss',
            'Current training loss'
        )
        self.train_step_counter = Counter(
            'nexus_train_steps_total',
            'Total training steps'
        )
        self.train_time_histogram = Histogram(
            'nexus_train_step_duration_seconds',
            'Training step duration'
        )
        
        # Model metrics
        self.model_memory_gauge = Gauge(
            'nexus_model_memory_bytes',
            'Model memory usage'
        )
        self.model_params_gauge = Gauge(
            'nexus_model_params_total',
            'Total model parameters'
        )
        
        # Inference metrics
        self.inference_requests_counter = Counter(
            'nexus_inference_requests_total',
            'Total inference requests',
            ['status']
        )
        self.inference_latency_histogram = Histogram(
            'nexus_inference_latency_seconds',
            'Inference latency'
        )
        self.inference_tokens_counter = Counter(
            'nexus_inference_tokens_total',
            'Total tokens generated'
        )
        
        # System metrics
        self.gpu_memory_gauge = Gauge(
            'nexus_gpu_memory_used_bytes',
            'GPU memory used',
            ['gpu_id']
        )
        self.gpu_utilization_gauge = Gauge(
            'nexus_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        # Start Prometheus server
        start_http_server(self.config.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
    
    def _setup_opentelemetry(self):
        """Setup OpenTelemetry tracing."""
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
        
        metrics.set_meter_provider(MeterProvider())
        self.meter = metrics.get_meter(__name__)
    
    def record_training_metrics(
        self,
        loss: float,
        step: int,
        duration: float,
    ):
        """Record training metrics."""
        if PROMETHEUS_AVAILABLE:
            self.train_loss_gauge.set(loss)
            self.train_step_counter.inc()
            self.train_time_histogram.observe(duration)
        
        # Store internally
        with self._metrics_lock:
            self._metrics_store.setdefault('train_loss', []).append(loss)
            self._metrics_store.setdefault('train_step', []).append(step)
    
    def record_inference_metrics(
        self,
        latency: float,
        tokens: int,
        status: str = "success",
    ):
        """Record inference metrics."""
        if PROMETHEUS_AVAILABLE:
            self.inference_requests_counter.labels(status=status).inc()
            self.inference_latency_histogram.observe(latency)
            self.inference_tokens_counter.inc(tokens)
        
        with self._metrics_lock:
            self._metrics_store.setdefault('inference_latency', []).append(latency)
            self._metrics_store.setdefault('inference_tokens', []).append(tokens)
    
    def record_system_metrics(self):
        """Record system resource metrics."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                if PROMETHEUS_AVAILABLE:
                    self.gpu_memory_gauge.labels(gpu_id=str(i)).set(memory_used)
                    self.gpu_utilization_gauge.labels(gpu_id=str(i)).set(
                        memory_used / memory_total * 100
                    )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._metrics_lock:
            summary = {}
            
            for metric_name, values in self._metrics_store.items():
                if values:
                    summary[metric_name] = {
                        'latest': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values),
                    }
            
            return summary
    
    def start_periodic_collection(self, interval: int = 60):
        """Start periodic metrics collection."""
        def collect_loop():
            while True:
                self.record_system_metrics()
                time.sleep(interval)
        
        thread = threading.Thread(target=collect_loop, daemon=True)
        thread.start()
        logger.info(f"Started periodic metrics collection with interval {interval}s")


class AuditLogger:
    """Audit logging for enterprise compliance."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.log_path = Path(config.audit_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup audit log file
        self.audit_file = open(self.log_path, 'a', encoding='utf-8')
        
        # Audit event queue for async logging
        self.audit_queue = queue.Queue()
        self._start_async_writer()
    
    def _start_async_writer(self):
        """Start async audit log writer."""
        def write_loop():
            while True:
                event = self.audit_queue.get()
                if event is None:
                    break
                self._write_event(event)
        
        thread = threading.Thread(target=write_loop, daemon=True)
        thread.start()
    
    def _write_event(self, event: Dict[str, Any]):
        """Write audit event to file."""
        event['timestamp'] = datetime.utcnow().isoformat()
        event_str = json.dumps(event, ensure_ascii=False)
        self.audit_file.write(event_str + '\n')
        self.audit_file.flush()
    
    def log_event(
        self,
        event_type: str,
        user: Optional[str] = None,
        action: str,
        resource: Optional[str] = None,
        details: Optional[Dict] = None,
        status: str = "success",
    ):
        """Log audit event."""
        event = {
            "event_type": event_type,
            "user": user or "anonymous",
            "action": action,
            "resource": resource,
            "details": details or {},
            "status": status,
            "ip_address": self._get_client_ip(),
        }
        
        self.audit_queue.put(event)
    
    def log_training_event(
        self,
        action: str,
        model_id: str,
        step: int,
        details: Optional[Dict] = None,
    ):
        """Log training-related audit event."""
        self.log_event(
            event_type="training",
            action=action,
            resource=model_id,
            details={"step": step, **(details or {})},
        )
    
    def log_inference_event(
        self,
        action: str,
        model_id: str,
        request_id: str,
        tokens_generated: int,
        latency_ms: float,
    ):
        """Log inference-related audit event."""
        self.log_event(
            event_type="inference",
            action=action,
            resource=model_id,
            details={
                "request_id": request_id,
                "tokens_generated": tokens_generated,
                "latency_ms": latency_ms,
            },
        )
    
    def log_security_event(
        self,
        action: str,
        user: Optional[str] = None,
        details: Optional[Dict] = None,
        status: str = "success",
    ):
        """Log security-related audit event."""
        self.log_event(
            event_type="security",
            user=user,
            action=action,
            details=details,
            status=status,
        )
    
    def _get_client_ip(self) -> str:
        """Get client IP address."""
        # In production, this would be extracted from request headers
        return "127.0.0.1"
    
    def cleanup_old_logs(self, retention_days: int = 90):
        """Remove audit logs older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Read and filter logs
        temp_path = self.log_path.with_suffix('.temp')
        
        with open(self.log_path, 'r') as old_file, open(temp_path, 'w') as new_file:
            for line in old_file:
                try:
                    event = json.loads(line)
                    event_date = datetime.fromisoformat(event['timestamp'])
                    
                    if event_date > cutoff_date:
                        new_file.write(line)
                except json.JSONDecodeError:
                    continue
        
        # Replace old file with filtered file
        temp_path.replace(self.log_path)
        logger.info(f"Cleaned up audit logs older than {retention_days} days")
    
    def close(self):
        """Close audit logger."""
        self.audit_queue.put(None)
        self.audit_file.close()


class SecurityManager:
    """Security management for enterprise deployment."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        if CRYPTO_AVAILABLE and config.encryption_enabled:
            self._setup_encryption()
        
        # API keys storage
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self._load_api_keys()
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data."""
        # Generate or load encryption key
        key_path = Path("config/encryption.key")
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                self.encryption_key = f.read()
        else:
            # Generate new key
            self.encryption_key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(self.encryption_key)
        
        self.fernet = Fernet(self.encryption_key)
        logger.info("Encryption setup completed")
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available, returning plaintext")
            return data.encode()
        
        return self.fernet.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available")
            return encrypted_data.decode()
        
        return self.fernet.decrypt(encrypted_data).decode()
    
    def generate_api_key(self, user: str, permissions: List[str] = None) -> str:
        """Generate new API key for user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[key_hash] = {
            "user": user,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat(),
            "is_active": True,
        }
        
        self._save_api_keys()
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated info."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        key_info = self.api_keys.get(key_hash)
        
        if key_info is None:
            return None
        
        # Check expiration
        expires_at = datetime.fromisoformat(key_info['expires_at'])
        if datetime.utcnow() > expires_at:
            return None
        
        # Check if active
        if not key_info['is_active']:
            return None
        
        return key_info
    
    def revoke_api_key(self, api_key: str):
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            self.api_keys[key_hash]['is_active'] = False
            self._save_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from storage."""
        keys_path = Path("config/api_keys.json")
        
        if keys_path.exists():
            with open(keys_path, 'r') as f:
                self.api_keys = json.load(f)
    
    def _save_api_keys(self):
        """Save API keys to storage."""
        keys_path = Path("config/api_keys.json")
        keys_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(keys_path, 'w') as f:
            json.dump(self.api_keys, f, indent=2)
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        if not self.config.rate_limit_enabled:
            return True
        
        now = time.time()
        
        # Get request timestamps for this client
        timestamps = self.rate_limits.get(client_id, [])
        
        # Filter timestamps within window
        window_start = now - self.config.rate_limit_window
        recent_requests = [t for t in timestamps if t > window_start]
        
        # Check limit
        if len(recent_requests) >= self.config.rate_limit_requests:
            return False
        
        # Record this request
        recent_requests.append(now)
        self.rate_limits[client_id] = recent_requests
        
        return True
    
    def check_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        key_info = self.validate_api_key(api_key)
        
        if key_info is None:
            return False
        
        return permission in key_info.get('permissions', [])


class HealthChecker:
    """Health checking for high availability."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.health_status: Dict[str, Any] = {}
        self._start_health_checks()
    
    def _start_health_checks(self):
        """Start periodic health checks."""
        def health_check_loop():
            while True:
                self.run_health_checks()
                time.sleep(self.config.health_check_interval)
        
        thread = threading.Thread(target=health_check_loop, daemon=True)
        thread.start()
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = {
            "gpu": self._check_gpu_health(),
            "memory": self._check_memory_health(),
            "model": self._check_model_health(),
            "storage": self._check_storage_health(),
            "network": self._check_network_health(),
        }
        
        overall_status = all(check.get('healthy', False) for check in checks.values())
        
        self.health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy" if overall_status else "unhealthy",
            "checks": checks,
        }
        
        return self.health_status
    
    def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health status."""
        if not torch.cuda.is_available():
            return {"healthy": True, "message": "No GPU available, using CPU"}
        
        gpu_status = []
        
        for i in range(torch.cuda.device_count()):
            try:
                memory_used = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                utilization = memory_used / memory_total
                
                healthy = utilization < self.config.gpu_memory_fraction
                
                gpu_status.append({
                    "gpu_id": i,
                    "healthy": healthy,
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                    "utilization": utilization,
                })
            except Exception as e:
                gpu_status.append({
                    "gpu_id": i,
                    "healthy": False,
                    "error": str(e),
                })
        
        overall_healthy = all(g['healthy'] for g in gpu_status)
        
        return {
            "healthy": overall_healthy,
            "gpus": gpu_status,
        }
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check system memory health."""
        import psutil
        
        memory = psutil.virtual_memory()
        
        healthy = memory.percent < 90  # Less than 90% used
        
        return {
            "healthy": healthy,
            "total": memory.total,
            "used": memory.used,
            "percent": memory.percent,
        }
    
    def _check_model_health(self) -> Dict[str, Any]:
        """Check model health status."""
        # This would check if model is loaded and responding
        # Placeholder implementation
        return {
            "healthy": True,
            "message": "Model operational",
        }
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage health."""
        import psutil
        
        disk = psutil.disk_usage('/')
        
        healthy = disk.percent < 90
        
        return {
            "healthy": healthy,
            "total": disk.total,
            "used": disk.used,
            "percent": disk.percent,
        }
    
    def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity."""
        # Placeholder for network health check
        return {
            "healthy": True,
            "message": "Network operational",
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.health_status


class CheckpointManager:
    """Checkpoint management with replication for high availability."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def save_checkpoint_with_replication(
        self,
        checkpoint_path: str,
        replication_factor: int = None,
    ):
        """Save checkpoint with replication across multiple nodes."""
        replication_factor = replication_factor or self.config.checkpoint_replication
        
        checkpoint_data = self._read_checkpoint(checkpoint_path)
        
        # Save locally
        self._save_local_checkpoint(checkpoint_path, checkpoint_data)
        
        # Replicate to other nodes (if Redis available)
        if REDIS_AVAILABLE:
            for i in range(replication_factor):
                replica_key = f"checkpoint_replica_{i}:{checkpoint_path}"
                self.redis_client.set(replica_key, checkpoint_data)
        
        logger.info(f"Checkpoint saved with {replication_factor} replicas")
    
    def load_checkpoint_with_recovery(
        self,
        checkpoint_path: str,
    ) -> Optional[bytes]:
        """Load checkpoint with automatic recovery from replicas."""
        # Try local first
        checkpoint_data = self._read_checkpoint(checkpoint_path)
        
        if checkpoint_data is not None:
            return checkpoint_data
        
        # Try replicas if local fails
        if REDIS_AVAILABLE:
            for i in range(self.config.checkpoint_replication):
                replica_key = f"checkpoint_replica_{i}:{checkpoint_path}"
                checkpoint_data = self.redis_client.get(replica_key)
                
                if checkpoint_data is not None:
                    # Restore local copy
                    self._save_local_checkpoint(checkpoint_path, checkpoint_data)
                    logger.info(f"Recovered checkpoint from replica {i}")
                    return checkpoint_data
        
        logger.error(f"Failed to load checkpoint from {checkpoint_path}")
        return None
    
    def _read_checkpoint(self, checkpoint_path: str) -> Optional[bytes]:
        """Read checkpoint from local storage."""
        path = Path(checkpoint_path)
        
        if path.exists():
            with open(path, 'rb') as f:
                return f.read()
        
        return None
    
    def _save_local_checkpoint(self, checkpoint_path: str, data: bytes):
        """Save checkpoint to local storage."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(data)


class EnterpriseManager:
    """Central manager for all enterprise features."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        
        # Initialize components
        self.metrics = MetricsCollector(config)
        self.audit = AuditLogger(config)
        self.security = SecurityManager(config)
        self.health = HealthChecker(config)
        self.checkpoint = CheckpointManager(config)
        
        logger.info("Enterprise features initialized")
    
    def setup(self):
        """Setup all enterprise components."""
        # Start metrics collection
        self.metrics.start_periodic_collection(self.config.metrics_interval)
        
        # Setup audit logging
        if self.config.audit_logging:
            self.audit.log_event(
                event_type="system",
                action="enterprise_setup",
                status="success",
            )
    
    def shutdown(self):
        """Shutdown all enterprise components."""
        self.audit.close()
        logger.info("Enterprise features shutdown completed")


def require_api_key(permission: str = "read"):
    """Decorator to require API key authentication."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API key from kwargs or headers
            api_key = kwargs.get('api_key')
            
            if api_key is None:
                raise PermissionError("API key required")
            
            # Validate API key
            security_manager = kwargs.get('security_manager')
            
            if security_manager is None:
                raise ValueError("Security manager not provided")
            
            if not security_manager.validate_api_key(api_key):
                raise PermissionError("Invalid API key")
            
            if not security_manager.check_permission(api_key, permission):
                raise PermissionError(f"Permission '{permission}' required")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def rate_limit():
    """Decorator for rate limiting."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            client_id = kwargs.get('client_id', 'default')
            security_manager = kwargs.get('security_manager')
            
            if security_manager and not security_manager.check_rate_limit(client_id):
                raise RuntimeError("Rate limit exceeded")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator