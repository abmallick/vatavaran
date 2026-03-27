"""
Infrastructure simulation engine for the Incident Response Environment.

Models a realistic microservice architecture with services, logs, metrics,
dependency graphs, deployments, and configuration. Each task injects specific
faults that the agent must diagnose and remediate.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass
class LogEntry:
    timestamp: str
    level: str  # INFO, WARN, ERROR, FATAL
    service: str
    message: str


@dataclass
class Deployment:
    version: str
    deployed_at: str
    deployed_by: str
    change_summary: str
    rollback_available: bool = True


@dataclass
class ServiceState:
    name: str
    status: str = "healthy"  # healthy, degraded, unhealthy, down
    cpu_percent: float = 15.0
    memory_percent: float = 30.0
    latency_p50_ms: float = 12.0
    latency_p99_ms: float = 45.0
    error_rate_percent: float = 0.1
    request_rate_rps: float = 500.0
    replicas: int = 3
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    recent_deploys: list[Deployment] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)


def _ts(base: datetime, offset_minutes: int = 0) -> str:
    return (base + timedelta(minutes=offset_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_healthy_infrastructure(base_time: datetime | None = None) -> dict[str, ServiceState]:
    """Return a baseline healthy microservice topology."""
    t = base_time or datetime(2026, 3, 27, 2, 0, 0)

    services: dict[str, ServiceState] = {
        "api-gateway": ServiceState(
            name="api-gateway",
            cpu_percent=22.0, memory_percent=35.0,
            latency_p50_ms=8.0, latency_p99_ms=35.0,
            error_rate_percent=0.05, request_rate_rps=2400.0,
            replicas=4,
            dependencies=["auth-service", "user-service", "order-service"],
            config={"rate_limit_rps": 5000, "timeout_ms": 3000, "circuit_breaker_threshold": 50},
            recent_deploys=[Deployment("v2.14.1", _ts(t, -1440), "ci-bot", "Minor logging improvements")],
            logs=[
                LogEntry(_ts(t, -10), "INFO", "api-gateway", "Health check passed"),
                LogEntry(_ts(t, -5), "INFO", "api-gateway", "Request routing nominal"),
            ],
        ),
        "auth-service": ServiceState(
            name="auth-service",
            cpu_percent=18.0, memory_percent=28.0,
            latency_p50_ms=5.0, latency_p99_ms=20.0,
            error_rate_percent=0.02, request_rate_rps=1800.0,
            replicas=3,
            dependencies=["cache-service", "database"],
            config={"token_ttl_seconds": 3600, "max_retries": 3, "cache_enabled": True},
            recent_deploys=[Deployment("v1.9.0", _ts(t, -4320), "ci-bot", "Token refresh optimization")],
            logs=[
                LogEntry(_ts(t, -8), "INFO", "auth-service", "Token validation cache hit rate: 94%"),
                LogEntry(_ts(t, -3), "INFO", "auth-service", "Authentication pipeline healthy"),
            ],
        ),
        "user-service": ServiceState(
            name="user-service",
            cpu_percent=12.0, memory_percent=25.0,
            latency_p50_ms=15.0, latency_p99_ms=60.0,
            error_rate_percent=0.1, request_rate_rps=800.0,
            replicas=2,
            dependencies=["database", "cache-service"],
            config={"page_size": 50, "cache_ttl_seconds": 300},
            recent_deploys=[Deployment("v3.2.0", _ts(t, -720), "deploy-svc", "Added profile search endpoint")],
            logs=[
                LogEntry(_ts(t, -6), "INFO", "user-service", "DB connection pool: 12/100 active"),
                LogEntry(_ts(t, -2), "INFO", "user-service", "Profile cache warm"),
            ],
        ),
        "order-service": ServiceState(
            name="order-service",
            cpu_percent=20.0, memory_percent=32.0,
            latency_p50_ms=25.0, latency_p99_ms=80.0,
            error_rate_percent=0.15, request_rate_rps=600.0,
            replicas=3,
            dependencies=["database", "payment-service", "message-queue"],
            config={"max_retry_attempts": 3, "idempotency_window_seconds": 300},
            recent_deploys=[Deployment("v4.1.2", _ts(t, -2160), "ci-bot", "Bug fix: duplicate order prevention")],
            logs=[
                LogEntry(_ts(t, -7), "INFO", "order-service", "Order processing pipeline nominal"),
                LogEntry(_ts(t, -1), "INFO", "order-service", "Batch reconciliation completed"),
            ],
        ),
        "payment-service": ServiceState(
            name="payment-service",
            cpu_percent=10.0, memory_percent=20.0,
            latency_p50_ms=50.0, latency_p99_ms=200.0,
            error_rate_percent=0.08, request_rate_rps=300.0,
            replicas=3,
            dependencies=["database", "message-queue"],
            config={"payment_timeout_ms": 5000, "fraud_check_enabled": True},
            recent_deploys=[Deployment("v2.0.1", _ts(t, -5760), "ci-bot", "PCI compliance update")],
            logs=[
                LogEntry(_ts(t, -9), "INFO", "payment-service", "Payment gateway connection healthy"),
                LogEntry(_ts(t, -4), "INFO", "payment-service", "Transaction success rate: 99.9%"),
            ],
        ),
        "cache-service": ServiceState(
            name="cache-service",
            cpu_percent=8.0, memory_percent=45.0,
            latency_p50_ms=1.0, latency_p99_ms=3.0,
            error_rate_percent=0.01, request_rate_rps=5000.0,
            replicas=2,
            dependencies=[],
            config={"max_memory_mb": 4096, "eviction_policy": "lru", "max_connections": 500},
            recent_deploys=[Deployment("v1.5.0", _ts(t, -8640), "ci-bot", "Memory optimization")],
            logs=[
                LogEntry(_ts(t, -5), "INFO", "cache-service", "Hit rate: 92%, memory usage: 45%"),
                LogEntry(_ts(t, -1), "INFO", "cache-service", "Eviction rate nominal"),
            ],
        ),
        "database": ServiceState(
            name="database",
            cpu_percent=30.0, memory_percent=55.0,
            latency_p50_ms=3.0, latency_p99_ms=15.0,
            error_rate_percent=0.01, request_rate_rps=3000.0,
            replicas=1,
            dependencies=[],
            config={
                "pool_size": 100, "max_connections": 200,
                "wal_level": "replica", "shared_buffers_mb": 2048,
                "work_mem_mb": 64,
            },
            recent_deploys=[Deployment("v15.4", _ts(t, -43200), "dba-team", "PostgreSQL minor version upgrade")],
            logs=[
                LogEntry(_ts(t, -4), "INFO", "database", "Active connections: 42/200, pool: 42/100"),
                LogEntry(_ts(t, -1), "INFO", "database", "Replication lag: 0ms"),
            ],
        ),
        "message-queue": ServiceState(
            name="message-queue",
            cpu_percent=5.0, memory_percent=15.0,
            latency_p50_ms=2.0, latency_p99_ms=8.0,
            error_rate_percent=0.005, request_rate_rps=1500.0,
            replicas=3,
            dependencies=[],
            config={"max_queue_depth": 100000, "consumer_prefetch": 100, "message_ttl_seconds": 86400},
            recent_deploys=[Deployment("v3.12.0", _ts(t, -14400), "infra-team", "Cluster rebalancing")],
            logs=[
                LogEntry(_ts(t, -3), "INFO", "message-queue", "Queue depth: 142, consumers: 12"),
                LogEntry(_ts(t, -1), "INFO", "message-queue", "All partitions healthy"),
            ],
        ),
    }
    return services


# ---------------------------------------------------------------------------
# Fault injection for each task
# ---------------------------------------------------------------------------

def inject_task_easy(services: dict[str, ServiceState], base_time: datetime) -> None:
    """Task 1 (Easy): api-gateway OOM crash → service down."""
    gw = services["api-gateway"]
    gw.status = "down"
    gw.cpu_percent = 0.0
    gw.memory_percent = 0.0
    gw.latency_p50_ms = 0.0
    gw.latency_p99_ms = 0.0
    gw.error_rate_percent = 100.0
    gw.request_rate_rps = 0.0
    gw.replicas = 0
    gw.logs = [
        LogEntry(_ts(base_time, -30), "INFO", "api-gateway", "Health check passed"),
        LogEntry(_ts(base_time, -20), "WARN", "api-gateway", "Memory usage at 85% (3.4GB/4GB)"),
        LogEntry(_ts(base_time, -15), "WARN", "api-gateway", "GC pressure increasing, heap nearly full"),
        LogEntry(_ts(base_time, -10), "ERROR", "api-gateway", "java.lang.OutOfMemoryError: Java heap space"),
        LogEntry(_ts(base_time, -10), "FATAL", "api-gateway", "Process killed by OOM killer (exit code 137)"),
        LogEntry(_ts(base_time, -9), "ERROR", "api-gateway", "All replicas terminated. Service unreachable."),
        LogEntry(_ts(base_time, -8), "WARN", "api-gateway", "Kubernetes liveness probe failed: connection refused"),
        LogEntry(_ts(base_time, -5), "ERROR", "api-gateway", "Upstream healthcheck: 0/4 replicas responding"),
    ]


def inject_task_medium(services: dict[str, ServiceState], base_time: datetime) -> None:
    """Task 2 (Medium): DB connection pool reduced from 100→10 via bad config push."""
    db = services["database"]
    db.config["pool_size"] = 10
    db.status = "degraded"
    db.cpu_percent = 78.0
    db.memory_percent = 72.0
    db.latency_p50_ms = 180.0
    db.latency_p99_ms = 4500.0
    db.error_rate_percent = 12.0
    db.logs = [
        LogEntry(_ts(base_time, -60), "INFO", "database", "Config update applied: pool_size changed from 100 to 10"),
        LogEntry(_ts(base_time, -55), "INFO", "database", "Active connections: 10/10 (pool exhausted)"),
        LogEntry(_ts(base_time, -45), "WARN", "database", "Connection wait queue growing: 47 pending requests"),
        LogEntry(_ts(base_time, -30), "ERROR", "database", "Connection acquisition timeout after 5000ms for 23 requests"),
        LogEntry(_ts(base_time, -20), "WARN", "database", "Slow query log: avg query time spiked to 180ms (was 3ms)"),
        LogEntry(_ts(base_time, -10), "ERROR", "database", "Max pool connections (10) saturated. Rejecting overflow."),
        LogEntry(_ts(base_time, -5), "WARN", "database", "Replication lag increased to 450ms due to connection starvation"),
    ]
    db.recent_deploys.insert(0, Deployment(
        "config-v42", _ts(base_time, -60), "config-bot",
        "Automated config push: pool_size optimization (100 → 10)",
        rollback_available=True,
    ))

    for svc_name in ("user-service", "order-service"):
        svc = services[svc_name]
        svc.status = "degraded"
        svc.latency_p50_ms = 1800.0
        svc.latency_p99_ms = 8000.0
        svc.error_rate_percent = 15.0
        svc.logs.extend([
            LogEntry(_ts(base_time, -40), "WARN", svc_name, "Database query timeout: exceeded 5000ms threshold"),
            LogEntry(_ts(base_time, -25), "ERROR", svc_name, f"Failed to acquire DB connection: pool exhausted (waited 5000ms)"),
            LogEntry(_ts(base_time, -15), "WARN", svc_name, f"Circuit breaker OPEN for database — 15% error rate exceeds threshold"),
            LogEntry(_ts(base_time, -5), "ERROR", svc_name, "Request latency p99 > 8000ms. SLO breach."),
        ])

    # Red herring: user-service had a recent deploy (unrelated)
    services["user-service"].recent_deploys.insert(0, Deployment(
        "v3.2.1", _ts(base_time, -180), "deploy-svc",
        "Minor: updated user avatar resizing library",
        rollback_available=True,
    ))

    services["api-gateway"].latency_p99_ms = 5200.0
    services["api-gateway"].error_rate_percent = 8.0
    services["api-gateway"].status = "degraded"
    services["api-gateway"].logs.extend([
        LogEntry(_ts(base_time, -20), "WARN", "api-gateway", "Upstream user-service: elevated latency (p99=8000ms)"),
        LogEntry(_ts(base_time, -10), "WARN", "api-gateway", "Upstream order-service: elevated latency (p99=8000ms)"),
    ])


def inject_task_hard(services: dict[str, ServiceState], base_time: datetime) -> None:
    """Task 3 (Hard): cache-service memory leak after deploy → cascading auth + gateway failures."""
    cache = services["cache-service"]
    cache.status = "unhealthy"
    cache.cpu_percent = 55.0
    cache.memory_percent = 97.0
    cache.latency_p50_ms = 800.0
    cache.latency_p99_ms = 5000.0
    cache.error_rate_percent = 45.0
    cache.request_rate_rps = 1200.0
    cache.recent_deploys.insert(0, Deployment(
        "v1.6.0", _ts(base_time, -120), "ci-bot",
        "Feature: added session-aware cache partitioning",
        rollback_available=True,
    ))
    cache.logs = [
        LogEntry(_ts(base_time, -120), "INFO", "cache-service", "Deploy v1.6.0 started — session-aware partitioning"),
        LogEntry(_ts(base_time, -110), "INFO", "cache-service", "Deploy v1.6.0 complete. All health checks passed."),
        LogEntry(_ts(base_time, -90), "INFO", "cache-service", "Memory usage: 52% — within normal range"),
        LogEntry(_ts(base_time, -60), "WARN", "cache-service", "Memory usage: 71% — trending higher than expected"),
        LogEntry(_ts(base_time, -40), "WARN", "cache-service", "Memory usage: 84% — eviction rate spiked 10x"),
        LogEntry(_ts(base_time, -25), "ERROR", "cache-service", "Memory usage: 93% — OOM protection triggered, dropping new writes"),
        LogEntry(_ts(base_time, -15), "ERROR", "cache-service", "Cache miss rate: 68% (was 8%). Possible memory leak in session partition module."),
        LogEntry(_ts(base_time, -5), "ERROR", "cache-service", "Memory usage: 97% — service severely degraded. GET latency >800ms."),
    ]

    auth = services["auth-service"]
    auth.status = "unhealthy"
    auth.cpu_percent = 82.0
    auth.memory_percent = 70.0
    auth.latency_p50_ms = 1500.0
    auth.latency_p99_ms = 8000.0
    auth.error_rate_percent = 35.0
    auth.logs = [
        LogEntry(_ts(base_time, -50), "WARN", "auth-service", "Cache miss rate increasing — falling back to DB for token validation"),
        LogEntry(_ts(base_time, -35), "WARN", "auth-service", "DB fallback causing elevated latency: p50=800ms (was 5ms)"),
        LogEntry(_ts(base_time, -20), "ERROR", "auth-service", "Token validation failures: 35% of requests timing out"),
        LogEntry(_ts(base_time, -15), "ERROR", "auth-service", "Dependency cache-service: unhealthy. 45% error rate on cache reads."),
        LogEntry(_ts(base_time, -10), "WARN", "auth-service", "DB connection pool nearing capacity due to cache fallback load"),
        LogEntry(_ts(base_time, -5), "ERROR", "auth-service", "Authentication pipeline degraded. SLO breach: error_rate=35%"),
    ]

    db = services["database"]
    db.status = "degraded"
    db.cpu_percent = 65.0
    db.latency_p50_ms = 45.0
    db.latency_p99_ms = 350.0
    db.error_rate_percent = 2.0
    db.logs.extend([
        LogEntry(_ts(base_time, -30), "WARN", "database", "Unusual spike in auth-related queries (+400%)"),
        LogEntry(_ts(base_time, -15), "WARN", "database", "Active connections: 85/100 — approaching pool limit"),
    ])

    gw = services["api-gateway"]
    gw.status = "degraded"
    gw.latency_p50_ms = 2500.0
    gw.latency_p99_ms = 12000.0
    gw.error_rate_percent = 28.0
    gw.logs = [
        LogEntry(_ts(base_time, -30), "WARN", "api-gateway", "Upstream auth-service: elevated error rate (35%)"),
        LogEntry(_ts(base_time, -20), "ERROR", "api-gateway", "Circuit breaker HALF-OPEN for auth-service"),
        LogEntry(_ts(base_time, -15), "ERROR", "api-gateway", "28% of requests returning HTTP 500/503"),
        LogEntry(_ts(base_time, -10), "WARN", "api-gateway", "Auto-scaler triggered: scaling from 4 to 8 replicas"),
        LogEntry(_ts(base_time, -5), "ERROR", "api-gateway", "Scaling did not resolve errors — root cause is upstream"),
    ]
    gw.replicas = 8
    gw.cpu_percent = 45.0

    # Red herring: order-service had a recent deploy (unrelated)
    services["order-service"].recent_deploys.insert(0, Deployment(
        "v4.2.0", _ts(base_time, -180), "ci-bot",
        "Feature: express checkout flow",
        rollback_available=True,
    ))


# ---------------------------------------------------------------------------
# Simulation runtime — processes actions against the infrastructure state
# ---------------------------------------------------------------------------

class InfraSimulation:
    """Mutable simulation of a microservice infrastructure that processes agent actions."""

    def __init__(self, services: dict[str, ServiceState]):
        self.services = services
        self.action_log: list[dict[str, Any]] = []

    def service_names(self) -> list[str]:
        return sorted(self.services.keys())

    def get_service(self, name: str) -> ServiceState | None:
        return self.services.get(name)

    # -- action handlers ---------------------------------------------------

    def check_logs(self, target: str) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}"}
        return {
            "service": target,
            "log_entries": [
                {"timestamp": l.timestamp, "level": l.level, "message": l.message}
                for l in svc.logs
            ],
        }

    def check_metrics(self, target: str) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}"}
        return {
            "service": target,
            "status": svc.status,
            "cpu_percent": svc.cpu_percent,
            "memory_percent": svc.memory_percent,
            "latency_p50_ms": svc.latency_p50_ms,
            "latency_p99_ms": svc.latency_p99_ms,
            "error_rate_percent": svc.error_rate_percent,
            "request_rate_rps": svc.request_rate_rps,
            "replicas": svc.replicas,
        }

    def check_service_status(self, target: str | None) -> dict[str, Any]:
        if target:
            svc = self.services.get(target)
            if svc is None:
                return {"error": f"Unknown service: {target}"}
            return {"service": target, "status": svc.status, "replicas": svc.replicas}
        return {
            "services": {
                name: {"status": s.status, "replicas": s.replicas}
                for name, s in sorted(self.services.items())
            }
        }

    def check_dependencies(self, target: str | None) -> dict[str, Any]:
        if target:
            svc = self.services.get(target)
            if svc is None:
                return {"error": f"Unknown service: {target}"}
            dep_status = {}
            for dep in svc.dependencies:
                d = self.services.get(dep)
                dep_status[dep] = d.status if d else "unknown"
            return {
                "service": target,
                "dependencies": svc.dependencies,
                "dependency_status": dep_status,
                "config": svc.config,
                "recent_deploys": [
                    {"version": d.version, "deployed_at": d.deployed_at, "change_summary": d.change_summary}
                    for d in svc.recent_deploys
                ],
            }
        return {
            "dependency_graph": {
                name: s.dependencies for name, s in sorted(self.services.items())
            }
        }

    def restart_service(self, target: str) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}", "success": False}
        old_status = svc.status
        svc.status = "healthy"
        svc.cpu_percent = random.uniform(10, 25)
        svc.memory_percent = random.uniform(20, 40)
        svc.latency_p50_ms = random.uniform(5, 20)
        svc.latency_p99_ms = random.uniform(30, 80)
        svc.error_rate_percent = random.uniform(0.01, 0.2)
        if svc.replicas == 0:
            svc.replicas = 3
        svc.logs.append(LogEntry(
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "INFO", target, f"Service restarted. Status changed from {old_status} to healthy.",
        ))
        return {
            "success": True,
            "service": target,
            "previous_status": old_status,
            "new_status": "healthy",
            "message": f"{target} has been restarted successfully. All replicas are healthy.",
        }

    def scale_service(self, target: str, params: dict[str, Any] | None) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}", "success": False}
        new_replicas = (params or {}).get("replicas", svc.replicas + 1)
        old = svc.replicas
        svc.replicas = new_replicas
        return {"success": True, "service": target, "previous_replicas": old, "new_replicas": new_replicas}

    def rollback_deploy(self, target: str) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}", "success": False}
        if not svc.recent_deploys:
            return {"error": f"No recent deployments for {target}", "success": False}
        deploy = svc.recent_deploys[0]
        if not deploy.rollback_available:
            return {"error": f"Rollback not available for {deploy.version}", "success": False}
        rolled_back_version = deploy.version
        svc.recent_deploys.pop(0)
        svc.status = "healthy"
        svc.cpu_percent = random.uniform(8, 20)
        svc.memory_percent = random.uniform(25, 50)
        svc.latency_p50_ms = random.uniform(1, 15)
        svc.latency_p99_ms = random.uniform(5, 50)
        svc.error_rate_percent = random.uniform(0.01, 0.1)
        svc.logs.append(LogEntry(
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "INFO", target,
            f"Rolled back deployment {rolled_back_version}. Service restored to previous version.",
        ))
        return {
            "success": True,
            "service": target,
            "rolled_back_version": rolled_back_version,
            "new_status": "healthy",
            "message": f"Successfully rolled back {target} from {rolled_back_version}. Service is healthy.",
        }

    def update_config(self, target: str, params: dict[str, Any] | None) -> dict[str, Any]:
        svc = self.services.get(target)
        if svc is None:
            return {"error": f"Unknown service: {target}", "success": False}
        if not params:
            return {"error": "No configuration parameters provided", "success": False}
        changes = {}
        for key, value in params.items():
            if key in svc.config:
                old = svc.config[key]
                svc.config[key] = value
                changes[key] = {"old": old, "new": value}
        if not changes:
            return {"error": f"None of the provided keys exist in {target} config", "success": False}
        svc.logs.append(LogEntry(
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "INFO", target, f"Configuration updated: {changes}",
        ))
        return {"success": True, "service": target, "changes": changes}

    def escalate(self) -> dict[str, Any]:
        return {
            "success": True,
            "message": "Incident escalated to the senior on-call engineer and service owners.",
        }

    def process_action(self, action_type: str, target: str | None, params: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
        """Process an action and return (data, message)."""
        record = {"action_type": action_type, "target_service": target, "parameters": params}
        self.action_log.append(record)

        if action_type == "check_logs":
            if not target:
                return {"error": "target_service is required for check_logs"}, "Please specify a target_service."
            data = self.check_logs(target)
            msg = f"Retrieved {len(data.get('log_entries', []))} log entries for {target}." if "error" not in data else data["error"]
            return data, msg

        if action_type == "check_metrics":
            if not target:
                return {"error": "target_service is required for check_metrics"}, "Please specify a target_service."
            data = self.check_metrics(target)
            msg = f"Metrics for {target}: status={data.get('status')}, latency_p99={data.get('latency_p99_ms')}ms, error_rate={data.get('error_rate_percent')}%" if "error" not in data else data["error"]
            return data, msg

        if action_type == "check_service_status":
            data = self.check_service_status(target)
            if target:
                msg = f"{target} status: {data.get('status')}" if "error" not in data else data["error"]
            else:
                statuses = data.get("services", {})
                unhealthy = [n for n, s in statuses.items() if s["status"] != "healthy"]
                msg = f"Infrastructure overview: {len(unhealthy)} service(s) not healthy: {', '.join(unhealthy)}" if unhealthy else "All services healthy."
            return data, msg

        if action_type == "check_dependencies":
            data = self.check_dependencies(target)
            msg = f"Dependency info for {target}." if target else "Full dependency graph retrieved."
            return data, msg

        if action_type == "restart_service":
            if not target:
                return {"error": "target_service is required for restart_service"}, "Please specify a target_service."
            data = self.restart_service(target)
            return data, data.get("message", str(data))

        if action_type == "scale_service":
            if not target:
                return {"error": "target_service is required for scale_service"}, "Please specify a target_service."
            data = self.scale_service(target, params)
            msg = f"Scaled {target} from {data.get('previous_replicas')} to {data.get('new_replicas')} replicas." if data.get("success") else data.get("error", "")
            return data, msg

        if action_type == "rollback_deploy":
            if not target:
                return {"error": "target_service is required for rollback_deploy"}, "Please specify a target_service."
            data = self.rollback_deploy(target)
            return data, data.get("message", str(data))

        if action_type == "update_config":
            if not target:
                return {"error": "target_service is required for update_config"}, "Please specify a target_service."
            data = self.update_config(target, params)
            if data.get("success"):
                msg = f"Configuration for {target} updated: {data.get('changes')}"
            else:
                msg = data.get("error", "Config update failed.")
            return data, msg

        if action_type == "escalate":
            data = self.escalate()
            return data, data["message"]

        if action_type == "resolve":
            return {"acknowledged": True}, "Incident resolution acknowledged."

        return {"error": f"Unknown action type: {action_type}"}, f"Unknown action: {action_type}"
