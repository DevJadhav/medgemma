"""
Load testing for MedAI Compass.
"""

import json
import logging
import random
import statistics
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""
    
    duration_seconds: int = 60
    users: int = 10
    ramp_up_seconds: int = 5
    requests_per_second: Optional[float] = None
    
    @classmethod
    def smoke_test(cls) -> "LoadTestConfig":
        """Quick smoke test."""
        return cls(
            duration_seconds=10,
            users=5,
            ramp_up_seconds=2,
        )
    
    @classmethod
    def ci_config(cls) -> "LoadTestConfig":
        """CI-friendly config."""
        return cls(
            duration_seconds=30,
            users=10,
            ramp_up_seconds=5,
        )
    
    @classmethod
    def full_test(cls) -> "LoadTestConfig":
        """Full load test."""
        return cls(
            duration_seconds=300,
            users=50,
            ramp_up_seconds=30,
        )


@dataclass
class LoadTestScenario:
    """Predefined load test scenario."""
    
    name: str
    users: int
    duration_seconds: int
    ramp_up_seconds: int
    
    @classmethod
    def normal_load(cls) -> "LoadTestScenario":
        """Normal operating load."""
        return cls(
            name="normal_load",
            users=20,
            duration_seconds=120,
            ramp_up_seconds=10,
        )
    
    @classmethod
    def peak_load(cls) -> "LoadTestScenario":
        """Peak load scenario."""
        return cls(
            name="peak_load",
            users=100,
            duration_seconds=180,
            ramp_up_seconds=30,
        )
    
    @classmethod
    def stress_test(cls) -> "LoadTestScenario":
        """Stress test scenario."""
        return cls(
            name="stress_test",
            users=200,
            duration_seconds=300,
            ramp_up_seconds=60,
        )
    
    @classmethod
    def spike_test(cls) -> "LoadTestScenario":
        """Spike test scenario."""
        return cls(
            name="spike_test",
            users=150,
            duration_seconds=60,
            ramp_up_seconds=5,  # Fast ramp-up
        )


@dataclass
class LoadTestResult:
    """Result of a load test."""
    
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    response_times: List[float]
    requests_per_second: float
    mean_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    concurrent_users: int
    duration_seconds: float
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @classmethod
    def from_response_times(
        cls,
        scenario_name: str,
        response_times: List[float],
        errors: int,
        duration_seconds: float,
        concurrent_users: int,
    ) -> "LoadTestResult":
        """Create result from response times."""
        if not response_times:
            return cls(
                scenario_name=scenario_name,
                total_requests=errors,
                successful_requests=0,
                failed_requests=errors,
                error_rate=1.0,
                response_times=[],
                requests_per_second=0,
                mean_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                concurrent_users=concurrent_users,
                duration_seconds=duration_seconds,
            )
        
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        total = n + errors
        
        return cls(
            scenario_name=scenario_name,
            total_requests=total,
            successful_requests=n,
            failed_requests=errors,
            error_rate=errors / total if total > 0 else 0,
            response_times=response_times,
            requests_per_second=n / duration_seconds if duration_seconds > 0 else 0,
            mean_response_time_ms=statistics.mean(response_times),
            p50_response_time_ms=sorted_times[int(n * 0.50)],
            p95_response_time_ms=sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1],
            p99_response_time_ms=sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1],
            concurrent_users=concurrent_users,
            duration_seconds=duration_seconds,
        )
    
    def passes_sla(self, targets) -> bool:
        """Check if result passes SLA."""
        return (
            self.p95_response_time_ms <= targets.latency_p95_ms and
            self.error_rate <= targets.max_error_rate
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.error_rate,
            "requests_per_second": self.requests_per_second,
            "mean_response_time_ms": self.mean_response_time_ms,
            "p50_response_time_ms": self.p50_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "concurrent_users": self.concurrent_users,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


@dataclass
class LoadTestReport:
    """Collection of load test results."""
    
    results: List[LoadTestResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_summary(self) -> str:
        """Get summary report."""
        lines = ["Load Test Report", "=" * 50]
        
        for result in self.results:
            lines.append(f"\n{result.scenario_name}:")
            lines.append(f"  Total Requests: {result.total_requests}")
            lines.append(f"  Successful: {result.successful_requests}")
            lines.append(f"  Failed: {result.failed_requests}")
            lines.append(f"  Error Rate: {result.error_rate * 100:.2f}%")
            lines.append(f"  RPS: {result.requests_per_second:.2f}")
            lines.append(f"  Mean RT: {result.mean_response_time_ms:.2f}ms")
            lines.append(f"  P95 RT: {result.p95_response_time_ms:.2f}ms")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export as JSON."""
        return json.dumps({
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }, indent=2)


class LoadTestRunner:
    """
    Runs load tests against API.
    """
    
    def __init__(
        self,
        config: LoadTestConfig,
        api_client: Any,
    ):
        """
        Initialize load test runner.
        
        Args:
            config: Load test configuration
            api_client: HTTP client for API calls
        """
        self.config = config
        self.api_client = api_client
    
    def run(self) -> LoadTestResult:
        """Run load test."""
        response_times: List[float] = []
        error_count = 0
        lock = threading.Lock()
        stop_flag = threading.Event()
        
        def worker(user_id: int):
            nonlocal error_count
            local_times = []
            local_errors = 0
            
            while not stop_flag.is_set():
                start = time.perf_counter()
                
                try:
                    self.api_client.get("/health")
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    local_times.append(elapsed_ms)
                except Exception:
                    local_errors += 1
                
                # Small delay between requests
                time.sleep(0.1)
            
            with lock:
                response_times.extend(local_times)
                error_count += local_errors
        
        # Run test
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.config.users) as executor:
            futures = [
                executor.submit(worker, i)
                for i in range(self.config.users)
            ]
            
            time.sleep(self.config.duration_seconds)
            stop_flag.set()
            
            for f in futures:
                f.result()
        
        duration = time.perf_counter() - start_time
        
        return LoadTestResult.from_response_times(
            scenario_name="load_test",
            response_times=response_times,
            errors=error_count,
            duration_seconds=duration,
            concurrent_users=self.config.users,
        )


class ConcurrentUserSimulator:
    """
    Simulates concurrent user load.
    """
    
    def __init__(
        self,
        endpoint: Callable,
        num_users: int = 10,
        duration_seconds: float = 10.0,
        ramp_up_seconds: float = 0.0,
    ):
        """
        Initialize simulator.
        
        Args:
            endpoint: Function to call
            num_users: Number of concurrent users
            duration_seconds: Test duration
            ramp_up_seconds: Ramp-up period
        """
        self.endpoint = endpoint
        self.num_users = num_users
        self.duration_seconds = duration_seconds
        self.ramp_up_seconds = ramp_up_seconds
    
    def run(self) -> LoadTestResult:
        """Run simulation."""
        response_times: List[float] = []
        errors = 0
        lock = threading.Lock()
        stop_flag = threading.Event()
        
        def worker():
            nonlocal errors
            local_times = []
            local_errors = 0
            
            while not stop_flag.is_set():
                start = time.perf_counter()
                
                try:
                    self.endpoint()
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    local_times.append(elapsed_ms)
                except Exception:
                    local_errors += 1
            
            with lock:
                response_times.extend(local_times)
                errors += local_errors
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.num_users) as executor:
            futures = [
                executor.submit(worker)
                for _ in range(self.num_users)
            ]
            
            time.sleep(self.duration_seconds)
            stop_flag.set()
            
            for f in futures:
                f.result()
        
        duration = time.perf_counter() - start_time
        
        return LoadTestResult.from_response_times(
            scenario_name="concurrent_simulation",
            response_times=response_times,
            errors=errors,
            duration_seconds=duration,
            concurrent_users=self.num_users,
        )


class APILoadTest:
    """
    Load test for specific API endpoint.
    """
    
    def __init__(
        self,
        client: Any,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict] = None,
        concurrent_users: int = 10,
        duration_seconds: float = 10.0,
    ):
        """Initialize API load test."""
        self.client = client
        self.endpoint = endpoint
        self.method = method.upper()
        self.payload = payload
        self.concurrent_users = concurrent_users
        self.duration_seconds = duration_seconds
    
    def run(self) -> LoadTestResult:
        """Run API load test."""
        response_times: List[float] = []
        errors = 0
        lock = threading.Lock()
        stop_flag = threading.Event()
        
        def make_request():
            if self.method == "GET":
                return self.client.get(self.endpoint)
            elif self.method == "POST":
                return self.client.post(self.endpoint, json=self.payload)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        
        def worker():
            nonlocal errors
            local_times = []
            local_errors = 0
            
            while not stop_flag.is_set():
                start = time.perf_counter()
                
                try:
                    response = make_request()
                    if response.status_code < 400:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        local_times.append(elapsed_ms)
                    else:
                        local_errors += 1
                except Exception:
                    local_errors += 1
                
                time.sleep(0.05)  # 50ms between requests
            
            with lock:
                response_times.extend(local_times)
                errors += local_errors
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(self.concurrent_users)]
            
            time.sleep(self.duration_seconds)
            stop_flag.set()
            
            for f in futures:
                f.result()
        
        duration = time.perf_counter() - start_time
        
        return LoadTestResult.from_response_times(
            scenario_name=f"api_{self.endpoint}",
            response_times=response_times,
            errors=errors,
            duration_seconds=duration,
            concurrent_users=self.concurrent_users,
        )


class MixedLoadTest:
    """
    Load test with mixed endpoint distribution.
    """
    
    def __init__(
        self,
        client: Any,
        endpoints: List[Dict],  # [{"endpoint": "/health", "method": "GET", "weight": 50}]
        concurrent_users: int = 10,
        duration_seconds: float = 10.0,
    ):
        """Initialize mixed load test."""
        self.client = client
        self.endpoints = endpoints
        self.concurrent_users = concurrent_users
        self.duration_seconds = duration_seconds
    
    def run(self) -> LoadTestResult:
        """Run mixed load test."""
        response_times: List[float] = []
        errors = 0
        lock = threading.Lock()
        stop_flag = threading.Event()
        
        # Build weighted endpoint list
        weighted_endpoints = []
        for ep in self.endpoints:
            weight = ep.get("weight", 1)
            weighted_endpoints.extend([ep] * weight)
        
        def worker():
            nonlocal errors
            local_times = []
            local_errors = 0
            
            while not stop_flag.is_set():
                ep = random.choice(weighted_endpoints)
                start = time.perf_counter()
                
                try:
                    if ep.get("method", "GET") == "GET":
                        response = self.client.get(ep["endpoint"])
                    else:
                        response = self.client.post(
                            ep["endpoint"],
                            json=ep.get("payload", {})
                        )
                    
                    if response.status_code < 400:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        local_times.append(elapsed_ms)
                    else:
                        local_errors += 1
                except Exception:
                    local_errors += 1
                
                time.sleep(0.05)
            
            with lock:
                response_times.extend(local_times)
                errors += local_errors
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(self.concurrent_users)]
            
            time.sleep(self.duration_seconds)
            stop_flag.set()
            
            for f in futures:
                f.result()
        
        duration = time.perf_counter() - start_time
        
        return LoadTestResult.from_response_times(
            scenario_name="mixed_load",
            response_times=response_times,
            errors=errors,
            duration_seconds=duration,
            concurrent_users=self.concurrent_users,
        )
