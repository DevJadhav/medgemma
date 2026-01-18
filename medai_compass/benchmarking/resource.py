"""
Resource benchmarks for MedAI Compass.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceResult:
    """Resource benchmark result."""
    
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    duration_seconds: float = 0.0


class MemoryBenchmark:
    """
    Memory usage benchmark.
    """
    
    def __init__(self):
        """Initialize memory benchmark."""
        pass
    
    def run(self) -> ResourceResult:
        """
        Run memory benchmark.
        
        Returns:
            ResourceResult with memory metrics
        """
        try:
            import psutil
            
            process = psutil.Process()
            
            # Current memory
            mem_info = process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            
            return ResourceResult(
                peak_memory_mb=current_mb,
                avg_memory_mb=current_mb,
            )
            
        except ImportError:
            logger.warning("psutil not available for memory benchmark")
            return ResourceResult()
        except Exception as e:
            logger.error(f"Memory benchmark error: {e}")
            return ResourceResult()


class GPUMemoryBenchmark:
    """
    GPU memory usage benchmark.
    """
    
    def __init__(self):
        """Initialize GPU memory benchmark."""
        pass
    
    def run(self) -> ResourceResult:
        """
        Run GPU memory benchmark.
        
        Returns:
            ResourceResult with GPU memory metrics
        """
        try:
            import torch
            
            if not torch.cuda.is_available():
                return ResourceResult(gpu_memory_mb=0)
            
            # Current GPU memory
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            
            return ResourceResult(
                gpu_memory_mb=allocated,
            )
            
        except ImportError:
            return ResourceResult(gpu_memory_mb=0)
        except Exception as e:
            logger.debug(f"GPU benchmark error: {e}")
            return ResourceResult(gpu_memory_mb=0)


class CPUBenchmark:
    """
    CPU utilization benchmark.
    """
    
    def __init__(self, duration_seconds: float = 5.0):
        """
        Initialize CPU benchmark.
        
        Args:
            duration_seconds: Monitoring duration
        """
        self.duration_seconds = duration_seconds
    
    def run(self) -> ResourceResult:
        """
        Run CPU benchmark.
        
        Returns:
            ResourceResult with CPU metrics
        """
        try:
            import psutil
            
            # Sample CPU over duration
            samples = []
            start = time.perf_counter()
            
            while time.perf_counter() - start < self.duration_seconds:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                samples.append(cpu_percent)
            
            avg_cpu = sum(samples) / len(samples) if samples else 0
            
            return ResourceResult(
                cpu_percent=avg_cpu,
                duration_seconds=self.duration_seconds,
            )
            
        except ImportError:
            logger.warning("psutil not available for CPU benchmark")
            return ResourceResult()
        except Exception as e:
            logger.error(f"CPU benchmark error: {e}")
            return ResourceResult()
