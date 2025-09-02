import time
import statistics
import pytest
from app.optimizer import PetPoojaOptimizer

class TestPerformance:
    def setup_class(self):
        self.optimizer = PetPoojaOptimizer()
        self.test_queries = [
            "add biryani ₹300",
            "check rice stock",
            "show sales for today",
            "I have a billing problem",
            "add 5kg flour to ingredients"
        ]
    
    def measure_latency(self, query, iterations=1000):
        """Measure average latency for a query."""
        # Warm-up
        self.optimizer.optimize_prompt(query)
        
        # Measure
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            self.optimizer.optimize_prompt(query)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'min': min(times),
            'max': max(times),
            'avg': statistics.mean(times),
            'p95': statistics.quantiles(times, n=20)[-1]  # 95th percentile
        }
    
    @pytest.mark.performance
    def test_query_latency(self):
        """Test that query latency is within acceptable limits."""
        max_avg_latency = 50  # ms
        max_p95_latency = 100  # ms
        
        for query in self.test_queries:
            metrics = self.measure_latency(query)
            
            print(f"\nPerformance for query: {query}")
            print(f"  Average: {metrics['avg']:.2f}ms")
            print(f"  P95: {metrics['p95']:.2f}ms")
            
            assert metrics['avg'] <= max_avg_latency, \
                f"Average latency {metrics['avg']:.2f}ms exceeds {max_avg_latency}ms for query: {query}"
            assert metrics['p95'] <= max_p95_latency, \
                f"P95 latency {metrics['p95']:.2f}ms exceeds {max_p95_latency}ms for query: {query}"
    
    @pytest.mark.performance
    def test_concurrent_requests(self):
        """Test performance under concurrent load."""
        import threading
        import queue
        
        num_threads = 10
        iterations = 100
        results = queue.Queue()
        
        def worker():
            for _ in range(iterations):
                for query in self.test_queries:
                    start = time.perf_counter()
                    self.optimizer.optimize_prompt(query)
                    end = time.perf_counter()
                    results.put((end - start) * 1000)  # ms
        
        # Start worker threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Calculate metrics
        times = []
        while not results.empty():
            times.append(results.get())
        
        avg_latency = statistics.mean(times)
        p95_latency = statistics.quantiles(times, n=20)[-1]
        
        print(f"\nConcurrent load test results (10 threads, 100 iterations):")
        print(f"  Total requests: {len(times)}")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        
        # Assert performance requirements
        assert avg_latency <= 100, f"Average concurrent latency {avg_latency:.2f}ms exceeds 100ms"
        assert p95_latency <= 200, f"P95 concurrent latency {p95_latency:.2f}ms exceeds 200ms"
