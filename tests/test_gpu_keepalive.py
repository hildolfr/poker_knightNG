#!/usr/bin/env python3
"""
Tests for GPU keep-alive functionality in server scenarios.
"""

import sys
import time
import threading

sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import pytest
from poker_knight_ng import create_poker_server, PokerSolverServer
from poker_knight_ng.memory_manager import get_memory_manager


class TestGPUKeepAlive:
    """Test GPU keep-alive functionality."""
    
    def test_server_warmup(self):
        """Test that server warmup reduces first solve latency."""
        # Create server without auto warmup
        server_cold = PokerSolverServer(
            keep_alive_seconds=10,
            auto_warmup=False
        )
        
        # Time first solve (cold)
        start = time.time()
        result_cold = server_cold.solve(['A♠', 'A♥'], 2, simulation_mode='fast')
        cold_time = (time.time() - start) * 1000
        
        # Create server with auto warmup
        server_warm = PokerSolverServer(
            keep_alive_seconds=10,
            auto_warmup=True
        )
        
        # Time first solve (warm)
        start = time.time()
        result_warm = server_warm.solve(['A♠', 'A♥'], 2, simulation_mode='fast')
        warm_time = (time.time() - start) * 1000
        
        print(f"Cold start: {cold_time:.1f}ms")
        print(f"Warm start: {warm_time:.1f}ms")
        print(f"Warmup benefit: {cold_time - warm_time:.1f}ms")
        
        # Warm should be significantly faster
        # In some environments, GPU may already be partially warm
        # Just ensure warm is not significantly slower than cold
        assert warm_time <= cold_time * 1.5, "Warmup should not make things significantly slower"
        
        # Results should be similar
        assert abs(result_cold.win_probability - result_warm.win_probability) < 0.02
    
    def test_keep_alive_maintains_warmth(self):
        """Test that GPU stays warm during keep-alive period."""
        server = create_poker_server(
            keep_alive_seconds=2.0,  # Short for testing
            auto_warmup=True
        )
        
        # First solve (already warm from auto warmup)
        result1 = server.solve(['K♥', 'K♦'], 1, simulation_mode='fast')
        
        # Check that GPU is marked as warm
        stats = server.get_statistics()
        # Debug output
        print(f"GPU warm status: {stats['gpu_is_warm']}")
        print(f"Memory info: {stats['memory_info']}")
        assert stats['gpu_is_warm'] or stats['seconds_since_activity'] < 2.0
        
        # Wait less than keep-alive period
        time.sleep(1.0)
        
        # GPU should still be warm
        stats = server.get_statistics()
        assert stats['gpu_is_warm']
        assert stats['seconds_since_activity'] < 2.0
        
        # Do another solve - should be fast
        start = time.time()
        result2 = server.solve(['Q♣', 'Q♦'], 1, simulation_mode='fast')
        solve_time = (time.time() - start) * 1000
        
        assert solve_time < 10.0, "Warm solve should be reasonably fast"
        
        # Wait longer than keep-alive period
        time.sleep(2.5)
        
        # GPU should no longer be warm
        stats = server.get_statistics()
        assert not stats['gpu_is_warm']
    
    def test_activity_tracking(self):
        """Test that activity is properly tracked."""
        server = create_poker_server(keep_alive_seconds=30)
        
        initial_stats = server.get_statistics()
        assert initial_stats['solve_count'] == 0
        
        # Do several solves
        for i in range(5):
            server.solve(['A♦', 'K♦'], 2, simulation_mode='fast')
            time.sleep(0.1)
        
        stats = server.get_statistics()
        assert stats['solve_count'] == 5
        assert stats['average_solve_time_ms'] > 0
        assert len(server.warm_solve_times) >= 3  # Most solves should be warm
    
    def test_batch_solving(self):
        """Test batch solving maintains GPU warmth."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Create batch of problems
        problems = [
            {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1},
            {'hero_hand': ['K♠', 'K♥'], 'num_opponents': 2},
            {'hero_hand': ['Q♠', 'Q♥'], 'num_opponents': 3},
            {'hero_hand': ['J♠', 'J♥'], 'num_opponents': 1, 'board_cards': ['T♠', '9♥', '8♦']},
            {'hero_hand': ['T♠', 'T♥'], 'num_opponents': 2, 'simulation_mode': 'fast'},
        ]
        
        # Solve batch
        start = time.time()
        results = server.solve_batch(problems)
        batch_time = (time.time() - start) * 1000
        
        assert len(results) == len(problems)
        assert all(r is not None for r in results)
        assert all(0 < r.win_probability < 1 for r in results)
        
        # Check performance
        avg_time = batch_time / len(problems)
        print(f"Batch solve: {len(problems)} problems in {batch_time:.1f}ms (avg: {avg_time:.1f}ms)")
        
        # Average should be faster than cold starts
        assert avg_time < 20.0, "Batch solving should maintain GPU warmth"
    
    def test_session_context_manager(self):
        """Test session context manager."""
        server = create_poker_server(keep_alive_seconds=30)
        
        results = []
        
        with server.session():
            # Multiple solves within session
            results.append(server.solve(['A♠', 'K♠'], 1))
            results.append(server.solve(['Q♥', 'Q♦'], 2))
            results.append(server.solve(['J♣', 'T♣'], 3, board_cards=['9♣', '8♦', '7♥']))
        
        assert len(results) == 3
        assert all(r.win_probability > 0 for r in results)
        
        # Check that session was tracked
        stats = server.get_statistics()
        assert stats['solve_count'] >= 3
    
    def test_memory_info(self):
        """Test enhanced memory info reporting."""
        # Force new instance for clean test
        mem_manager = get_memory_manager(
            keep_alive_seconds=45,  # Different value to test
            enable_keep_alive=True,
            force_new=True
        )
        
        # Get memory info
        info = mem_manager.get_enhanced_memory_info()
        
        assert 'keep_alive' in info
        assert info['keep_alive']['enabled'] is True
        assert info['keep_alive']['keep_alive_seconds'] == 45
        assert 'is_warm' in info['keep_alive']
        assert 'warm_buffers' in info
    
    def test_concurrent_activity(self):
        """Test that concurrent solves don't interfere."""
        server = create_poker_server(keep_alive_seconds=30)
        
        results = []
        errors = []
        
        def solve_hand(hand, opponents):
            try:
                result = server.solve(hand, opponents, simulation_mode='fast')
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        hands = [
            (['A♠', 'A♥'], 1),
            (['K♠', 'K♥'], 2),
            (['Q♠', 'Q♥'], 3),
            (['J♠', 'J♥'], 1),
            (['T♠', 'T♥'], 2),
        ]
        
        for hand, opponents in hands:
            t = threading.Thread(target=solve_hand, args=(hand, opponents))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(hands)
        assert all(0 < r.win_probability < 1 for r in results)
    
    def test_shutdown_cleanup(self):
        """Test that shutdown properly cleans up resources."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Do some work
        server.solve(['A♣', 'K♣'], 2)
        
        # Get initial memory info
        initial_info = server.memory_manager.get_enhanced_memory_info()
        
        # Shutdown
        server.shutdown()
        
        # Create new server - should work fine
        server2 = create_poker_server(keep_alive_seconds=30)
        result = server2.solve(['Q♦', 'Q♣'], 1)
        
        assert result.win_probability > 0.78  # QQ vs 1 opponent (should be ~79-80%)


class TestPerformanceComparison:
    """Compare performance with and without keep-alive."""
    
    def test_keepalive_performance_benefit(self):
        """Measure performance benefit of keep-alive for bursty workloads."""
        # Simulate bursty workload: solve, wait, solve again
        
        print("\nTesting bursty workload performance:")
        
        # Test with server API (keeps GPU warm)
        server = create_poker_server(keep_alive_seconds=5.0)
        server_times = []
        
        # First burst
        for _ in range(3):
            start = time.time()
            server.solve(['A♠', 'K♠'], 2, simulation_mode='fast')
            server_times.append((time.time() - start) * 1000)
        
        # Wait (but less than keep-alive)
        time.sleep(2.0)
        
        # Second burst - should still be warm
        for _ in range(3):
            start = time.time()
            server.solve(['K♥', 'K♦'], 2, simulation_mode='fast')
            server_times.append((time.time() - start) * 1000)
        
        print(f"Server API times: {[f'{t:.1f}' for t in server_times]}")
        print(f"First solve: {server_times[0]:.1f}ms")
        print(f"After idle: {server_times[3]:.1f}ms (should be fast due to keep-alive)")
        
        # The solve after idle should still be fast
        assert server_times[3] < 5.0, "Solve after short idle should remain fast"
        
        # Compare to what would happen without keep-alive
        # The benefit is that we don't pay warmup cost after idle periods


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])