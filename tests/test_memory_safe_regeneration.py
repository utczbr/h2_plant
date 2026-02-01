import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from h2_plant.visualization.streaming_downsampler import StreamingDownsampler, MemoryMonitor
from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
from h2_plant.visualization.graph_catalog import GraphCatalog, GraphMetadata, GraphLibrary, GraphPriority

@pytest.fixture
def empty_catalog():
    """Create a pristine empty catalog."""
    catalog = GraphCatalog()
    # Force clear everything loaded by default
    catalog._registry.clear()
    catalog._enabled_graphs.clear()
    return catalog

@pytest.fixture
def temp_dirs():
    """Create temp directories for chunks and output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        chunks_dir = tmp_path / "chunks"
        d = tmp_path / "output"
        chunks_dir.mkdir()
        d.mkdir()
        yield chunks_dir, d

def create_synthetic_chunk(path: Path, start_min: int, end_min: int):
    """Create a synthetic parquet chunk."""
    minutes = np.arange(start_min, end_min)
    data = {
        'minute': minutes,
        'temperature': np.random.rand(len(minutes)) * 100,
        'pressure': np.random.rand(len(minutes)) * 10,
        'power': np.random.rand(len(minutes)) * 50
    }
    df = pd.DataFrame(data)
    df.to_parquet(path)
    return df

class TestStreamingDownsampler:
    
    def test_downsampling_logic(self, temp_dirs):
        chunks_dir, output_dir = temp_dirs
        
        # Create 2 chunks of 60 mins each (1-min resolution)
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60)
        create_synthetic_chunk(chunks_dir / "chunk_2.parquet", 60, 120)
        
        # Target: 10-min resolution
        downsampler = StreamingDownsampler(
            max_memory_mb=1000,
            target_resolution_minutes=10 
        )
        
        output_cache = output_dir / "cache.parquet"
        result_df = downsampler.process_chunks_directory(chunks_dir, output_cache)
        
        # Expected: 2 chunks * (60/10 = 6 rows) = 12 rows
        assert len(result_df) == 12
        assert output_cache.exists()
        
        # Check resolution (diff should be approx 10)
        diffs = result_df['minute'].diff().dropna()
        assert np.allclose(diffs, 10.0)

    def test_column_filtering(self, temp_dirs):
        chunks_dir, output_dir = temp_dirs
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60)
        
        downsampler = StreamingDownsampler(max_memory_mb=1000)
        output_cache = output_dir / "filtered_cache.parquet"
        
        # Request specific columns
        required = ['minute', 'temperature']
        result_df = downsampler.process_chunks_directory(
            chunks_dir, output_cache, required_columns=required
        )
        
        assert 'pressure' not in result_df.columns
        assert 'power' not in result_df.columns
        assert 'temperature' in result_df.columns
        assert len(result_df) == 1 # 60 min chunk, 60 min default target = 1 row

    def test_emergency_consolidation(self, temp_dirs):
        chunks_dir, output_dir = temp_dirs
        
        # Create 3 chunks to simulate sequence
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60) # 0-59
        create_synthetic_chunk(chunks_dir / "chunk_2.parquet", 60, 120) # 60-119
        create_synthetic_chunk(chunks_dir / "chunk_3.parquet", 120, 180) # 120-179
        
        # Initialize with low memory to trigger pressure
        downsampler = StreamingDownsampler(
            max_memory_mb=10, 
            target_resolution_minutes=1,
            emergency_resolution_minutes=10
        )
        
        output_cache = output_dir / "emergency_cache.parquet"
        
        # Mock memory check to force pressure at chunk 2
        # Calls: Chunk 1 (False), Chunk 2 (True), Chunk 3 (True - already in mode)
        with patch.object(downsampler, '_check_memory_pressure', side_effect=[False, True, True]):
             result_df = downsampler.process_chunks_directory(chunks_dir, output_cache)
             
             # Check data presence
             assert len(result_df) > 0
             assert result_df.minute.min() == 0 # Must preserve start
             assert result_df.minute.max() >= 170 # Must reach end
             
             # Check resolution
             # Should be mostly 10-min resolution (emergency)
             diffs = result_df['minute'].diff().dropna()
             mean_res = diffs.mean()
             assert mean_res >= 5.0 # Should be closer to 10 than 1

class TestMemoryMonitor:
    def test_thresholds(self):
        monitor = MemoryMonitor(max_memory_mb=100)
        
        with patch('psutil.Process') as mock_proc:
            mock_proc.return_value.memory_info.return_value.rss = 50 * 10**6 # 50 MB
            monitor._process = mock_proc.return_value
            
            assert monitor.get_pressure() == 0.5
            assert not monitor.is_critical()
            assert monitor.has_headroom(40) # 100 - 50 = 50 > 40
            
            mock_proc.return_value.memory_info.return_value.rss = 90 * 10**6 # 90 MB
            assert monitor.is_critical()
            assert not monitor.has_headroom(20)

class TestUnifiedExecutorBatched:
    def test_batched_execution(self, temp_dirs, empty_catalog):
        _, output_dir = temp_dirs
        
        # Create catalog with mocked graphs
        catalog = empty_catalog
        for i in range(5):
             meta = GraphMetadata(
                 graph_id=f"g{i}",
                 title=f"Graph {i}",
                 description=f"Test Graph {i}",
                 function=lambda df, dpi: None, # No-op
                 library=GraphLibrary.MATPLOTLIB,
                 data_required=['minute'],
                 priority=GraphPriority.MEDIUM,
                 category="test",
                 enabled=True
             )
             catalog.register(meta)
             
        executor = UnifiedGraphExecutor(catalog, output_dir)
        
        # execute_batched request batch_size=2
        # Should result in 3 batches (2, 2, 1)
        
        # Mock execute to verify calls
        executor.execute = MagicMock(return_value={})
        
        df = pd.DataFrame({'minute': [1, 2, 3]})
        executor.execute_batched(df, batch_size=2)
        
        assert executor.execute.call_count == 3
        
    def test_insufficient_memory_skips(self, temp_dirs, empty_catalog):
        _, output_dir = temp_dirs
        catalog = empty_catalog
        # 1 graph
        meta = GraphMetadata(
            graph_id="g1", 
            title="G1", 
            description="Test G1",
            function=lambda x,y:None, 
            enabled=True, 
            category="test", 
            library=GraphLibrary.MATPLOTLIB,
            data_required=['minute'],
            priority=GraphPriority.HIGH
        )
        catalog.register(meta)
        
        executor = UnifiedGraphExecutor(catalog, output_dir)
        
        # Force no headroom
        executor.memory_monitor.has_headroom = MagicMock(return_value=False)
        
        results = executor.execute_batched(pd.DataFrame(), batch_size=1)
        
        assert results['g1'].status == 'skipped'
        assert 'Insufficient memory' in results['g1'].error
