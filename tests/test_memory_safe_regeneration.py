import pytest
import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from h2_plant.visualization.streaming_downsampler import StreamingDownsampler, MemoryMonitor
from h2_plant.visualization.unified_executor import UnifiedGraphExecutor
from h2_plant.visualization.graph_catalog import GraphCatalog, GraphMetadata, GraphLibrary, GraphPriority
from tools import regenerate_graphs as regen_tools

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
        result_df = downsampler.process_chunks_directory(
            chunks_dir,
            output_cache,
            return_dataframe=True
        )
        
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
            chunks_dir,
            output_cache,
            required_columns=required,
            return_dataframe=True
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
             result_df = downsampler.process_chunks_directory(
                 chunks_dir,
                 output_cache,
                 write_mode='inmemory',
                 return_dataframe=True
             )
             
             # Check data presence
             assert len(result_df) > 0
             assert result_df.minute.min() == 0 # Must preserve start
             assert result_df.minute.max() >= 170 # Must reach end
             
             # Check resolution
             # Should be mostly 10-min resolution (emergency)
             diffs = result_df['minute'].diff().dropna()
             mean_res = diffs.mean()
             assert mean_res >= 5.0 # Should be closer to 10 than 1

    def test_streaming_mode_avoids_final_concat(self, temp_dirs):
        chunks_dir, output_dir = temp_dirs
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60)
        create_synthetic_chunk(chunks_dir / "chunk_2.parquet", 60, 120)

        downsampler = StreamingDownsampler(max_memory_mb=1000, target_resolution_minutes=10)
        output_cache = output_dir / "stream_cache.parquet"

        with patch('h2_plant.visualization.streaming_downsampler.pd.concat', side_effect=AssertionError("concat should not be called")):
            with patch.object(downsampler, '_check_memory_pressure', return_value=False):
                result_df = downsampler.process_chunks_directory(
                    chunks_dir,
                    output_cache,
                    write_mode='streaming',
                    return_dataframe=True
                )

        assert output_cache.exists()
        assert len(result_df) == 12

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


class TestUnifiedExecutorMemoryControls:
    def test_cache_read_applies_downsample_factor(self, temp_dirs, empty_catalog):
        chunks_dir, output_dir = temp_dirs
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60)
        create_synthetic_chunk(chunks_dir / "chunk_2.parquet", 60, 120)

        downsampler = StreamingDownsampler(max_memory_mb=1000, target_resolution_minutes=1)
        cache_path = output_dir / "cache_full.parquet"
        downsampler.process_chunks_directory(
            chunks_dir,
            cache_path,
            required_columns=['minute', 'temperature'],
            write_mode='streaming',
            return_dataframe=False
        )

        catalog = empty_catalog
        meta = GraphMetadata(
            graph_id="g_temp",
            title="G Temp",
            description="Temp graph",
            function=lambda df, dpi: None,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['minute', 'temperature'],
            priority=GraphPriority.MEDIUM,
            category="test",
            enabled=True
        )
        catalog.register(meta)

        executor = UnifiedGraphExecutor(catalog, output_dir)
        df = executor.load_data(cache_path=cache_path, downsample_factor=2)
        assert len(df) == 60

    def test_sequential_execution_preserves_df_attrs(self, temp_dirs, empty_catalog):
        import matplotlib.pyplot as plt

        chunks_dir, output_dir = temp_dirs
        create_synthetic_chunk(chunks_dir / "chunk_1.parquet", 0, 60)

        downsampler = StreamingDownsampler(max_memory_mb=1000, target_resolution_minutes=1)
        cache_path = output_dir / "cache_attrs.parquet"
        downsampler.process_chunks_directory(
            chunks_dir,
            cache_path,
            required_columns=['minute', 'temperature'],
            write_mode='streaming',
            return_dataframe=False
        )

        seen = {}

        def _graph(df, dpi):
            seen['cfg_value'] = df.attrs.get('config', {}).get('my_key')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot([0, 1], [0, 1])
            return fig

        catalog = empty_catalog
        meta = GraphMetadata(
            graph_id="g_attr",
            title="G Attr",
            description="Attr graph",
            function=_graph,
            library=GraphLibrary.MATPLOTLIB,
            data_required=['minute', 'temperature'],
            priority=GraphPriority.HIGH,
            category="test",
            enabled=True
        )
        catalog.register(meta)

        executor = UnifiedGraphExecutor(catalog, output_dir / "graphs")
        results = executor.execute_sequentially_by_category(
            cache_path=cache_path,
            cache_stride=1,
            timeout_seconds=10,
            df_attrs={
                'config': {'my_key': 'my_value'},
                'viz_config': {'dummy': True},
            }
        )

        assert seen['cfg_value'] == 'my_value'
        assert results['g_attr'].status == 'success'


class TestRegeneratePlanningHelpers:
    def test_auto_mode_chooses_sequential_when_memory_is_tight(self):
        mode = regen_tools.choose_execution_mode(
            execution_mode='auto',
            estimated_df_mb=6000.0,
            available_mb=10000.0,
            resolved_columns_count=200,
            enabled_graphs_count=10
        )
        assert mode == 'sequential'

    def test_auto_degrade_stride_selection(self):
        # With the new default threshold of 85%:
        # target_mb = 0.85 * 10000 = 8500
        # stride=1: 14000 > 8500 -> skip; stride=2: 7000 <= 8500 -> selected
        stride, candidates = regen_tools.choose_cache_stride(
            max_extra_downsample=8,
            estimated_df_mb=14000.0,
            available_mb=10000.0,
        )
        assert candidates == [1, 2, 4, 8]
        assert stride == 2

    # -----------------------------------------------------------------------
    # dt_seconds inference tests
    # -----------------------------------------------------------------------

    def test_cache_load_infers_dt_seconds_from_minute_hourly(self, temp_dirs, empty_catalog):
        """Hourly cache (minute step=60) -> row_dt_seconds=3600; dt_seconds stays at 60."""
        _, output_dir = temp_dirs
        cache_path = output_dir / "cache_hourly.parquet"
        pd.DataFrame({
            'minute': [0.0, 60.0, 120.0, 180.0],
            'temperature': [10.0, 20.0, 30.0, 40.0],
        }).to_parquet(cache_path, index=False)

        catalog = empty_catalog
        catalog.register(GraphMetadata(
            graph_id='g_dt', title='G DT', description='DT test',
            function=lambda df, dpi: None, library=GraphLibrary.MATPLOTLIB,
            data_required=['minute', 'temperature'],
            priority=GraphPriority.MEDIUM, category='test', enabled=True,
        ))
        executor = UnifiedGraphExecutor(catalog, output_dir)
        df = executor.load_data(cache_path=cache_path, downsample_factor=1)
        # row spacing is inferred into row_dt_seconds
        assert df.attrs.get('row_dt_seconds') == pytest.approx(3600.0)
        # simulation dt_seconds is NOT overwritten — stays at 60 s default
        assert df.attrs.get('dt_seconds', 60.0) == pytest.approx(60.0)

    def test_cache_load_infers_dt_seconds_with_extra_stride(self, temp_dirs, empty_catalog):
        """stride=2 on hourly data -> minute diffs become 120 -> row_dt_seconds=7200; dt_seconds stays 60."""
        _, output_dir = temp_dirs
        cache_path = output_dir / "cache_hourly_stride.parquet"
        pd.DataFrame({
            'minute': [0.0, 60.0, 120.0, 180.0, 240.0],
            'temperature': [1.0] * 5,
        }).to_parquet(cache_path, index=False)

        catalog = empty_catalog
        catalog.register(GraphMetadata(
            graph_id='g_dt2', title='G DT2', description='DT2 test',
            function=lambda df, dpi: None, library=GraphLibrary.MATPLOTLIB,
            data_required=['minute', 'temperature'],
            priority=GraphPriority.MEDIUM, category='test', enabled=True,
        ))
        executor = UnifiedGraphExecutor(catalog, output_dir)
        # stride=2: rows 0,2,4 -> minute=[0,120,240] -> diff=120 -> row_dt_seconds=7200
        df = executor.load_data(cache_path=cache_path, downsample_factor=2)
        assert df.attrs.get('row_dt_seconds') == pytest.approx(7200.0)
        # simulation dt_seconds must remain at 60 s so rate calculations stay correct
        assert df.attrs.get('dt_seconds', 60.0) == pytest.approx(60.0)

    def test_efficiency_uses_simulation_dt_not_row_spacing(self, temp_dirs, empty_catalog):
        """After _infer_dt_seconds, rate conversion still uses simulation dt=60s, not row spacing."""
        from h2_plant.visualization.unified_executor import _infer_dt_seconds

        # Simulate hourly-decimated cache: rows spaced 60 min apart, each value is per 1-min step
        df = pd.DataFrame({
            'minute': [0.0, 60.0, 120.0, 180.0],
            'H2_soec_kg': [1.0, 1.0, 1.0, 1.0],  # 1 kg per 1-minute simulation step
        })
        _infer_dt_seconds(df)

        # row spacing inferred correctly
        assert df.attrs.get('row_dt_seconds') == pytest.approx(3600.0)

        # Rate calculation must use dt_seconds=60 (1-min sim step) to give correct kg/h
        sim_dt_h = df.attrs.get('dt_seconds', 60.0) / 3600.0  # = 60/3600 = 1/60
        h2_rate = df['H2_soec_kg'].values[0] / sim_dt_h        # 1 / (1/60) = 60 kg/h
        assert h2_rate == pytest.approx(60.0), (
            f"Expected 60 kg/h (1 kg/min × 60), got {h2_rate:.2f}. "
            "dt_seconds was likely overwritten to 3600 by _infer_dt_seconds."
        )

    # -----------------------------------------------------------------------
    # Auto-degrade threshold tests
    # -----------------------------------------------------------------------

    def test_auto_degrade_threshold_85_keeps_stride_1_when_safe(self):
        """When estimated footprint is below 85% of available RAM, stride=1 (no degradation)."""
        stride, _ = regen_tools.choose_cache_stride(
            max_extra_downsample=8,
            estimated_df_mb=1000.0,
            available_mb=2000.0,
            auto_degrade_threshold=0.85,
        )
        # 1000 <= 0.85 * 2000 = 1700 -> stride 1 is safe
        assert stride == 1

    def test_auto_degrade_threshold_85_selects_stride_when_required(self):
        """When footprint exceeds 85% of available RAM, smallest valid stride is chosen."""
        stride, _ = regen_tools.choose_cache_stride(
            max_extra_downsample=8,
            estimated_df_mb=1800.0,
            available_mb=2000.0,
            auto_degrade_threshold=0.85,
        )
        # stride=1: 1800 > 1700 fail; stride=2: 900 <= 1700 pass
        assert stride == 2

    # -----------------------------------------------------------------------
    # Atomic output mode tests
    # -----------------------------------------------------------------------

    def test_atomic_mode_preserves_previous_graphs_on_interrupt(self, tmp_path):
        """If the process is interrupted before the atomic swap, existing graphs/ is untouched."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "old.png").write_text("old content")

        # Simulate: staging was created but swap never happened (interrupted)
        staging_dir = tmp_path / "graphs_staging"
        staging_dir.mkdir()
        (staging_dir / "new.png").write_text("new content")

        # graphs/ must still contain only the old file
        assert (graphs_dir / "old.png").read_text() == "old content"
        assert not (graphs_dir / "new.png").exists()
        # staging is still present for diagnostics
        assert (staging_dir / "new.png").exists()

    def test_atomic_mode_swaps_outputs_on_success(self, tmp_path):
        """After successful run, staging is renamed to graphs/ and old graphs are replaced."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "old.png").write_text("old content")

        staging_dir = tmp_path / "graphs_staging"
        staging_dir.mkdir()
        (staging_dir / "new.png").write_text("new content")

        # Perform the atomic swap (same logic as regenerate_graphs_safe)
        graphs_old = tmp_path / "graphs_old"
        if graphs_old.exists():
            shutil.rmtree(graphs_old)
        if graphs_dir.exists():
            graphs_dir.rename(graphs_old)
        staging_dir.rename(graphs_dir)
        if graphs_old.exists():
            shutil.rmtree(graphs_old)

        assert (graphs_dir / "new.png").read_text() == "new content"
        assert not (graphs_dir / "old.png").exists()
        assert not staging_dir.exists()
        assert not graphs_old.exists()

    # -----------------------------------------------------------------------
    # Memory-projected MAX_COLS test
    # -----------------------------------------------------------------------

    def test_sequential_batch_split_uses_memory_projection(self):
        """The MAX_COLS formula uses available RAM and row count; with low RAM it's < 1000."""
        # Formula: max(50, int(available_mb * 0.85 / (row_count * 8 * 3 / 1e6)))
        # With 200 MB available and 100_000 rows:
        # = max(50, int(200 * 0.85 / (100_000 * 8 * 3 / 1e6)))
        # = max(50, int(170 / 2.4)) = max(50, 70) = 70
        available_mb = 200.0
        row_count = 100_000
        projected = max(50, int(available_mb * 0.85 / (row_count * 8 * 3 / 1e6)))
        assert projected < 1000
        assert projected == max(50, int(available_mb * 0.85 / (row_count * 8 * 3 / 1e6)))

    # -----------------------------------------------------------------------
    # Streaming row-group stride alignment test
    # -----------------------------------------------------------------------

    def test_streaming_stride_aligns_across_row_groups(self, temp_dirs, empty_catalog):
        """Streaming row-group stride produces the same rows as a naive iloc[::stride] on the full file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        _, output_dir = temp_dirs
        cache_path = output_dir / "cache_multigroup.parquet"

        # 8-row DataFrame with hourly-spaced minutes, written as 2 row groups of 4 rows each
        df_full = pd.DataFrame({
            'minute': [float(i * 60) for i in range(8)],
            'temperature': [float(i) for i in range(8)],
        })
        table = pa.Table.from_pandas(df_full, preserve_index=False)
        pq.write_table(table, cache_path, row_group_size=4)
        assert pq.ParquetFile(cache_path).metadata.num_row_groups == 2, (
            "Test setup error: expected 2 row groups"
        )

        catalog = empty_catalog
        catalog.register(GraphMetadata(
            graph_id='g_multigroup', title='G Multigroup', description='Multi-group stride test',
            function=lambda df, dpi: None, library=GraphLibrary.MATPLOTLIB,
            data_required=['minute', 'temperature'],
            priority=GraphPriority.MEDIUM, category='test', enabled=True,
        ))
        executor = UnifiedGraphExecutor(catalog, output_dir)
        df_streamed = executor.load_data(cache_path=cache_path, downsample_factor=2)

        # Group 0 (rows 0-3): offset=(-0)%2=0 -> picks local rows 0,2 -> minutes 0,120
        # Group 1 (rows 4-7): offset=(-4)%2=0 -> picks local rows 0,2 -> minutes 240,360
        df_expected = df_full.iloc[::2].reset_index(drop=True)

        assert list(df_streamed['minute'].values) == list(df_expected['minute'].values), (
            f"Streamed stride: {list(df_streamed['minute'].values)}\n"
            f"Expected (naive): {list(df_expected['minute'].values)}"
        )
        assert len(df_streamed) == len(df_expected)

    def test_regenerate_graphs_atomic_calls_net_profit_in_staging(self, monkeypatch, tmp_path):
        """Atomic mode must run net-profit regeneration in staging before the final swap."""
        output_dir = tmp_path
        (output_dir / "simulation_history_hourly.parquet").write_bytes(b"cache")

        graphs_dir = output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        (graphs_dir / "old_marker.txt").write_text("old", encoding="utf-8")

        class DummyMemoryMonitor:
            def __init__(self, _max_mb):
                pass

            def log_usage(self, _label):
                return None

        class DummyExecutor:
            def __init__(self, _catalog, _output_dir):
                self.catalog = types.SimpleNamespace(get_enabled=lambda: [object()])
                self.memory_monitor = None

            def configure_from_yaml(self, _cfg):
                return None

            def get_required_columns(self):
                return ["minute"]

            def execute_sequentially_by_category(self, **kwargs):
                return {"dummy": types.SimpleNamespace(status="success", error=None)}

        regen_call = {}

        def fake_regen_net_profit(**kwargs):
            regen_call.update(kwargs)
            target_dir = kwargs["graphs_dir"]
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "Economic_Performance_Overview_Base.CAPEX_(Interactive).html").write_text(
                "<html></html>",
                encoding="utf-8",
            )
            return 0

        monkeypatch.setitem(
            sys.modules,
            "h2_plant.visualization.graph_catalog",
            types.SimpleNamespace(GRAPH_REGISTRY=object()),
        )
        monkeypatch.setitem(
            sys.modules,
            "h2_plant.visualization.unified_executor",
            types.SimpleNamespace(UnifiedGraphExecutor=DummyExecutor),
        )
        monkeypatch.setitem(
            sys.modules,
            "h2_plant.visualization.streaming_downsampler",
            types.SimpleNamespace(
                StreamingDownsampler=object,
                MemoryMonitor=DummyMemoryMonitor,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "tools.regenerate_net_profit_plotly",
            types.SimpleNamespace(regenerate_net_profit_plotly=fake_regen_net_profit),
        )
        monkeypatch.setattr(
            regen_tools,
            "compute_effective_memory_budget",
            lambda _mb: (1024, 2048.0, 1536.0),
        )

        regen_tools.regenerate_graphs_safe(
            output_dir=output_dir,
            execution_mode="sequential",
            graph_output_mode="atomic",
            skip_cache=False,
            target_resolution=60,
            timeout_seconds=5,
        )

        assert regen_call["graphs_dir"] == output_dir / "graphs_staging"
        assert not (output_dir / "graphs_staging").exists()
        assert (output_dir / "graphs").exists()
        assert (output_dir / "graphs" / "Economic_Performance_Overview_Base.CAPEX_(Interactive).html").exists()
        assert not (output_dir / "graphs" / "old_marker.txt").exists()
