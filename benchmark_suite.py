
import sys
import os
import argparse
import time
import logging
import psutil
import cProfile
import pstats
import csv
from pathlib import Path
from io import StringIO
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from h2_plant.config.loader import ConfigLoader
from h2_plant.core.graph_builder import PlantGraphBuilder
from h2_plant.core.component_registry import ComponentRegistry
from h2_plant.simulation.engine import SimulationEngine
from h2_plant.control.engine_dispatch import HybridArbitrageEngineStrategy
from h2_plant.data.price_loader import EnergyPriceLoader
from h2_plant.config.plant_config import ConnectionConfig

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkSuite")

def get_memory_usage_mb():
    """Return current process Resident Set Size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(scenarios_dir: str, hours: int, output_dir: Path, enable_profiling: bool = True):
    """
    Run detailed benchmark of the simulation.
    
    Phases:
    1. Initialization (Timed)
    2. Simulation Loop (Profiled & Timed)
    """
    start_total = time.time()
    memory_log = []
    
    print("=" * 80)
    print(f"BENCHMARK SUITE - Starting {hours} hour simulation")
    print("=" * 80)
    print(f"PID: {os.getpid()}")
    print(f"Initial Memory: {get_memory_usage_mb():.2f} MB")
    
    # =========================================================================
    # PHASE 1: INITIALIZATION
    # =========================================================================
    print("\n--- PHASE 1: INITIALIZATION ---")
    t_init_start = time.perf_counter()
    
    # 1. Load configuration
    logger.info(f"Loading configuration from {scenarios_dir}")
    loader = ConfigLoader(scenarios_dir)
    context = loader.load_context()

    # 2. Build component graph
    builder = PlantGraphBuilder(context)
    components = builder.build()

    # 3. Registry
    registry = ComponentRegistry()
    for cid, comp in components.items():
        registry.register(cid, comp)

    # 4. Topology Connections
    topology_connections = []
    for node in context.topology.nodes:
        source_id = node.id
        for conn in node.connections:
            topology_connections.append(ConnectionConfig(
                source_id=source_id,
                source_port=conn.source_port,
                target_id=conn.target_name,
                target_port=conn.target_port,
                resource_type=conn.resource_type
            ))
            
    # 5. Load Data
    data_loader = EnergyPriceLoader(scenarios_dir)
    prices, wind = data_loader.load_data(
        context.simulation.energy_price_file,
        context.simulation.wind_data_file,
        hours,
        context.simulation.timestep_hours
    )
    total_steps = len(prices)

    # 6. Setup Strategy
    dispatch_strategy = HybridArbitrageEngineStrategy()
    dispatch_strategy._strategy_override = 'REFERENCE_HYBRID'

    # 7. Engine Setup
    engine = SimulationEngine(
        registry=registry,
        config=context.simulation,
        output_dir=output_dir,
        dispatch_strategy=dispatch_strategy,
        topology=topology_connections
    )

    # 8. Engine Initialization
    engine.initialize()
    engine.set_dispatch_data(prices, wind)
    engine.initialize_dispatch_strategy(context, total_steps, use_chunked_history=True)

    t_init_end = time.perf_counter()
    init_duration = t_init_end - t_init_start
    init_mem = get_memory_usage_mb()

    print(f"Initialization Complete.")
    print(f"Duration: {init_duration:.4f} sec")
    print(f"Memory after Init: {init_mem:.2f} MB")

    # JIT Pre-warming — compile all Numba kernels before the timed loop
    print("\n--- JIT WARMUP ---")
    t_warmup_start = time.perf_counter()
    from h2_plant.optimization.numba_ops import warmup_jit_kernels
    lut_mgr = getattr(engine, 'lut_manager', None)
    if lut_mgr is None:
        for _, comp in engine.registry.list_components():
            if hasattr(comp, 'lut_manager'):
                lut_mgr = comp.lut_manager
                break
    warmup_jit_kernels(lut_mgr)
    t_warmup_end = time.perf_counter()
    warmup_duration = t_warmup_end - t_warmup_start
    print(f"JIT Warmup Complete: {warmup_duration:.4f} sec")
    
    memory_log.append({'step': -1, 'phase': 'init_end', 'memory_mb': init_mem, 'duration_sec': init_duration})

    # =========================================================================
    # PHASE 2: SIMULATION LOOP
    # =========================================================================
    print("\n--- PHASE 2: SIMULATION LOOP (Profiling Enabled) ---")
    
    resume_from_hour = 0
    start_hour = resume_from_hour
    end_hour = hours
    steps_per_hour = int(1 / engine.config.timestep_hours)
    start_step = int(start_hour * steps_per_hour)
    end_step = int(end_hour * steps_per_hour)

    effective_start_hour = start_hour
    effective_end_hour = end_hour
    chk_interval = engine.config.checkpoint_interval_hours
    chk_steps = chk_interval * steps_per_hour if chk_interval > 0 else 0

    engine.simulation_start_time = time.time()
    engine.is_running = True
    engine.steps_run = 0
    
    profiler = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()  # START PROFILING
    
    loop_start_time = time.perf_counter()
    
    # Manual loop execution to allow step-by-step tracking
    step_timings = []
    
    # Pre-resolve local references to match engine.run() fast path
    execution_list = engine._execution_list
    flow_execute = engine.flow_network.execute_flows
    signal_execute = engine.flow_network.execute_signals
    event_process = engine.event_scheduler.process_events
    dispatch_decide = engine._dispatch_decide_method
    dispatch_record = engine._dispatch_record_method
    dispatch_prices = engine._dispatch_prices
    dispatch_wind = engine._dispatch_wind
    pre_step_callback = engine.pre_step_callback
    post_step_callback = engine.post_step_callback
    flow_tracker = engine.monitoring.flow_tracker
    monitoring = engine.monitoring
    registry = engine.registry

    try:
        current_step = start_step
        while current_step < end_step:
            step_start = time.perf_counter()

            # --- SINGLE TIMESTEP EXECUTION (mirrors engine.run() fast path) ---
            hour = current_step / steps_per_hour
            engine.current_hour = hour
            flow_tracker.set_current_hour(hour)

            # Pre-step callback
            if pre_step_callback:
                pre_step_callback(hour)

            # Event processing (present in production, was missing here)
            event_process(hour, registry)

            # Dispatch Decision
            if dispatch_decide and dispatch_prices is not None:
                dispatch_decide(hour, dispatch_prices, dispatch_wind)

            # Signal Pre-Pass
            signal_execute(hour)

            # Physics Step — pre-resolved execution list, no try/except
            for component in execution_list:
                component.step(hour)

            # Flow Propagation
            flow_execute(hour)

            # Dispatch Recording
            if dispatch_record:
                dispatch_record()

            if post_step_callback:
                post_step_callback(hour)

            engine.steps_run += 1

            # Periodic tasks — hourly monitoring, daily FlowTracker flush
            if current_step % steps_per_hour == 0:
                int_hour = int(hour)
                if chk_steps > 0 and current_step > start_step and current_step % chk_steps == 0:
                    engine._save_checkpoint(int_hour)

                if int_hour > 0 and int_hour % 168 == 0:
                    engine._log_progress(int_hour, effective_end_hour, effective_start_hour)

                monitoring.collect(hour, registry)

                if current_step % (24 * steps_per_hour) == 0:
                    flow_tracker.flush(
                        output_dir / "flows.jsonl",
                        rolling=True,
                        reset_aggregates=True
                    )

            # --- END SINGLE TIMESTEP ---

            step_duration = time.perf_counter() - step_start
            step_timings.append(step_duration)

            current_step += 1

            # Log memory periodically (every hour)
            if current_step % steps_per_hour == 0:
                mem_mb = get_memory_usage_mb()
                memory_log.append({'step': current_step, 'phase': 'loop', 'memory_mb': mem_mb, 'duration_sec': step_duration})

    except Exception as e:
        logger.error(f"Simulation crashed at step {current_step}: {e}", exc_info=True)
        raise

    if profiler:
        profiler.disable()  # STOP PROFILING
    engine.is_running = False
    flow_tracker.flush(
        output_dir / "flows.jsonl",
        rolling=True,
        reset_aggregates=True
    )
    loop_end_time = time.perf_counter()
    loop_duration = loop_end_time - loop_start_time
    
    final_mem = get_memory_usage_mb()
    print(f"Simulation Loop Complete.")
    print(f"Duration: {loop_duration:.4f} sec")
    print(f"Final Memory: {final_mem:.2f} MB")
    print(f"Average Step Time: {np.mean(step_timings)*1000:.2f} ms")
    
    memory_log.append({'step': current_step, 'phase': 'end', 'memory_mb': final_mem, 'duration_sec': 0})

    # =========================================================================
    # REPORTS
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Profile Stats
    stats_path = output_dir / "profile.stats"
    if profiler:
        profiler.dump_stats(stats_path)
    
    txt_report_path = output_dir / "benchmark_report.txt"
    with open(txt_report_path, 'w') as f:
        f.write(f"BENCHMARK REPORT\n")
        f.write(f"================\n")
        f.write(f"Hours: {hours}\n")
        f.write(f"Total Initialization Time: {init_duration:.4f} s\n")
        f.write(f"Total Simulation Loop Time: {loop_duration:.4f} s\n")
        f.write(f"Total Run Time: {loop_duration + init_duration:.4f} s\n\n")
        
        f.write(f"Memory Start: {memory_log[0]['memory_mb']:.2f} MB\n") # Init end is index 0
        f.write(f"Memory End: {final_mem:.2f} MB\n")
        f.write(f"Memory Growth: {final_mem - memory_log[0]['memory_mb']:.2f} MB\n\n")
        
        f.write(f"Average Step Time: {np.mean(step_timings)*1000:.3f} ms\n")
        f.write(f"Max Step Time: {np.max(step_timings)*1000:.3f} ms\n\n")
        
        if profiler:
            f.write("TOP 50 BOTTLENECKS (Cumulative Time in Loop):\n")
            f.write("-" * 60 + "\n")

            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
            ps.print_stats(50)
            f.write(s.getvalue())
        else:
            f.write("(Profiling disabled - run without --no-profile for hotspot analysis)\n")
    
    print(f"\nReport saved to: {txt_report_path}")
    
    # 2. Save Memory Log
    csv_path = output_dir / "memory_usage.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'phase', 'memory_mb', 'duration_sec'])
        writer.writeheader()
        writer.writerows(memory_log)
    print(f"Memory log saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed Simulation Benchmark")
    parser.add_argument("--hours", type=int, default=168, help="Simulation duration (default: 168 hours = 1 week)")
    parser.add_argument("--scenarios", type=str, default="scenarios", help="Scenarios directory")
    parser.add_argument("--no-profile", action="store_true", help="Disable cProfile for accurate absolute timing (5-15%% overhead removed)")
    args = parser.parse_args()

    output_dir = Path("benchmark_output")

    run_benchmark(args.scenarios, args.hours, output_dir, enable_profiling=not args.no_profile)
