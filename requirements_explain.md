# Software Dependencies and Scientific Function

Based on the `requirements.txt` file and the architectural logic of the **Simulation Framework for Industrial Hydrogen Production v2.0**, this document details the scientific function of each library.

The dependencies are categorized not just by software type, but by their role in the **Physical-Economic Reconciliation** problem.

## 1. Thermodynamics & Physics Engine (The "Physical Reality")

*   **CoolProp (`CoolProp>=6.4.0`)**: The fundamental source of truth for fluid properties.
    *   *Scientific Role:* Provides the Helmholtz energy equations of state (EOS). To circumvent the high computational cost (0.1–1.0 ms/call), the framework uses CoolProp primarily to generate **Look-Up Table (LUT)** caches, enabling the simulation to trade memory (RAM) for speed (50–200x speedup) while maintaining thermodynamic rigor.

*   **SciPy (`scipy>=1.9.0`)**: Advanced numerical algorithms.
    *   *Scientific Role:* Handles stiff Ordinary Differential Equation (ODE) integration for reactor kinetics (e.g., PFR thermal profiles) and provides the root-finding algorithms (Newton-Raphson) used in complex equilibrium calculations where analytical solutions do not exist.

## 2. High-Performance Computing (The "Speed Necessity")

*   **Numba (`numba>=0.57.0`)**: The engine for **Time-Scale Separation**.
    *   *Scientific Role:* Facilitates **Just-In-Time (JIT)** compilation of the "Fast Manifold" algebraic constraints (flash calculations, iterative loops). By compiling Python to native machine code, it reduces inner-loop execution time from microseconds to nanoseconds, making the 13.1 million timestep annual simulation computationally feasible.

*   **NumPy (`numpy>=1.24.0`)**: The foundation for **Memory Stability**.
    *   *Scientific Role:* Used for static memory allocation. To enforce the **Lifecycle Contract**, all state vectors are pre-allocated as contiguous C-struct arrays at initialization (`t=0`), preventing dynamic memory fragmentation and ensuring cache locality during the "Stepping Phase."

## 3. Environmental & Economic Boundaries (The "Inputs")

*   **windpowerlib (`windpowerlib>=0.2.2`)**: Renewable energy physical modeling.
    *   *Scientific Role:* Transforms raw meteorological data (wind speed, roughness) into power availability time-series, acting as the physical boundary condition for the **Environment Manager**.

*   **entsoe-py (`entsoe-py>=0.5.0`)**: Grid economic modeling.
    *   *Scientific Role:* Fetches Day-Ahead and Intraday electricity prices from the ENTSO-E transparency platform. These signals feed the **Dispatch Logic** (Intent), creating the economic "Push" characterizing the arbitrage strategy.

*   **Pandas (`pandas>=2.0.0`)**: Time-series alignment.
    *   *Scientific Role:* Synchronization of asynchronous input data strings (weather vs. market ticks) into the unified \SI{1}{\minute} discrete timestep grid required by the **Simulation Engine**.

## 4. Topology & Orchestration (The "Structure")

*   **NodeGraphQt (`NodeGraphQt>=0.6.1`)**: Visual topology management.
    *   *Scientific Role:* Enables **Topology Agnosticism**. By visually constructing the **Component Graph** (nodes and edges), it allows researchers to modify plant design (e.g., inserting a buffer tank) without rewriting solver code, facilitating structural sensitivity analysis.

*   **PyYAML (`PyYAML>=6.0`)**: Configuration persistence.
    *   *Scientific Role:* Serializes the "Component Registry" state. It defines the static attributes (capacity, efficiency curves) of every component, separating the *definition* of the machine from its *simulation* logic.

*   **Pydantic (`pydantic>=2.0.0`)**: Data validation.
    *   *Scientific Role:* Enforces **Physical Validity Constraints** at the input gate. Ensures that parameters (e.g., "efficiency must be 0 < eta < 1") are validated before the simulation starts, preventing "Garbage In, Garbage Out" scenarios.

## 5. Visualization & Reporting (The "Outputs")

*   **h5py (`h5py>=3.1.0`)**: High-volume data persistence.
    *   *Scientific Role:* Stores the complete 525,600-step state history for 50+ components. HDF5 format is essential for handling the >1 GB datasets generated per run, supporting the **Closed-Loop Mass Accounting** verification.

*   **PySide6 (`PySide6>=6.4.0`)** & **Matplotlib (`matplotlib>=3.5.0`)**: Analysis interface.
    *   *Scientific Role:* Renders the "Anatomy of a Timestep" into human-readable performance curves, allowing immediate visual verification of the dispatch strategy's adherence to physical constraints.

## 6. Quality Assurance

*   **pytest (`pytest>=7.0.0`)**: Verification framework.
    *   *Scientific Role:* Validates that the **Lifecycle Contract** logic holds true across software updates, ensuring that the "Two Rivers" (Physics vs. Control) never accidentally cross-contaminate.