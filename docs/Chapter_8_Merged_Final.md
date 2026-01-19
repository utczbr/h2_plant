# Chapter 8

## Computational Representation of the System under Design

### 8.1 Role of the Computational Toolbox within the System under Design

In this dissertation, the developed computational toolbox is not merely a peripheral simulation utility; it constitutes a primary engineering deliverable—a "digital bridge" between the high-level system requirements defined in Chapter 3 and the physical architecture established in Chapter 4.

The "System under Design" (SuD) operates in a landscape of high-frequency volatility, where renewable energy curtailment and grid congestion signals vary on a minute-by-minute basis. Traditional steady-state modeling or low-resolution techno-economic tools are insufficient to capture the transient interactions between electrochemical stacks, balance-of-plant (BoP) equipment, and market-driven operational logic. Consequently, the role of this framework is to provide an operationalized environment where the dual-path hydrogen production strategy can be tested over its entire 25-year lifecycle. By doing so, the toolbox transforms abstract grid constraints into concrete performance indicators, serving as the essential decision-making instrument for the technical and economic feasibility of the plant.

### 8.2 Design Principles Derived from System Requirements

The architecture of the simulation framework was not arbitrarily selected; it was systematically derived from the specific system requirements and the "wicked" nature of the congestion problems identified in the preceding chapters. To ensure the framework remains a scientifically rigorous representation of the SuD, four core design principles were established:

**1. Modular Abstraction and Scalability (Link to Cap. 3):** As the SuD involves heterogeneous technologies (PEM, SOEC, and ATR), the framework must support a "plug-and-play" architecture. This modularity allows for the rapid reconfiguration of plant topologies—such as varying stack capacities or hydrogen storage volumes—without requiring a rewrite of the core simulation engine. This directly addresses the requirement for iterative design under uncertainty.

**2. Decoupling of Physics, Control, and Economics (Link to Cap. 2):** A critical gap identified in current literature is the entanglement of economic assumptions with physical limitations. Our framework enforces a strict separation of layers. This ensures that an economic command (e.g., "produce maximum hydrogen due to low prices") is always filtered through the physical reality of the equipment (e.g., ramping limits and thermal constraints), providing an "honest" representation of the system's capabilities.

**3. High-Fidelity Temporal Resolution (Link to Cap. 1):** To address transmission grid congestion, the framework must operate at a resolution that aligns with grid signal variations (1-minute intervals). This requirement imposed a significant computational challenge: simulating 25 years of operation at this frequency involves over 13 million sequential timesteps. This led to the adoption of high-performance computing strategies, such as Just-In-Time (JIT) compilation and pre-allocated memory structures, ensuring the tool remains responsive during the design process.

**4. Performance-Accuracy Symbiosis:** The framework adopts a "dual-path" computational approach. While the physics models (Chapter 6) provide the necessary rigor, the implementation layer utilizes advanced optimization techniques—such as the Lookup Table (LUT) mechanism—to bypass the heavy computational overhead of real-time thermodynamic solvers. This ensures that the fidelity required by thermal engineering is maintained while achieving the execution speeds required for long-term techno-economic analysis.

### 8.3 High-Level Computational Architecture

To translate the system requirements into a functional simulation, the framework is organized into a hierarchical, six-layer architecture. This structure ensures a strict separation of concerns, where each layer abstracts a specific level of complexity—from the high-level economic objectives down to the fundamental thermophysical properties of the working fluids. This modularity is what enables the "symbiotic" nature of the tool, allowing it to function simultaneously as a design instrument and a rigorous physical model.

The architecture is partitioned into the following functional layers:

1. **Orchestration Layer (The Motor):** At the highest level, this layer manages the temporal evolution of the system. It is responsible for the simulation's "heartbeat," advancing the time-stepping logic and ensuring that all downstream components are synchronized across the 25-year lifecycle.

2. **Logic and Strategy Layer (Decision Making):** This layer hosts the operational intelligence of the plant. It interprets external signals—such as grid congestion, renewable availability, and electricity prices—and converts them into setpoints for the physical equipment. It is here where the dual-path strategies are implemented.

3. **Plant Topology Layer (The Graph):** This layer represents the physical interconnection of the facility. Using a graph-based approach, it defines how mass and energy flow between different subsystems (e.g., from the electrolyzer to the compressor or storage). It utilizes topological sorting algorithms to ensure the correct causal order of calculations.

4. **Component Layer (Physical Models):** Each unit of equipment (PEM, SOEC, ATR, compressors, etc.) exists as a discrete object within this layer. These objects encapsulate the specific engineering equations and performance curves described in Chapter 6, responding to control signals while reporting their physical state.

5. **State and Property Layer (Thermodynamics):** This layer provides the rigorous physical foundation. It calculates the thermodynamic states (enthalpy, entropy, density) of the process streams. To bridge the gap between accuracy and speed, this layer is the primary beneficiary of the optimization strategies discussed later in this chapter.

6. **Infrastructure and Data Layer (The Foundation):** The base layer handles the "plumbing" of the software, including configuration loading (YAML), logging, data persistence, and the interface with high-performance computing libraries like Numba and NumPy.

By decoupling the architecture into these discrete layers, the framework ensures that a modification in the physical model of a stack (Layer 4) does not require a redesign of the economic logic (Layer 2). This "encapsulation of complexity" is the key to the toolbox's flexibility and its ability to represent a complex *System under Design* without becoming computationally unmanageable.

### 8.4 Simulation Lifecycle and Execution Phases

The operationalization of the simulation framework follows a structured lifecycle, meticulously designed to manage the interplay between physical rigor and economic complexity while maintaining numerical stability. This execution process is partitioned into three chronological phases: **Setup**, **Motor (Execution)**, and **Results**.

#### *8.4.1 Setup Phase: Configuration and Model Assembly*

The lifecycle begins with the translation of user-defined plant topologies into a functional computational graph. This phase utilizes a configuration-driven approach via YAML files, allowing the "System under Design" to be iteratively refined without modifications to the underlying source code.

During **Component Instantiation**, the framework parses these configuration files to initialize specific objects for electrolyzers (PEM/SOEC), reactors (ATR), and auxiliary equipment, each embedded with its unique design parameters and boundary conditions. A pivotal challenge in modeling complex industrial plants is the inherent sequential dependency between mass and energy streams. To address this, the framework employs **Kahn's Algorithm for Topological Sorting**. By establishing a Directed Acyclic Graph (DAG) execution order, the system prevents mathematical circularity and ensures that the physical state propagates with causal integrity across the entire plant layout.

#### *8.4.2 Motor Phase: The Temporal Execution Loop*

Once the plant architecture is assembled, the Motor Phase—implemented as a dedicated simulation engine class—orchestrates the temporal progression of the simulation. This stage serves as the primary theater for the symbiotic interaction between external grid signals and internal plant physics.

The **Main Execution Loop** iterates through a 25-year horizon at high temporal resolution—typically in one-minute steps. The simulation engine architecture supports variable time-stepping, though the current techno-economic baseline utilizes a fixed 1-minute resolution for numerical stability. Within each increment, the motor synchronizes external inputs, such as electricity prices and renewable profiles, with the internal **Lifecycle Contract**. This contract enforces a strict behavioral protocol for every component: first, initializing the state for the current step; second, executing the internal physics and control equations; and finally, broadcasting the resulting mass, energy, and economic flows. To preserve the causal integrity of time-series data—crucial for managing state-of-charge in batteries and hydrogen storage—the stepping logic remains sequential, even while leveraging high-performance libraries for numerical acceleration.

Each simulation timestep follows a strict **5-phase time-marching scheme**: (1) scheduled events are processed via the event scheduling system, (2) the dispatch strategy evaluates electricity prices and storage states to compute power setpoints, (3) all components execute their physics equations in topologically-sorted order—ensuring upstream sources are evaluated before downstream sinks, (4) mass and energy flows are propagated through the connection graph via the flow network, and (5) results are recorded for post-processing. This explicit ordering—conceptually equivalent to an **Euler forward integration scheme**—ensures that control decisions are based on valid thermodynamic states and that no component reads stale data from its upstream sources. The topological ordering, computed via Kahn's algorithm during initialization, guarantees that information flows causally forward through the plant graph within each timestep.

#### *8.4.3 Results Phase: Data Aggregation and Visualization*

The final phase of the lifecycle governs the management of the high-volume data generated during the execution. Given the millions of data points produced in a full-lifecycle simulation, the Results Phase employs optimized post-processing techniques to translate raw numerical output into actionable engineering insights.

Through **Dynamic State Persistence**, data is aggregated into optimized NumPy structures during runtime to minimize I/O overhead. This organized dataset allows the system to automatically derive critical **Performance Metrics (KPIs)**, ranging from the Levelized Cost of Hydrogen (LCOH) to stack efficiency degradation and grid congestion impacts. To conclude the process, an **Automated Visualization** module generates comprehensive process profiles and cumulative performance graphs, providing a robust visual verification of the system's long-term behavior and its alignment with the initial design requirements.

### 8.5 Data Structures and State Management

The dual nature of the framework—balancing stringent physical rigor with the necessity for high-speed execution—demands a sophisticated approach to data handling and state persistence. As the simulation must track millions of data points across a 25-year horizon, the design of its internal data structures is purposefully optimized to minimize memory overhead and eliminate redundant computational cycles.

#### *8.5.1 Thermodynamic Acceleration via Lookup Table (LUT) Mechanisms*

A primary bottleneck in high-resolution process simulation is the repetitive calculation of thermophysical properties, such as enthalpy, entropy, and density. Relying exclusively on real-time calls to rigorous thermodynamic libraries like CoolProp would increase simulation time by orders of magnitude, effectively rendering long-term lifecycle analysis unfeasible. To resolve this, the framework implements an advanced Lookup Table (LUT) mechanism based on multi-dimensional interpolation. During the setup phase, the framework generates or loads pre-computed grids of thermodynamic states for relevant fluid mixtures, including hydrogen, oxygen, water, and syngas.

By replacing complex iterative flash calculations with localized bilinear interpolations executed via Numba JIT-compiled kernels, the framework achieves a measured **speedup factor of 158.5x** for direct kernel calls and approximately 49x when including Python wrapper overhead. Empirical benchmarking against CoolProp (5,000 H₂ enthalpy lookups) demonstrated that the JIT kernel executes at 787,441 calls/second compared to CoolProp's 4,967 calls/second. This efficiency allows a full annual simulation of 525,600 minutes to be completed in under fifteen minutes on standard workstation hardware. Crucial to this approach is the preservation of accuracy; the LUT grids (2000×2000 pressure-temperature points) are strategically densified using geometric spacing in pressure and linear spacing in temperature. This selective refinement ensures that interpolation errors remain well below the engineering tolerances required for reliable techno-economic assessment.

#### *8.5.2 State Management and the Evolution of Lifecycle Health*

Unlike conventional static models, this framework treats the hydrogen plant as a dynamic, "living" system by managing two distinct categories of data states. The first category comprises **Transient Flow States**, which represent the instantaneous mass and energy flows at each minute-long interval. These states serve as the vital inputs for successive components within the topological graph, facilitating the data propagation described previously.

The second category, **Persistent Component States**, represents the internal condition of the equipment over time. This is where the framework addresses the critical long-term lifecycle requirements of the System under Design (SuD) through integrated degradation modeling. By tracking cumulative operating hours, start-stop cycles, and current density peaks, the system applies empirical degradation curves—specifically efficiency and capacity factor arrays indexed by operational years—to adjust electrolyzer performance at every timestep. This persistent memory allows the simulation to carry the "damage" incurred in the early years of operation through the entire horizon, accurately reflecting its cumulative impact on efficiency and OPEX as the plant approaches its twentieth year of service.

#### *8.5.3 Memory Optimization, Vectorization, and JIT Compilation*

To manage the massive datasets generated by this longitudinal approach, the framework eschews native Python list structures in favor of pre-allocated NumPy arrays and vectorized operations. This strategy significantly reduces the overhead associated with dynamic memory allocation during the execution loop. Furthermore, critical numerical kernels are decorated with **Numba for Just-In-Time (JIT) compilation**, which translates Python logic into optimized machine code. This ensures that the state management and numerical logic execute at speeds comparable to compiled languages like C++ or Fortran.

Finally, the framework optimizes data persistence by avoiding frequent disk I/O operations, which would otherwise throttle the simulation. Instead, the simulation history is managed by a dedicated history manager class, which stores data in memory-mapped buffers and flushes to **Parquet files with Snappy compression** at configurable chunk intervals (default: 10,000 steps). This **chunked architecture** ensures constant memory usage (~100 MB) regardless of simulation length, while the "heartbeat" of the simulation remains rapid and consistent.

### 8.6 Interaction between Control, Physics, and Economics

The core scientific value of this framework lies in its capacity to simulate the interdependent relationship between high-level economic decisions and low-level physical constraints. Unlike traditional models that often assume ideal or instantaneous operation, this toolbox enforces a "symbiotic" interaction. In this environment, every operational intent is systematically validated against the physical reality of the plant, ensuring that long-term economic indicators are grounded in technical feasibility.

#### *8.6.1 The Lifecycle Contract and Agent Synchronization*

To manage the complexity of multiple subsystems—such as electrolyzers, storage units, and autothermal reformers—interacting over vast time horizons, the framework implements a strict **Lifecycle Contract**. This protocol, defined as an abstract base class from which all physical components inherit, serves as a standardized communication interface that every component must adhere to during each simulation step, regardless of its underlying physical principles. The interaction begins with the **Initialization** phase, where components receive setpoints from the Control Layer and boundary conditions from the environment, such as electricity prices or grid congestion signals. This is followed by the **Stepping** phase, where the component executes its specific internal equations—ranging from electrochemical polarization curves to heat transfer—to resolve its physical state for that specific minute. Finally, through **State Reporting**, the component broadcasts its outputs, such as mass flow, purity, and power consumption, to the rest of the system. This contractual approach ensures that "economic intent" is accurately translated into a "physical outcome," providing a reliable data foundation for subsequent techno-economic analysis.

#### *8.6.2 Unidirectional Push Architecture with Demand Signal Propagation*

To maintain numerical stability and prevent the circular dependencies common in complex industrial simulations, the framework employs a **unidirectional push architecture** with **demand signal propagation**. This design achieves the benefits of both push and pull patterns while preserving strict causal ordering.

Through the **Push Mechanism**, mass and energy flows are propagated downstream through the topological graph. Each connection follows a three-step pattern: (1) retrieve the output available at the source port, (2) deliver it to the target port, and (3) notify the source of the accepted quantity for mass balance reconciliation. For instance, hydrogen produced by an SOEC stack is "pushed" to a compressor, which calculates the required power based on the received throughput.

For scenarios requiring **demand-driven behavior**—such as tank level control zones or curtailment signals—the framework propagates **control signals** as a special resource type. When a downstream component (e.g., a storage tank reaching capacity) requires reduced inflow, it outputs a demand signal (e.g., a multiplier of 0.0 for "stop") to its upstream source. This signal is pushed through the connection graph like any other flow; the upstream component receives it and adjusts its production in the **next timestep**. This one-cycle delay is an intentional design choice that preserves causal integrity: a component cannot retroactively modify its own production in response to downstream conditions within the same timestep.

This architecture guarantees that the simulation remains causally consistent—a prerequisite for modeling the "wicked" problem of grid congestion over decades of operation—while enabling the flexible plant topologies established in Chapter 4.

#### *8.6.3 Bridging the Gap: From Physical Performance to Economic Reality*

The interaction cycle is finalized by translating transient physical performance into long-term economic reality, a process that monitors what can be described as the "operational cost of physics." This is achieved through **Honest Economic Reporting**: if the control layer requests a setpoint that exceeds the physical ramping limits of the ATR or the electrolyzer, the physics layer reports only the actual achieved output. The economic model subsequently calculates the resulting "lost opportunity cost" or grid penalties based on this physical limitation, rather than on the idealized command. Furthermore, the integration of **Degradation-Adjusted OPEX** ensures that as components age, the resulting decrease in efficiency—modeled via the cumulative operating hours and efficiency degradation factors described in Section 8.5.2—is directly reflected in rising operational costs in the final years of the project. By establishing these rigorous interaction patterns, the toolbox transcends the role of a simple calculator, becoming a dynamic representation of the **System under Design** that proves how technical robustness directly dictates economic viability.

### 8.7 Scope, Limitations, and Link to Subsequent Chapters

As the computational foundation of this Engineering Doctorate, defining the operational boundaries of the framework is as imperative as the development of the framework itself. Establishing these limits, alongside the rigorous procedures employed to ensure numerical and physical reliability, provides the necessary confidence in the outputs that drive the subsequent engineering analysis. This final section documents the quality assurance infrastructure and provides the transition from the architectural design to the practical application of the tool.

#### *8.7.1 Validation and Quality Assurance (QA) Protocols*

The "dual deliverable" nature of this project—comprising both a software framework and a comprehensive engineering analysis—demands that the codebase be as robust as the physical plant it represents. To meet this standard, the framework incorporates a multi-tier validation infrastructure. At the most granular level, **Unit and Integration Testing** ensures that every component model, from PEM stacks to ATR reactors, is verified against literature benchmarks. These tests confirm that when components are coupled within the plant graph, mass and energy conservation laws are strictly obeyed.

To ensure that the optimization strategies discussed in Section 8.5 do not compromise scientific accuracy, the framework undergoes periodic **Performance Benchmarking** against "ground-truth" simulations using raw CoolProp calls. The observed deviations in thermodynamic properties remain consistently below 0.1%, a margin that justifies the speed-accuracy trade-off for lifecycle-scale studies. Finally, through **Physical Verification**, dynamic profiles are cross-referenced with established process models to ensure that transient behaviors—such as temperature ramps and pressure swings—align with industrial expectations and empirical data.

#### *8.7.2 Framework Scope and Boundary Limitations*

While the toolbox is a versatile instrument for long-term techno-economic assessment, its design is strategically focused on specific engineering goals, which necessitates certain boundary limitations. The framework is optimized for **system-level dynamics** with a temporal resolution of minutes; it is therefore not intended for microsecond-scale power electronics stability analysis or high-fidelity computational fluid dynamics (CFD) of internal stack flows.

Furthermore, to maintain the viability of a 25-year simulation, the framework employs **Algebraic Transient Approximations**. This means that while fast transients are treated algebraically to save computational cost, slower dynamics—such as thermal inertia and battery state-of-charge—are integrated explicitly to maintain physical relevance. Lastly, the model assumes that the balance-of-plant (BoP) equipment operates within pre-defined control envelopes, allowing the computational complexity to remain focused on the primary hydrogen production and conversion pathways.

#### *8.7.3 Transition to Subsequent Analysis*

By establishing this computational framework, the dissertation moves from the "Why" (Chapters 1–2) and the "What" (Chapters 3–4) to the "How." This chapter has operationalized the physical models of Chapter 6 and the operational strategies of Chapter 5 into a single, cohesive, and high-performance environment.

This toolbox now serves as the primary engine for the remainder of this research. In the following chapters, this framework will be utilized to execute the **Techno-Economic Analysis (Chapter 7)** and the final **Case Studies**, where the symbiosis between physics, control, and economics will be tested against real-world grid congestion scenarios in the Netherlands. The speed and reliability of this framework allow for the exploration of a vast design space, ensuring that the final recommendations for the System under Design are both technically robust and economically viable in a transitioning energy market.
