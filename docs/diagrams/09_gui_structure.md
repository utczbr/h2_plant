# GUI Structure Diagrams

Detailed architecture of the graphical user interface components.

## Main Window Architecture

```mermaid
classDiagram
    class PlantEditorWindow {
        -NodeGraph graph
        -QTabWidget tabs
        -QDockWidget properties_dock
        -QDockWidget nodes_dock
        -AllNodesListWidget nodes_list
        -SimulationReportWidget report_widget
        -AdvancedValidator validator
        -GraphPersistenceManager persistence
        +_run_simulation()
        +_on_simulation_complete()
    }
    
    class AllNodesListWidget {
        -NodeGraph graph
        +startDrag()
    }
    
    class SimulationReportWidget {
        -FigureCache cache
        -QTreeWidget tree
        -QScrollArea scroll_area
        +set_simulation_data()
        +_on_graph_ready()
    }
    
    class FigureCache {
        -int max_size
        -dict cache
        +get(graph_id)
        +put(graph_id, figure)
        +set_data(simulation_data)
    }
    
    PlantEditorWindow *-- AllNodesListWidget
    PlantEditorWindow *-- SimulationReportWidget
    SimulationReportWidget *-- FigureCache
```

## Window Layout

```mermaid
flowchart TB
    subgraph MainWindow["PlantEditorWindow"]
        subgraph MenuBar["Menu Bar"]
            File["File"]
            Edit["Edit"]
            View["View"]
            Validation["Validation"]
        end
        
        subgraph Central["Central Widget (Tabs)"]
            NodeEdit["Node Editor Tab<br/>(NodeGraphQt)"]
            Reports["Simulation Reports Tab"]
        end
        
        subgraph LeftDock["Left Dock"]
            NodesPalette["All Nodes List<br/>(Drag-Drop Source)"]
        end
        
        subgraph RightDock["Right Dock"]
            PropsPanel["Properties Panel<br/>(Node Parameters)"]
        end
    end
```

---

## Node Editor Workflow

```mermaid
sequenceDiagram
    participant User
    participant Palette as AllNodesListWidget
    participant Graph as NodeGraph
    participant Adapter as GraphToConfigAdapter
    participant File as GraphPersistenceManager
    
    User->>Palette: Drag node type
    Palette->>Graph: startDrag() with MIME data
    User->>Graph: Drop on canvas
    Graph->>Graph: create_node(node_type)
    
    User->>Graph: Connect ports
    Graph->>Graph: acyclic_check()
    
    User->>Graph: Select node
    Graph->>User: Show properties in dock
    
    User->>File: Save (Ctrl+S)
    File->>Adapter: export_config()
    Adapter->>File: Write .h2plant YAML
```

## Available Node Types

```mermaid
flowchart LR
    subgraph Production["Production"]
        PEM["PEMStackNode"]
        SOEC["SOECStackNode"]
        Rect["RectifierNode"]
    end
    
    subgraph Storage["Storage"]
        Tank["HydrogenTankNode"]
        Battery["BatteryNode"]
    end
    
    subgraph BoP["Balance of Plant"]
        Comp["CompressorNode"]
        Pump["WaterPumpNode"]
        Mixer["MixerNode"]
    end
    
    subgraph Thermal["Thermal"]
        Chiller["ChillerNode"]
        HX["HeatExchangerNode"]
        Steam["SteamGeneratorNode"]
    end
    
    subgraph Sources["Sources"]
        Wind["WindEnergySourceNode"]
        Grid["GridConnectionNode"]
        Water["WaterSourceNode"]
    end
    
    subgraph Economics["Economics"]
        Arb["ArbitrageNode"]
    end
```

---

## Simulation Report Flow

```mermaid
flowchart TD
    Start([Run Simulation]) --> Worker["SimulationWorker<br/>(QThread)"]
    Worker --> Orch["Orchestrator.run_simulation()"]
    Orch --> Results["history Dict"]
    
    Results --> Signal["simulation_complete Signal"]
    Signal --> Report["SimulationReportWidget"]
    
    Report --> SetData["set_simulation_data(history)"]
    SetData --> InvalidateCache["FigureCache.set_data()"]
    InvalidateCache --> BuildTree["Build graph tree"]
    
    BuildTree --> LazyLoad["Create LazyGraphSlots"]
    LazyLoad --> Viewport["Check viewport visibility"]
    
    Viewport --> Visible{Slot visible?}
    Visible -->|Yes| Generate["GraphWorker.run()"]
    Visible -->|No| Wait["Wait for scroll"]
    
    Generate --> Plotter["plotter.py<br/>generate_graph()"]
    Plotter --> Figure["Matplotlib Figure"]
    Figure --> Cache["Cache figure"]
    Cache --> Display["Display in slot"]
```

## Lazy Loading Architecture

```mermaid
flowchart LR
    subgraph ScrollArea["QScrollArea"]
        subgraph Viewport["Visible Viewport"]
            Slot1["LazyGraphSlot 1<br/>(Loaded ✓)"]
            Slot2["LazyGraphSlot 2<br/>(Loading...)"]
        end
        
        subgraph Hidden["Below Viewport"]
            Slot3["LazyGraphSlot 3<br/>(Placeholder)"]
            Slot4["LazyGraphSlot 4<br/>(Placeholder)"]
        end
    end
    
    subgraph ThreadPool["QThreadPool"]
        Worker1["GraphWorker"]
        Worker2["GraphWorker"]
    end
    
    Slot2 --> Worker1
    
    style Slot1 fill:#90EE90
    style Slot2 fill:#FFE4B5
    style Slot3 fill:#E0E0E0
    style Slot4 fill:#E0E0E0
```

## Graph Categories

```mermaid
flowchart TB
    subgraph Tree["Graph Selection Tree"]
        Production["📈 Production"]
        Economics["💰 Economics"]
        Components["🔧 Components"]
        Advanced["📊 Advanced"]
    end
    
    subgraph ProductionGraphs["Production Graphs"]
        p1["H₂ Production Rate"]
        p2["Cumulative H₂"]
        p3["SOEC vs PEM Split"]
    end
    
    subgraph EconomicsGraphs["Economics Graphs"]
        e1["Revenue Analysis"]
        e2["Arbitrage Decisions"]
        e3["Price vs Production"]
    end
    
    subgraph ComponentGraphs["Component Graphs"]
        c1["Tank Levels"]
        c2["Compressor Power"]
        c3["Electrolyzer Efficiency"]
    end
    
    Production --> ProductionGraphs
    Economics --> EconomicsGraphs
    Components --> ComponentGraphs
```

---

## Figure Cache Strategy

```mermaid
sequenceDiagram
    participant Report as SimulationReportWidget
    participant Cache as FigureCache
    participant Worker as GraphWorker
    participant Pool as QThreadPool
    
    Report->>Cache: get(graph_id)
    
    alt Cache Hit
        Cache-->>Report: Return cached figure
    else Cache Miss
        Cache-->>Report: None
        Report->>Pool: start(GraphWorker)
        Worker->>Worker: Generate figure
        Worker-->>Report: graph_ready signal
        Report->>Cache: put(graph_id, figure)
        Report->>Report: Display figure
    end
```

## Key UI Components

| Component | File | Purpose |
|-----------|------|---------|
| `PlantEditorWindow` | `main_window.py` | Main application window |
| `AllNodesListWidget` | `main_window.py` | Node palette with drag-drop |
| `SimulationReportWidget` | `main_window.py` | Report display with lazy loading |
| `GraphWorker` | `main_window.py` | Background graph generation |
| `FigureCache` | `main_window.py` | LRU cache for figures |
| `LazyGraphSlot` | `main_window.py` | Placeholder for lazy graphs |
| `AdvancedValidator` | `advanced_validation.py` | Topology validation |
| `GraphPersistenceManager` | `graph_persistence.py` | Save/load .h2plant files |
