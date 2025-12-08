# Hybrid Dual-Path Plant Topology
 
 This diagram represents the **Hybrid Plant Topology** validated in `test_full_plant.py`.
 It combines high-efficiency baseload SOEC with fast-response PEM electrolysis, both feeding a common compression stage.
 
 ## Components & Flow
 
 ```mermaid
 flowchart LR
     subgraph Inputs
         Grid["Grid Power"]
         Water["Water Source<br/>(Liquid)"]
         Steam["Steam Source<br/>(Gas 450K)"]
     end
 
     subgraph PEM_Chain ["PEM Production Chain (5 MW)"]
         direction TB
         PEM["DetailedPEMElectrolyzer<br/>(5 MW)"]
         P_Chiller["Chiller<br/>(Cool to 25C)"]
         P_KOD["Knock-Out Drum<br/>(Flash Separation)"]
         P_Coal["Coalescer<br/>(Mist Removal)"]
         
         PEM -->|Wet H2 (353K)| P_Chiller
         P_Chiller -->|Cooled H2| P_KOD
         P_KOD -->|Drying| P_Coal
     end
 
     subgraph SOEC_Chain ["SOEC Production Chain (11.52 MW)"]
         direction TB
         SOEC["SOECOperator<br/>(6x2.4MW @ 80%)"]
         S_Chiller["Large Chiller<br/>(Cool 1073K -> 298K)"]
         S_KOD["Knock-Out Drum"]
         
         SOEC -->|Hot H2 (1073K)| S_Chiller
         S_Chiller -->|Cooled H2| S_KOD
     end
 
     subgraph Common
         Compressor["CompressorStorage<br/>(350 bar)"]
         Tank["H2 Storage"]
     end
 
     %% Connections
     Grid --> PEM
     Grid --> SOEC
     Water --> PEM
     Steam --> SOEC
 
     P_Coal -->|Dry H2 (76 kg/h)| Compressor
     S_KOD -->|Dry H2 (307 kg/h)| Compressor
     
     Compressor -->|Combined (383 kg/h)| Tank
 ```
 
 ## Validated Specifications
 
 | Path | Nominal Power | Config | Verified H₂ Output |
 |---|---|---|---|
 | **PEM** | 5.0 MW | Single Stack | ~76 kg/h |
 | **SOEC** | 14.4 MW | 6 Mods × 2.4 MW | ~307 kg/h (@ 80% limit) |
 | **Combined** | 19.4 MW | Hybrid | ~383 kg/h |
 
 _Validated via `h2_plant/tests/test_full_plant.py` on 2025-12-07._
