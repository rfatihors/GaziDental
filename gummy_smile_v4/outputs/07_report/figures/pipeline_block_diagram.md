# System block diagram (Reviewer 1)

```mermaid
flowchart LR
    A[Smile photograph] --> B[YOLOv11x-seg<br/>imgsz 640, retina_masks]
    B --> C[Class-separated masks<br/>gingiva / lip, union of instances,<br/>original resolution]
    C --> D[Column-wise thickness profile t(x)<br/>longest vertical run, empty columns = 0]
    D --> E[Tooth regioning<br/>midline-anchored zeniths (C), fallback A]
    E --> F[Region values p25 -> image mean<br/>px / px_per_mm = mm]
    F --> G[Rule engine<br/>E1 <4, E2 3-6, E3 4-8, E4 >8 mm<br/>overlaps -> combined label]
    G --> H[Report: class, candidates,<br/>treatment alternatives, QC flags]
    C -. lip mask: window, midline,<br/>boundary check .-> D
```
