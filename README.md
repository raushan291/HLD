# HLD — System Design Collection

End-to-end system designs, each including High-Level Design (HLD),
Low-Level Design (LLD), and pseudocode.

| # | System | HLD | LLD | Pseudocode | Folder |
|---|--------|-----|-----|------------|--------|
| 1 | Person Attribute-Based Image Search | [hld1](attribute-based-search/hld1.jpg) · [hld2](attribute-based-search/hld2.jpg) | [lld](attribute-based-search/lld.jpg) · [search-lld](attribute-based-search/att-search-lld.png) | [pseudocode](attribute-based-search/pseudocode.py) | [attribute-based-search/](attribute-based-search) |
| 2 | Real-Time Dice Game Detection | [hld](gaming/hld.jpg) | [lld](gaming/lld.jpg) | [pseudocode](gaming/pseudocode.py) | [gaming/](gaming) |
| 3 | Manufacturing Defect Detection | [hld](manufacturing-defect-detection/hld.jpg) | [lld](manufacturing-defect-detection/lld.jpg) | [pseudocode](manufacturing-defect-detection/pseudocode.py) | [manufacturing-defect-detection/](manufacturing-defect-detection) |

---

## 1. Person Attribute-Based Image Search

Scalable, real-time search for individuals across 200 surveillance cameras using
visual attributes (age group, gender, shirt color, pant color), with 60 days of
searchable history.

- **HLD:** [hld1.jpg](attribute-based-search/hld1.jpg) · [hld2.jpg](attribute-based-search/hld2.jpg)
- **LLD:** [lld.jpg](attribute-based-search/lld.jpg) · [att-search-lld.png](attribute-based-search/att-search-lld.png)
- **Pseudocode:** [pseudocode.py](attribute-based-search/pseudocode.py)
- **Design notes:** [attr-hld-lld.txt](attribute-based-search/attr-hld-lld.txt)
- **Folder:** [attribute-based-search/](attribute-based-search)

## 2. Real-Time Dice Game Detection

Detects game start/end via motion, detects 3 dice in a transparent jar, and
classifies the top face of each die at ~100 ms latency on CPU.

- **HLD:** [hld.jpg](gaming/hld.jpg)
- **LLD:** [lld.jpg](gaming/lld.jpg)
- **Pseudocode:** [pseudocode.py](gaming/pseudocode.py)
- **Design notes:** [game-hld-lld.txt](gaming/game-hld-lld.txt)
- **Folder:** [gaming/](gaming)

## 3. Manufacturing Defect Detection

One camera per conveyor belt: detects products, classifies them PASS / FAIL /
REVIEW, records results with metadata, and can signal a PLC to reject defects.

- **HLD:** [hld.jpg](manufacturing-defect-detection/hld.jpg)
- **LLD:** [lld.jpg](manufacturing-defect-detection/lld.jpg)
- **Pseudocode:** [pseudocode.py](manufacturing-defect-detection/pseudocode.py)
- **Design notes:** [defect-det-hld-lld.txt](manufacturing-defect-detection/defect-det-hld-lld.txt)
- **Folder:** [manufacturing-defect-detection/](manufacturing-defect-detection)
