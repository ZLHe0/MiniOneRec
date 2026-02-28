# Training Data Composition

## SFT Training Data

Three datasets concatenated (~80K total samples):

### 1. SidSFTDataset (~36K samples)
- **Source**: Train CSV (`data/Amazon/train/*.csv`)
- **Task**: Sequential recommendation (SID history → next SID)
- **Input**: `"The user has interacted with items <a_X><b_Y><c_Z>, <a_A><b_B><c_C>, ... in chronological order. Can you predict the next possible item that the user may expect?"`
- **Output**: Target SID (e.g., `<a_1><b_2><c_3>`)

### 2. SidItemFeatDataset (~7.4K samples)
- **Source**: `item.json` + `index.json` (from `data/Amazon/index/`)
- **Task**: Bidirectional title-SID alignment
- **Two sub-tasks** (~3.7K each):
  - **title2sid**: `"Which item has the title: {title}?"` → SID
  - **sid2title**: `"What is the title of item {SID}?"` → title
- **Purpose**: Teaches the model the mapping between natural language item names and semantic IDs

### 3. FusionSeqRecDataset (~36K samples)
- **Source**: Train CSV + `item.json` + `index.json`
- **Task**: Enriched sequential recommendation with item features
- **Two prompt types**:
  - Title output: `"The user has sequentially interacted with items {SID history}. Can you recommend the next item? Tell me the title of the item"` → item title
  - Description output: `"Please review the user's historical interactions: {SID history}, and describe what kind of item he still needs."` → item description
- **Purpose**: Grounds SID understanding in natural language by connecting SIDs to titles/descriptions

### SFT Eval Data
- **SidSFTDataset** only, on the validation CSV (~4.5K samples)

---

## RL Training Data

Three datasets concatenated (~53K total samples):

### 1. SidDataset (~36K samples)
- **Source**: Train CSV
- **Task**: Sequential recommendation (SID history → next SID)
- **Input**: Same format as SidSFTDataset but returns prompt/completion pairs (no tokenization) for GRPO
- **Output**: Target SID

### 2. RLTitle2SidDataset (~7.4K samples)
- **Source**: `item.json` + `index.json`
- **Task**: One-directional item identification → SID
- **Two sub-tasks**:
  - **title2sid**: `"Which item has the title: {title}?"` → SID
  - **description2sid**: `"An item can be described as follows: {description}. Which item is it describing?"` → SID
- **Difference from SFT**: Only predicts SIDs (no reverse SID→title direction), adds description→SID

### 3. RLSeqTitle2SidDataset (10K samples, subsampled)
- **Source**: Train CSV
- **Task**: Title-based sequential recommendation → SID
- **Input**: `"Given the title sequence of user historical interactive items: "Title A", "Title B", ..., can you recommend a suitable next item for the user?"` → SID
- **Purpose**: Cross-modal task — user history expressed in titles, but output is always a SID
- **Note**: Capped at `sample=10000` (vs full ~36K)

### RL Eval Data
- **SidDataset** only, on the validation CSV (~4.5K samples)

---

## Key Differences: SFT vs RL

| Aspect | SFT | RL |
|--------|-----|-----|
| **Total samples** | ~80K | ~53K |
| **Output modality** | SIDs + titles + descriptions | SIDs only |
| **Item feature dataset** | Bidirectional (SID↔title) | One-directional (title/description→SID) |
| **Seq rec with titles** | FusionSeqRecDataset (outputs title/description) | RLSeqTitle2SidDataset (outputs SID, 10K subsample) |

**Design rationale**: SFT teaches the model both the SID vocabulary and natural language understanding of items. RL then optimizes specifically for SID prediction accuracy using GRPO with ranking rewards, so all outputs are SIDs.

---

## Data Files (Industrial_and_Scientific)

| File | Path | Records |
|------|------|---------|
| Train CSV | `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` | 36,259 |
| Valid CSV | `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` | 4,532 |
| Test CSV | `data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv` | — |
| Item index | `data/Amazon/index/Industrial_and_Scientific.index.json` | 3,686 items |
| Item metadata | `data/Amazon/index/Industrial_and_Scientific.item.json` | 3,686 items |
| Item info | `data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt` | 3,686 items |

Each item has a 3-level semantic ID: `<a_X><b_Y><c_Z>`, generated via RQ-VAE/RQ-Kmeans clustering on item embeddings.
