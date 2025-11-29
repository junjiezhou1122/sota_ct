# Multi-Scale Hierarchical Vision-Language Architecture for CT Report Generation

**Version:** 1.0  
**Date:** 2025-11-29  
**Status:** Design Phase

---

## Overview

This architecture generates radiology findings from 3D chest CT scans using multi-scale visual tokenization, hierarchical feature injection into LLM layers, and structured region/object representations with graph-based reasoning.

**Pipeline:**
```
CT Scan → Multi-scale Tubelets → 3D ViT (multi-layer) 
  → Region/Slot/Graph Tokens → Hierarchical LLM Cross-Attention 
  → Findings → GPT-5.1 → Impression
```

---

## Phase 0: Multi-Scale Tubelet Tokenizer

### Motivation
CT semantics are scale-dependent:
- **Fine scale** → local density, edges, nodules, GGO
- **Mid scale** → lobe/segment-level patterns, consolidation
- **Coarse scale** → global lung structure, mediastinum

### Method

**Tubelet Dimensions:**
```
fine:   (t=2, h=16, w=16)  → N1 tokens
mid:    (t=4, h=32, w=32)  → N2 tokens  
coarse: (t=8, h=64, w=64)  → N3 tokens
```

**Output:**
```python
tokens_fine:   [N1, C]  # High-resolution local details
tokens_mid:    [N2, C]  # Mid-level anatomical structures
tokens_coarse: [N3, C]  # Global context
```

### Design Considerations

✅ **Current Design:**
- Non-overlapping tubelets
- Fixed scales across all CT scans

⚠️ **Improvements to Consider:**
1. **Overlapping tubelets** (stride < size) to avoid boundary artifacts
2. **Adaptive tubelet depth** based on slice thickness (1mm vs 5mm scans)
3. **Scale-specific positional encoding** (RoPE-3D or learned) to distinguish scales
4. **Multi-scale consistency loss** to ensure semantic alignment across scales

---

## Phase 1: Multi-Layer 3D ViT Integration

### Motivation
Different ViT layers encode different semantic levels:
- **Shallow layers (1-4)** → textures, local edges
- **Middle layers (5-8)** → local structures, patterns
- **Deep layers (9-12)** → global anatomy, spatial relationships

### Method

**Scale-to-Layer Injection:**
```
ViT layers 1-4   ← tokens_fine    (local details)
ViT layers 5-8   ← tokens_mid     (regional patterns)
ViT layers 9-12  ← tokens_coarse  (global structure)
```

**Implementation Options:**
1. Concatenation + projection
2. Cross-attention injection
3. Residual injection (recommended)

**Output:**
```python
vit_fine_tokens:   [N1', C']  # Shallow layer features
vit_mid_tokens:    [N2', C']  # Middle layer features  
vit_coarse_tokens: [N3', C']  # Deep layer features
```

### Design Considerations

⚠️ **Challenges:**
1. Early layers may lack capacity if later layers also inject tokens
2. Hard-coded layer assignment may not be optimal

💡 **Improvements to Consider:**
1. **Residual injection** instead of concat to avoid feature collapse
2. **Learnable routing** (MoE-style) to let model decide layer-scale mapping
3. **Parallel branches** for each scale instead of sequential injection
4. **Video Swin Transformer** might already provide hierarchical features

**Architecture Choice:** ViT-3D vs Video Swin Transformer vs TimeSformer?

---

## Phase 2: Region + Slot Structured Representation

### 2.1 Region Tokens (Anatomical)

**Purpose:** Encode predefined anatomical regions

**Method:**
```python
region_tokens = ROIAlign3D(vit_features, region_masks)
```

**Regions:**
- Whole lung (left/right)
- Lobes (RUL, RML, RLL, LUL, LLL)
- Trachea
- Main bronchi
- Mediastinum
- Lesion masks (if available)

**Output:**
```python
region_tokens: [N_regions, C]  # Typically 10-20 tokens
```

### Design Considerations

⚠️ **Challenges:**
1. Requires ground-truth region masks (may not always be available)
2. Fixed regions may not capture all relevant anatomy

💡 **Improvements to Consider:**
1. **Pseudo-labels** from segmentation models (TotalSegmentator, nnU-Net)
2. **Hierarchical regions** (Lung → Lobe → Segment → Lesion) as tree structure
3. **Soft ROI pooling** instead of hard masks for gradients

---

### 2.2 Slot Tokens (Object-Centric)

**Purpose:** Automatically discover object-level representations

**Method:**
```python
slot_tokens = SlotAttention(vit_tokens_all, num_slots=16)
```

**What Slots Capture:**
- Individual nodules
- Vessels
- Airways
- Lymph nodes
- Pleural regions
- Background

**Output:**
```python
slot_tokens: [N_slots, C]  # Typically 8-16 slots
```

### Design Considerations

⚠️ **Challenges:**
1. Medical objects have huge scale variance (1mm nodule vs entire lung)
2. Slots might collapse to "lungs" and "background" without supervision

💡 **Improvements to Consider:**
1. **Scale-conditioned slots** or **hierarchical slots** (Object-Centric Learning with Hierarchy)
2. **Slot supervision**: If bounding boxes available, use Hungarian matching loss (like DETR)
3. **Slot-to-region alignment loss**: Encourage slots to align with anatomical regions
4. **Recurrent slot attention**: Iteratively refine slots over multiple steps

---

## Phase 3: Graph Transformer (Relational Reasoning)

### Motivation
Medical findings are relational:
- "结节位于右上叶" (nodule in RUL)
- "结节靠近胸膜" (nodule near pleura)
- "气道未见狭窄" (airways without stenosis)
- "纵隔未见肿大淋巴结" (mediastinum without enlarged lymph nodes)

### Method

**Graph Construction:**
```python
# Input: slot_tokens + region_tokens
# Build graph with multiple edge types:
graph = build_graph(
    nodes=concat(slot_tokens, region_tokens),
    edges=[spatial_edges, anatomical_edges, learned_edges]
)

graph_tokens = GraphTransformer(graph)
```

**Edge Types:**
1. **Spatial edges**: k-NN in 3D space
2. **Anatomical hierarchy**: lung → lobe → segment → lesion
3. **Learned edges**: from attention weights

**Output:**
```python
graph_tokens: [N_nodes, C]  # All nodes after message passing
```

### Design Considerations

💡 **Hybrid Graph Structure:**
- **Static edges**: anatomical hierarchy (known structure)
- **Dynamic edges**: learned from slot attention
- **Spatial edges**: 3D distance-based (e.g., k=5 nearest neighbors)

🔥 **Medical Priors as Constraints:**
- "Nodule cannot be inside mediastinum"
- "Vessel connects to heart"
- "Airway forms tree structure"

Hard-code these as graph structure constraints.

---

## Phase 4: Hierarchical Cross-Attention to LLM

### Motivation
LLM layers encode different semantic levels:
- **Early layers (1-8)**: lexical, low-level features
- **Middle layers (9-16)**: local composition, syntax
- **Deep layers (17-24)**: global reasoning, semantics
- **Top layers (25-32)**: text generation, style

### Method

**Layer-Specific Token Injection:**

| LLM Layers | Visual Tokens | Purpose |
|------------|---------------|---------|
| **1-8** | `vit_fine_tokens` | Low-level: density, edges, GGO, air bronchogram |
| **9-16** | `vit_mid_tokens` | Mid-level: lobe patterns, consolidation, texture |
| **17-24** | `vit_coarse_tokens` | High-level: global lung structure, mediastinum |
| **25-32** | `region_tokens + slot_tokens + graph_tokens` | Reasoning: objects, relations, anatomy |

**Implementation:**
```python
for layer_idx, llm_layer in enumerate(llm.layers):
    if 1 <= layer_idx <= 8:
        visual_tokens = vit_fine_tokens
    elif 9 <= layer_idx <= 16:
        visual_tokens = vit_mid_tokens
    elif 17 <= layer_idx <= 24:
        visual_tokens = vit_coarse_tokens
    else:  # 25-32
        visual_tokens = concat(region_tokens, slot_tokens, graph_tokens)
    
    hidden_states = llm_layer(
        hidden_states,
        cross_attn_kv=visual_tokens  # Cross-attention
    )
```

### Design Considerations

⚠️ **Challenges:**
1. **Training complexity**: How to train this hierarchical injection?
   - Freeze ViT + train LLM cross-attention only?
   - End-to-end training?
2. **Sequential limitation**: Tokens injected at layer 17 cannot influence layers 1-16

💡 **Alternative Strategies:**

**Option 1: Parallel Injection (Flamingo-style)**
```python
# Inject ALL token types at ALL layers with learnable gating
hidden = hidden + gate_fine[i] * CrossAttn(hidden, vit_fine_tokens)
hidden = hidden + gate_mid[i] * CrossAttn(hidden, vit_mid_tokens)
hidden = hidden + gate_coarse[i] * CrossAttn(hidden, vit_coarse_tokens)
hidden = hidden + gate_struct[i] * CrossAttn(hidden, structured_tokens)
```
Let the model learn which layer needs which tokens.

**Option 2: Top-Down Refinement**
- **Pass 1**: Inject coarse tokens at all layers
- **Pass 2**: Inject fine tokens to refine
- Iterative decoding approach

**Option 3: Use Encoder-Decoder LLM**
- Use ViT as encoder, LLM as decoder (like Flan-T5)
- Cross-attention is built-in
- Simpler than modifying decoder-only Llama

**Option 4: Q-Former (BLIP-2 style)**
```python
# Use learnable query tokens to compress visual features
query_tokens = nn.Parameter(torch.randn(32, C))
compressed = QFormer(query_tokens, all_visual_tokens)
# Only inject compressed queries into LLM (reduces cost)
```

### LLM Model Selection

| Model | Layers | Considerations |
|-------|--------|---------------|
| Llama-3-8B | ~32 | Good fit for 4-level hierarchy |
| Llama-3-13B | ~40 | More layers for finer hierarchy |
| Llama-3-70B | ~80 | Overkill? But best performance |
| Flan-T5-XXL | Enc+Dec | Encoder-decoder might be simpler |

---

## Phase 5: GPT-5.1 for Impression Generation

### Motivation
Medical reasoning (diagnosis, recommendations, urgency) requires:
- Extensive medical knowledge
- Multi-hop reasoning
- Guidelines awareness (Fleischner, Lung-RADS)

Training this from scratch is expensive and data-hungry.

### Method

**Pipeline:**
```
Findings (from Phase 4) → GPT-5.1 API → Impression
```

**Prompt Template:**
```
You are an expert radiologist. Based on the following CT findings, 
provide an impression with:
1. Primary diagnosis or differential
2. Significance assessment
3. Follow-up recommendations

Findings:
{generated_findings}

Impression:
```

### Design Considerations

⚠️ **Potential Issues:**
1. **Hallucination**: GPT-5.1 might add details not in findings
2. **Lack of medical guidelines**: May not follow Fleischner/Lung-RADS
3. **Cost**: API calls can be expensive at scale

💡 **Improvements to Consider:**

**Option 1: Retrieval-Augmented Generation (RAG)**
```python
# Retrieve relevant medical knowledge
knowledge = retrieve_from_kb(findings, sources=["UpToDate", "Radiopedia"])
prompt = f"Findings: {findings}\nKnowledge: {knowledge}\nImpression:"
impression = gpt5(prompt)
```

**Option 2: Constrained Decoding**
- Force GPT-5.1 to generate only from controlled vocabulary
- Use medical ontology (RadLex, SNOMED-CT)
- Prevent hallucination of non-existent findings

**Option 3: LoRA Fine-tuning**
- Fine-tune GPT on small dataset of (Findings → Impression) pairs
- Preserve general knowledge while adapting to radiology style
- Much cheaper than full fine-tuning

**Option 4: Train Your Own Impression Head**
- Add final LLM layers specifically for impression generation
- Train only these layers on (Findings, Impression) pairs
- More control, but requires data

---

## Training Strategy

### Pretraining

**ViT Pretraining:**
1. **Contrastive learning** on large-scale unlabeled CT (NLST, LIDC)
   - CT-CLIP style: CT ↔ radiology text
   - Self-supervised: masked tubelet prediction
2. **Transfer from video models** (VideoMAE, InternVideo)
3. **Multi-task pretraining**: segmentation + classification

**LLM:**
- Use pretrained Llama-3 (don't train from scratch)

### Fine-tuning Strategy

**Option 1: Stage-wise Training**
```
Stage 1: Freeze ViT, train cross-attention only (fast convergence)
Stage 2: Unfreeze top ViT layers, train end-to-end (refinement)
Stage 3: Train hierarchical injection gates (if using parallel injection)
```

**Option 2: End-to-End Training**
- Train everything jointly from start
- Requires more compute and careful hyperparameter tuning
- Better final performance if done right

**Recommended: Stage-wise** (more stable, easier to debug)

---

## Loss Functions

### 1. Report Generation Loss
```python
loss_report = CrossEntropy(predicted_tokens, target_tokens)
```
Standard autoregressive language modeling loss.

### 2. Slot Supervision Loss (if bounding boxes available)
```python
# Hungarian matching like DETR
slot_boxes = predict_boxes_from_slots(slot_tokens)
loss_slot = hungarian_loss(slot_boxes, gt_boxes) + dice_loss(slot_masks, gt_masks)
```

### 3. Graph Structure Loss (if anatomical labels available)
```python
# Enforce correct anatomical hierarchy
loss_graph = edge_classification_loss(predicted_edges, anatomical_edges)
```

### 4. Multi-Scale Consistency Loss
```python
# Ensure fine/mid/coarse tokens encode consistent information
loss_consistency = contrastive_loss(
    pool(tokens_fine), 
    pool(tokens_mid), 
    pool(tokens_coarse)
)
```

### 5. Region Alignment Loss (optional)
```python
# Align slot tokens with region tokens
loss_align = cosine_similarity(slot_tokens, region_tokens)
```

### Total Loss
```python
loss_total = (
    λ1 * loss_report +
    λ2 * loss_slot +
    λ3 * loss_graph +
    λ4 * loss_consistency +
    λ5 * loss_align
)
```

**Recommended weights:** λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.2, λ5=0.1

---

## Evaluation Metrics

### 1. Natural Language Metrics (Baseline)
- **BLEU-1/2/3/4**: N-gram overlap
- **ROUGE-L**: Longest common subsequence
- **METEOR**: Semantic similarity
- **CIDEr**: Consensus-based metric

⚠️ These don't measure clinical accuracy!

### 2. Medical Metrics (Critical)

**CheXbert / RadGraph F1:**
- Extracts clinical entities and relations
- Measures precision/recall of medical facts
- Gold standard for medical report evaluation

**Clinical Entity Accuracy:**
- Precision/Recall/F1 for:
  - Anatomical locations
  - Findings (nodule, consolidation, etc.)
  - Attributes (size, shape, density)
  - Negations (未见, 无)

**Hallucination Rate:**
```
Hallucination % = (False Positive Entities) / (Total Generated Entities)
```
Measure how often the model generates findings not supported by the image.

### 3. Radiologist Evaluation

**Human Assessment (Gold Standard):**
- Clinical accuracy (1-5 scale)
- Completeness (did it miss findings?)
- Relevance (did it report irrelevant details?)
- Actionability (are recommendations appropriate?)

**Inter-rater Agreement:**
- Cohen's Kappa between model and multiple radiologists

### 4. Downstream Task Performance

**Nodule Detection F1:**
If findings mention nodules, can we extract their locations accurately?

**Disease Classification:**
Can we classify diseases (pneumonia, COVID-19, tumor) from generated findings?

---

## Data Requirements

### Minimum Dataset Size
- **10K CT-report pairs**: Basic fine-tuning
- **50K pairs**: Competitive performance
- **100K+ pairs**: State-of-the-art

### Data Augmentation Strategies
1. **3D augmentations**: rotation, flipping, scaling, elastic deformation
2. **Intensity augmentations**: windowing, contrast, noise
3. **Text augmentations**: paraphrase findings with GPT
4. **Synthetic data**: Generate reports for public datasets (LIDC, NLST)

### Public Datasets
- **MIMIC-CXR** (chest X-ray, but adaptable)
- **OpenI** (chest X-ray + reports)
- **LIDC-IDRI** (CT nodules, no reports)
- **NLST** (CT screening, limited reports)
- **Your private dataset** (hopefully large!)

---

## Implementation Considerations

### Compute Requirements

**Training:**
- **ViT-3D**: 4-8 A100 GPUs (3D convolution is expensive)
- **Full pipeline**: 8-16 A100 GPUs for end-to-end training
- **Training time**: 1-2 weeks for 100K samples

**Inference:**
- **Single CT scan**: ~2-5 seconds on A100
- **Batch inference**: 50-100 scans/hour

### Memory Optimization
1. **Gradient checkpointing**: Save memory at cost of speed
2. **Mixed precision (FP16/BF16)**: 2x memory reduction
3. **Flash Attention**: Faster and more memory-efficient attention
4. **LoRA for LLM**: Train only low-rank adapters (~1% parameters)

### Code Framework
- **PyTorch** (most flexible)
- **Hugging Face Transformers** (for LLM)
- **timm** (for ViT)
- **MMDetection3D** (for 3D operations)
- **DGL or PyG** (for graph transformer)

---

## Open Questions & Future Improvements

### Current Uncertainties

1. **Hierarchical injection vs parallel injection?**
   - Need experiments to validate which works better
   - Parallel might be simpler and just as effective

2. **How many slots are optimal?**
   - Too few: Can't capture all objects
   - Too many: Slots collapse or become redundant
   - Typical range: 8-32 slots

3. **Graph structure: learned or fixed?**
   - Fixed anatomical graph might be too rigid
   - Fully learned might lose medical priors
   - Hybrid approach likely best

4. **LLM size: 8B, 13B, or 70B?**
   - Larger = better but slower and more expensive
   - Need to balance performance vs practical deployment

### Future Enhancements

1. **Multi-modal inputs:**
   - Add prior CT scans (temporal comparison)
   - Add clinical history text
   - Add lab results

2. **Interactive report generation:**
   - Allow radiologist to query specific regions
   - Iterative refinement of findings

3. **Uncertainty quantification:**
   - Model should express confidence
   - "Likely nodule" vs "Definite nodule"

4. **Multi-lingual support:**
   - Train on English + Chinese reports
   - Or use translation augmentation

5. **Explainability:**
   - Visualize which image regions contribute to each finding
   - Attention maps from cross-attention
   - Saliency maps for slot tokens

---

## References & Related Work

### Multi-scale Vision Transformers
- Swin Transformer (Liu et al., ICCV 2021)
- Video Swin Transformer (Liu et al., CVPR 2022)
- Multiscale Vision Transformers (Fan et al., ICCV 2021)

### Vision-Language Models
- Flamingo (Alayrac et al., NeurIPS 2022)
- BLIP-2 (Li et al., ICML 2023)
- LLaVA (Liu et al., NeurIPS 2023)

### Medical Report Generation
- R2Gen (Chen et al., ACL 2020)
- M²Transformer (Miura et al., EMNLP 2021)
- Medical-VLBERT (Yan et al., AAAI 2021)
- CheXbert (Smit et al., EMNLP 2020)
- RadGraph (Jain et al., NeurIPS 2021)

### Object-Centric Learning
- Slot Attention (Locatello et al., NeurIPS 2020)
- Object-Centric Learning with Hierarchy (Elsayed et al., NeurIPS 2022)

### Graph Neural Networks for Medical Imaging
- Graph Convolutional Networks (Kipf & Welling, ICLR 2017)
- Graph Attention Networks (Veličković et al., ICLR 2018)

---

## Version History

### v1.0 (2025-11-29)
- Initial architecture design
- Five-phase pipeline defined
- Key design considerations identified
- Training strategy outlined
- Evaluation metrics specified

### Future Versions
- v1.1: After initial experiments and architecture refinements
- v1.2: After ablation studies on hierarchical injection
- v2.0: Production-ready version with optimizations

---

## Contact & Collaboration

**Author:** [Your Name]  
**Institution:** [Your Institution]  
**Email:** [Your Email]

For questions, suggestions, or collaboration opportunities, please reach out!

---

**Last Updated:** 2025-11-29
