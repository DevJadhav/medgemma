# Algorithm Documentation

This document provides detailed explanations of all algorithms, machine learning techniques, and computational methods used in MedAI Compass.

## Table of Contents

1. [Multi-Agent Orchestration](#multi-agent-orchestration)
2. [Intent Classification](#intent-classification)
3. [Diagnostic Workflow Algorithms](#diagnostic-workflow-algorithms)
4. [Training Algorithms](#training-algorithms)
5. [Inference Optimization](#inference-optimization)
6. [Guardrails & Safety](#guardrails--safety)
7. [Distributed Computing](#distributed-computing)

---

## Multi-Agent Orchestration

### Master Orchestrator Algorithm

The Master Orchestrator implements a **hierarchical intent-based routing** algorithm that coordinates three specialized agent domains.

```
Algorithm: MultiAgentOrchestration
Input: UserRequest R with content, attachments, metadata
Output: ProcessedResponse with agent outputs and guardrails

1. APPLY InputGuardrails(R)
   - Jailbreak detection (8 categories + encoding bypass)
   - PHI detection and masking
   - Out-of-scope query filtering

2. intent ← CLASSIFY_INTENT(R.content)
   - Semantic keyword matching with synonym expansion
   - Context phrase detection
   - Confidence calibration

3. ROUTE based on intent.domain:
   - DIAGNOSTIC → LangGraph workflow (image analysis)
   - WORKFLOW → CrewAI crew (clinical operations)
   - COMMUNICATION → AutoGen team (patient engagement)

4. response ← EXECUTE_AGENT(intent.domain, R)

5. APPLY OutputGuardrails(response)
   - Medical disclaimer injection
   - Hallucination prevention
   - Confidence flagging

6. IF response.confidence < THRESHOLD:
   QUEUE_FOR_HUMAN_REVIEW(response)

7. RETURN response with audit logging
```

### Confidence Thresholds

| Agent Type | High Confidence | Medium | Low (Escalation) |
|------------|-----------------|--------|------------------|
| Diagnostic | ≥ 0.90 | 0.80-0.89 | < 0.80 |
| Workflow | ≥ 0.85 | 0.75-0.84 | < 0.75 |
| Communication | ≥ 0.80 | 0.70-0.79 | < 0.70 |

---

## Intent Classification

### Semantic Keyword Matching with Synonym Expansion

The intent classifier uses a **multi-layer matching algorithm**:

```
Algorithm: SemanticIntentClassification
Input: Text query Q
Output: IntentClassification with domain, sub_intent, confidence

1. PREPROCESSING:
   Q_normalized ← lowercase(Q)
   Q_tokens ← tokenize(Q_normalized)

2. KEYWORD MATCHING:
   FOR each domain D in [DIAGNOSTIC, WORKFLOW, COMMUNICATION]:
     base_score[D] ← COUNT(Q_tokens ∩ KEYWORDS[D])

3. SYNONYM EXPANSION:
   FOR each token T in Q_tokens:
     FOR each synonym S in SYNONYMS[T]:
       IF S ∈ KEYWORDS[D]:
         base_score[D] += SYNONYM_WEIGHT (0.8)

4. CONTEXT PHRASE MATCHING:
   FOR each phrase P in CONTEXT_PHRASES[D]:
     IF P ∈ Q_normalized:
       base_score[D] += PHRASE_WEIGHT (1.5)

5. CONFIDENCE CALIBRATION:
   total_score ← sum(base_score.values())
   FOR each domain D:
     confidence[D] ← base_score[D] / max(total_score, 1.0)

6. DETERMINE WINNER:
   best_domain ← argmax(confidence)
   IF confidence[best_domain] < MIN_THRESHOLD (0.3):
     RETURN IntentClassification(UNKNOWN, 0.0)

7. SUB_INTENT DETECTION:
   sub_intent ← DETECT_SUBINTENT(Q, best_domain)

8. RETURN IntentClassification(best_domain, sub_intent, confidence[best_domain])
```

### Synonym Mappings

The system includes 50+ medical synonym mappings:

| Term | Synonyms |
|------|----------|
| x-ray | xray, radiograph, plain film, chest film |
| ct scan | cat scan, computed tomography, contrast ct |
| diagnosis | diagnose, identify, determine, assess, evaluate |
| appointment | visit, office visit, consultation, check-up |

---

## Diagnostic Workflow Algorithms

### LangGraph State Machine

The diagnostic agent implements a **directed acyclic graph (DAG)** for medical image analysis:

```
┌─────────────────┐
│  preprocess     │  DICOM loading, normalization
│  _images        │  Image quality validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  analyze_with   │  MedGemma inference
│  _medgemma      │  Multi-modal understanding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  localize       │  Bounding box detection
│  _findings      │  Anatomical localization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  generate       │  Structured report
│  _report        │  FHIR-compatible output
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  confidence     │  Uncertainty quantification
│  _check         │  Escalation routing
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│finalize│ │ human  │
│        │ │ review │
└────────┘ └───┬────┘
               │
               ▼
           ┌────────┐
           │finalize│
           └────────┘
```

### Modality-Based Routing Algorithm

```
Algorithm: ModalityRouting
Input: DICOM image with metadata
Output: Specialized analysis pipeline

1. EXTRACT modality from DICOM header:
   modality_tag ← dicom.Modality
   body_part ← dicom.BodyPartExamined

2. ROUTING DECISION:
   IF modality_tag == "CR" OR modality_tag == "DX":
     IF body_part == "CHEST":
       RETURN CXRFoundationPipeline
     ELSE:
       RETURN MedGemmaPipeline

   IF modality_tag in ["CT", "MR", "US"]:
     RETURN MedGemmaPipeline

   IF body_part contains "PATHOLOGY" OR file is WSI:
     RETURN PathFoundationPipeline

3. FALLBACK:
   RETURN MedGemmaPipeline (general-purpose)
```

### Finding Localization Algorithm

```
Algorithm: FindingLocalization
Input: Medical image I, findings list F
Output: Annotated image with bounding boxes

1. FOR each finding f in F:

   2. ATTENTION_MAP ← COMPUTE_GRAD_CAM(model, I, f)
      - Extract attention weights from model
      - Apply Grad-CAM visualization

   3. THRESHOLD ← OTSU(ATTENTION_MAP)
      - Automatic threshold selection

   4. BINARY_MASK ← ATTENTION_MAP > THRESHOLD

   5. BOUNDING_BOXES ← FIND_CONTOURS(BINARY_MASK)
      - Connected component analysis
      - Minimum bounding rectangle

   6. FOR each box B in BOUNDING_BOXES:
      IF area(B) > MIN_AREA and area(B) < MAX_AREA:
        ANNOTATE(I, B, f.label, f.confidence)

7. RETURN annotated_image, bounding_boxes
```

---

## Training Algorithms

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA (Low-Rank Adaptation)

LoRA injects trainable low-rank decomposition matrices into frozen model weights:

```
Mathematical Formulation:
  W' = W + BA

Where:
  W ∈ R^(d×k)  : Original frozen weight matrix
  B ∈ R^(d×r)  : Trainable down-projection (r << min(d,k))
  A ∈ R^(r×k)  : Trainable up-projection
  r            : LoRA rank (typically 8-64)

Algorithm: LoRATraining
Input: Model M, Dataset D, rank r, alpha α
Output: Trained adapter weights (A, B)

1. INITIALIZATION:
   FOR each target layer L in [q_proj, k_proj, v_proj, o_proj]:
     A[L] ← GaussianInit(mean=0, std=0.02)
     B[L] ← ZeroInit()

2. FREEZE base model weights W

3. TRAINING LOOP:
   FOR each batch (x, y) in D:

     4. FORWARD PASS with LoRA:
        h = W @ x + (α/r) * (B @ A @ x)

     5. COMPUTE loss L(y, h)

     6. BACKWARD PASS:
        ∂L/∂A, ∂L/∂B ← BACKPROP(L)

     7. UPDATE only A, B parameters

4. MERGE (optional for inference):
   W_merged = W + (α/r) * B @ A
```

**LoRA Configuration:**
```yaml
lora:
  r: 16                    # Rank
  lora_alpha: 32           # Scaling factor
  lora_dropout: 0.05       # Dropout probability
  target_modules:          # Layers to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  bias: none               # Bias training mode
```

#### QLoRA (Quantized LoRA)

QLoRA combines 4-bit quantization with LoRA for memory efficiency:

```
Algorithm: QLoRATraining
Input: Model M, Dataset D, rank r

1. QUANTIZE base model to 4-bit:
   W_q = NF4_QUANTIZE(W)  # NormalFloat4 quantization

2. DOUBLE QUANTIZATION:
   - Quantize quantization constants
   - Reduces memory from 0.5 bits/param to 0.37 bits/param

3. PAGED OPTIMIZER:
   - Offload optimizer states to CPU when GPU OOM
   - Automatic page-in/page-out

4. APPLY LoRA on quantized base:
   h = DEQUANT(W_q) @ x + (α/r) * (B @ A @ x)

5. TRAIN only LoRA adapters (A, B)
```

**Memory Comparison:**
| Method | Memory (27B model) | Trainable Params |
|--------|-------------------|------------------|
| Full Fine-tuning | 200+ GB | 27B (100%) |
| LoRA | 60 GB | 50M (0.18%) |
| QLoRA | 24 GB | 50M (0.18%) |

#### DoRA (Weight-Decomposed LoRA)

DoRA decomposes weight updates into magnitude and direction:

```
Mathematical Formulation:
  W' = m * (W + BA) / ||W + BA||

Where:
  m ∈ R^d      : Trainable magnitude vector
  ||·||        : Column-wise L2 norm

Benefits:
  - Better captures weight magnitude changes
  - Improved performance on complex tasks
  - ~10% better accuracy than LoRA on medical QA
```

#### IA³ (Infused Adapter by Inhibiting and Amplifying)

IA³ uses learned vectors to rescale activations:

```
Algorithm: IA3
Input: Activation h, learned vectors l_k, l_v, l_ff

1. KEY rescaling:
   k' = l_k ⊙ k    # Element-wise multiplication

2. VALUE rescaling:
   v' = l_v ⊙ v

3. FEEDFORWARD rescaling:
   ff' = l_ff ⊙ ff

Trainable Parameters: Only 3 vectors per layer (~0.01% of model)
```

### Alignment Algorithms

#### DPO (Direct Preference Optimization)

```
Algorithm: DirectPreferenceOptimization
Input: Dataset of preference pairs (x, y_w, y_l), reference model π_ref
Output: Aligned policy π_θ

1. LOSS FUNCTION:
   L_DPO = -E[(x,y_w,y_l)] [log σ(β · (r_θ(x,y_w) - r_θ(x,y_l)))]

   Where:
     r_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))
     β = temperature parameter (typically 0.1-0.5)
     y_w = preferred response
     y_l = rejected response

2. TRAINING:
   - No separate reward model needed
   - Direct optimization of preference likelihood
   - Reference model frozen for KL regularization
```

#### GRPO (Group Relative Policy Optimization)

```
Algorithm: GroupRelativePolicyOptimization
Input: Query x, group of responses G = {y_1, ..., y_n}, reward model R

1. COMPUTE rewards for group:
   FOR each y_i in G:
     r_i = R(x, y_i)

2. NORMALIZE within group:
   μ_G = mean(r_i)
   σ_G = std(r_i)
   r̂_i = (r_i - μ_G) / σ_G

3. COMPUTE advantages:
   A_i = r̂_i - baseline

4. POLICY GRADIENT:
   ∇L = E[A_i · ∇log π_θ(y_i|x)]

Benefits:
  - Reduces reward hacking
  - More stable training
  - Better generalization
```

---

## Inference Optimization

### H100-Specific Optimizations

#### Flash Attention 2

```
Algorithm: FlashAttention2
Input: Q, K, V matrices, block sizes B_r, B_c

1. TILING:
   - Split Q into blocks of size B_r
   - Split K, V into blocks of size B_c

2. ONLINE SOFTMAX:
   FOR each Q block:
     FOR each K,V block:
       - Compute block attention scores
       - Update running max and sum
       - Accumulate weighted values

3. MEMORY EFFICIENCY:
   - O(N) memory instead of O(N²)
   - No materialization of attention matrix
   - Fused kernel execution

Performance:
  - 2-4x faster than standard attention
  - Enables longer context (8192+ tokens)
  - 5-10x memory reduction
```

#### CUDA Graph Execution

```
Algorithm: CUDAGraphInference
Input: Model M, input batch X

1. CAPTURE PHASE (once):
   graph = cuda.CUDAGraph()
   with cuda.graph(graph):
     static_input = cuda.empty_like(X)
     static_output = M(static_input)

2. REPLAY PHASE (per inference):
   static_input.copy_(X)
   graph.replay()
   return static_output

Benefits:
  - Eliminates kernel launch overhead
  - 20-40% latency reduction
  - Consistent performance
```

#### KV Cache Optimization

```
Algorithm: PagedKVCache
Input: max_seq_len, num_layers, num_heads, head_dim

1. PAGED ALLOCATION:
   - Allocate cache in fixed-size pages (16-64 tokens)
   - Virtual memory management for efficiency

2. FP8 QUANTIZATION:
   - Quantize KV values to FP8
   - 2x memory reduction
   - Minimal accuracy loss

3. CACHE STRATEGY:
   - LRU eviction for long sequences
   - Prefix caching for repeated prompts
   - Dynamic page allocation

Memory Formula:
  cache_size = num_layers × num_heads × head_dim × seq_len × 2 (K+V)

With FP8: 24GB can hold 8192 tokens for 27B model
```

### Dynamic Batching

```
Algorithm: DynamicBatching
Input: Request queue Q, max_wait_ms, max_batch_size

1. ACCUMULATE requests:
   batch = []
   start_time = now()

   WHILE len(batch) < max_batch_size:
     IF Q.has_request():
       batch.append(Q.pop())

     IF now() - start_time > max_wait_ms:
       BREAK

2. PAD sequences to common length:
   max_len = max(len(r.tokens) for r in batch)
   FOR each request r:
     r.tokens = PAD(r.tokens, max_len)

3. EXECUTE batch inference

4. UNPAD and return individual responses
```

---

## Guardrails & Safety

### PHI Detection Algorithm

```
Algorithm: PHIDetection
Input: Text T
Output: List of PHI matches with types and positions

1. PATTERN MATCHING (30+ patterns):
   matches = []
   FOR each pattern P in PHI_PATTERNS:
     FOR each match M in P.findall(T):
       matches.append(PHIMatch(type=P.type, value=M, position))

2. CONTEXT-AWARE DETECTION:
   FOR each context_pattern CP in CONTEXT_PATTERNS:
     FOR each match M in CP.findall(T):
       # Extract name + identifier pairs
       matches.append(PHIMatch(type="context", value=M))

3. NER-BASED NAME DETECTION:
   entities = NER_MODEL(T)
   FOR each entity E in entities:
     IF E.type == "PERSON":
       IF NOT is_medical_term(E.text):
         matches.append(PHIMatch(type="name", value=E.text))

4. DEDUPLICATION:
   matches = deduplicate(matches, overlap_threshold=0.8)

5. RETURN matches with risk scores
```

**PHI Pattern Coverage:**
| Category | Patterns | Examples |
|----------|----------|----------|
| Core HIPAA | 6 | SSN, MRN, Phone, Email, DOB, Address |
| Extended | 11 | Driver's License, Passport, Medicare, Medicaid |
| Context-Aware | 6 | "Patient John admitted", "Dr. Smith's patient" |
| Financial | 3 | Credit Card, Bank Account, Health Plan ID |

### Jailbreak Detection

```
Algorithm: JailbreakDetection
Input: Query Q
Output: Boolean (is_jailbreak), confidence, category

1. DIRECT CATEGORY DETECTION (8 categories):
   - Prompt injection
   - Role-play attacks
   - Encoding bypass (Base64, ROT13)
   - Instruction override
   - Boundary testing
   - Social engineering
   - Hypothetical scenarios
   - Multi-turn manipulation

2. ENCODING BYPASS DETECTION:
   decoded = TRY_DECODE(Q, [base64, rot13, hex, url])
   IF decoded != Q:
     Q = decoded
     REPEAT detection on decoded

3. LEETSPEAK NORMALIZATION:
   Q_normalized = NORMALIZE_LEETSPEAK(Q)
   # 4 → a, 3 → e, 1 → i, 0 → o, etc.

4. FUZZY MATCHING:
   FOR each known_attack_pattern P:
     similarity = LEVENSHTEIN_RATIO(Q_normalized, P)
     IF similarity > 0.8:
       RETURN (True, similarity, P.category)

5. SEMANTIC ANALYSIS:
   embedding = EMBED(Q)
   FOR each known_attack_embedding E:
     IF cosine_similarity(embedding, E) > THRESHOLD:
       RETURN (True, similarity, E.category)

6. RETURN (False, 0.0, None)
```

---

## Distributed Computing

### 5D Parallelism Strategy

```
Algorithm: 5DParallelism
Dimensions: DP (Data), TP (Tensor), PP (Pipeline), EP (Expert), OP (Optimizer)

1. DATA PARALLELISM (DP):
   - Replicate model across DP groups
   - Split batch across DP workers
   - AllReduce gradients after backward

2. TENSOR PARALLELISM (TP):
   - Split attention heads across TP ranks
   - Column-parallel for Q, K, V projections
   - Row-parallel for output projection
   - AllReduce after attention

3. PIPELINE PARALLELISM (PP):
   - Split layers into PP stages
   - Micro-batching with 1F1B schedule
   - Overlap computation and communication

4. EXPERT PARALLELISM (EP):
   - For MoE models: distribute experts
   - Token routing across expert ranks
   - All-to-all communication for tokens

5. OPTIMIZER PARALLELISM (OP):
   - ZeRO Stage 3: shard optimizer states
   - Gather parameters on demand
   - Reduce memory 8x

Communication Pattern:
  DP: AllReduce (gradients)
  TP: AllReduce (activations)
  PP: Point-to-Point (activations)
  EP: All-to-All (tokens)
  OP: AllGather + ReduceScatter
```

### DeepSpeed ZeRO Stages

```
ZeRO Stage 1: Optimizer State Partitioning
  - Partition optimizer states (momentum, variance)
  - Memory reduction: 4x
  - Communication: AllReduce gradients

ZeRO Stage 2: Gradient Partitioning
  - Partition gradients in addition to optimizer
  - Memory reduction: 8x
  - Communication: ReduceScatter + AllGather

ZeRO Stage 3: Parameter Partitioning
  - Partition everything including parameters
  - Memory reduction: Linear with #GPUs
  - Communication: AllGather before forward/backward

ZeRO-Offload:
  - Offload optimizer states to CPU
  - Offload gradients to CPU
  - NVMe offload for ZeRO-Infinity

Memory Formula (Stage 3):
  per_gpu_memory = model_size / num_gpus + activation_memory
```

### Ray Tune Hyperparameter Optimization

#### ASHA Scheduler

```
Algorithm: ASHAScheduler (Asynchronous Successive Halving)
Input: Trials T, max_t (max resources), reduction_factor η, brackets s

1. INITIALIZATION:
   rungs = [max_t / η^(s-i) for i in range(s)]

2. FOR each trial t in T:

   3. PROMOTE/STOP DECISION at each rung r:
      - Wait until 1/η of trials reach rung r
      - Keep top 1/η trials, stop rest
      - Promotion is asynchronous (no waiting for all)

4. RESOURCE ALLOCATION:
   - New trials start with min resources
   - Promoted trials get more resources
   - Aggressive early stopping

Benefits:
  - 10x faster than random search
  - Handles variable training times
  - Asynchronous = better GPU utilization
```

#### Population-Based Training (PBT)

```
Algorithm: PBT
Input: Population P of size n, hyperparameters H

1. INITIALIZE population:
   FOR i in 1..n:
     P[i] = RandomHyperparameters(H)

2. TRAINING LOOP:
   WHILE not converged:

     3. TRAIN each member for interval t

     4. EXPLOIT (copy from better members):
        FOR each member m:
          IF m.fitness < median(population.fitness):
            best = SelectBest(population)
            m.weights = best.weights

     5. EXPLORE (mutate hyperparameters):
        FOR each member m:
          m.hyperparams = Perturb(m.hyperparams, noise=0.2)

Benefits:
  - No restarts needed
  - Adapts hyperparameters during training
  - Finds good schedules (e.g., learning rate annealing)
```

---

## Custom Kernel Optimizations

### Fused Cross-Entropy Loss

```
CUDA Kernel: FusedCrossEntropyLoss
Input: logits (B×S×V), labels (B×S)
Output: loss scalar, gradients

Standard Implementation (3 kernels):
  1. Softmax: exp(logits) / sum(exp(logits))
  2. Log: log(softmax)
  3. NLLLoss: -log_softmax[label]

Fused Implementation (1 kernel):
  1. Online softmax + log + loss in single pass
  2. Gradient computation fused with forward

Benefits:
  - 3x fewer kernel launches
  - 40% memory bandwidth reduction
  - 20-30% faster training
```

### Fused RoPE (Rotary Position Embeddings)

```
CUDA Kernel: FusedRoPE
Input: Query Q, Key K, position_ids

Algorithm:
  FOR each position p, dimension d:
    θ = 10000^(-2d/dim)
    cos_p = cos(p × θ)
    sin_p = sin(p × θ)

    # Rotate pairs of dimensions
    Q[..., 2d:2d+2] = RotationMatrix(θ_p) @ Q[..., 2d:2d+2]
    K[..., 2d:2d+2] = RotationMatrix(θ_p) @ K[..., 2d:2d+2]

Fusion:
  - Precompute cos/sin tables
  - Fuse with attention kernel
  - Memory-bound → compute-bound optimization
```

### Fused SwiGLU Activation

```
Standard SwiGLU (3 kernels):
  1. Linear1: gate = W_gate @ x
  2. Swish: swish(gate) = gate × sigmoid(gate)
  3. Linear2 + Mul: output = swish(gate) × (W_up @ x)

Fused SwiGLU (1 kernel):
  output = swish(W_gate @ x) × (W_up @ x)

Benefits:
  - Single kernel launch
  - Better memory locality
  - 25% speedup in MLP layers
```

---

## Summary

MedAI Compass implements state-of-the-art algorithms across:

| Category | Algorithms |
|----------|------------|
| Orchestration | Hierarchical intent routing, confidence-based escalation |
| Training | LoRA, QLoRA, DoRA, IA³, DPO, GRPO, KTO, RLHF |
| Inference | Flash Attention 2, CUDA Graphs, Paged KV Cache, vLLM |
| Distribution | 5D Parallelism, DeepSpeed ZeRO, Megatron-LM |
| Safety | PHI detection (30+ patterns), Jailbreak prevention (8 categories) |
| Optimization | ASHA, PBT, Hyperband HPO schedulers |

All algorithms are optimized for NVIDIA H100 GPUs with NVLink interconnect, achieving production-grade performance for medical AI workloads.
