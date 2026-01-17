# 🏥 MedAI Compass: Production-Grade Multi-Agent Medical AI System

## Building with MedGemma & HAI-DEF for the Kaggle Impact Challenge

**Competition:** [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge)  
**Prize Pool:** $100,000  
**Deadline:** February 24, 2026  
**Status:** Active (Started January 13, 2026)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [HAI-DEF Model Ecosystem](#2-hai-def-model-ecosystem)
3. [Medical Datasets by Domain](#3-medical-datasets-by-domain)
4. [Multi-Agent System Architecture](#4-multi-agent-system-architecture)
5. [On-Premises Deployment Strategy](#5-on-premises-deployment-strategy)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Guardrails & Safety Implementation](#7-guardrails--safety-implementation)
8. [Security & HIPAA Compliance](#8-security--hipaa-compliance)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Competition Submission Strategy](#11-competition-submission-strategy)
12. [References & Resources](#12-references--resources)

---

## 1. Executive Summary

### Project Vision: MedAI Compass

A **production-grade, privacy-first multi-agent medical AI system** integrating three clinical domains:

| Domain | Primary Function | HAI-DEF Models |
|--------|------------------|----------------|
| **Diagnostic Imaging** | Radiology/Pathology analysis | MedGemma 1.5 4B, Path Foundation, CXR Foundation |
| **Clinical Workflow** | EHR summarization, documentation | MedGemma 27B, MedASR |
| **Patient Communication** | Intelligent health communication | MedGemma 4B-IT |

### Key Differentiators

```
┌─────────────────────────────────────────────────────────────────┐
│                    MedAI Compass Advantages                     │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Full on-premises deployment (HIPAA compliant)                 │
│ ✓ Multi-agent orchestration with human-in-the-loop              │
│ ✓ 3D imaging + whole-slide pathology (MedGemma 1.5 exclusive)   │
│ ✓ Production-grade inference optimization (vLLM/Triton)         │
│ ✓ Comprehensive safety guardrails with uncertainty quantification│
│ ✓ Real-time clinical decision support with audit trails         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. HAI-DEF Model Ecosystem

### 2.1 Complete Model Catalog

> **Official Resources:**
> - [HAI-DEF Main Site](https://developers.google.com/health-ai-developer-foundations)
> - [Complete HAI-DEF Collection on Hugging Face](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def)
> - [MedGemma Collection](https://huggingface.co/collections/google/medgemma-release)
> - [HAI-DEF Developer Forum](https://discuss.ai.google.dev/c/hai-def/62)

#### 2.1.1 Generative Models

| Model | Parameters | Context | VRAM (BF16) | Key Capabilities | HuggingFace Link |
|-------|------------|---------|-------------|------------------|------------------|
| **MedGemma 1.5 4B** | 4B | 128K | ~8GB | 3D CT/MRI, WSI pathology, longitudinal CXR | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |
| **MedGemma 27B** | 27B | 128K | ~60GB | Complex reasoning, FHIR EHR | [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| **TxGemma 2B** | 2B | 8K | ~4GB | Drug discovery, SMILES | [google/txgemma-2b](https://huggingface.co/google/txgemma-2b) |
| **TxGemma 9B** | 9B | 8K | ~20GB | Therapeutics prediction | [google/txgemma-9b](https://huggingface.co/google/txgemma-9b) |
| **TxGemma 27B** | 27B | 8K | ~60GB | Advanced drug discovery | [google/txgemma-27b](https://huggingface.co/google/txgemma-27b) |

#### 2.1.2 Foundation/Embedding Models

| Model | Architecture | Input Spec | Output | Use Case | Link |
|-------|--------------|------------|--------|----------|------|
| **MedSigLIP** | Dual-tower (800M) | 448×448 + 64 tokens | Aligned embeddings | Zero-shot classification | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |
| **CXR Foundation** | EfficientNet-L2 | DICOM CXR | Language-aligned vectors | Chest X-ray classification | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |
| **Path Foundation** | ViT-S | 224×224 H&E patches | 384-dim embeddings | Tumor detection/grading | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |
| **Derm Foundation** | BiT ResNet-101x3 | 448×448 skin | 6,144-dim embeddings | Dermatology | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |
| **HeAR** | ViT-L (MAE) | 2-sec audio | Audio embeddings | Cough/breath analysis | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |
| **MedASR** | Whisper-based | Medical audio | Transcribed text | Clinical dictation | [HAI-DEF Collection](https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def) |

### 2.2 MedGemma 1.5 4B Deep Dive

> **Documentation:** [MedGemma 1.5 Model Card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)

#### Architecture Specifications

```yaml
Base Model: Gemma 3 4B
Vision Encoder: MedSigLIP (integrated)
Context Length: 128,000 tokens
Image Resolution: 896×896 (encoded to 256 tokens)
Precision: BF16 (default), supports 4-bit/8-bit quantization
Training: Medical domain fine-tuning on clinical data
```

#### Performance Benchmarks

| Benchmark | MedGemma 1.0 | MedGemma 1.5 | Improvement |
|-----------|--------------|--------------|-------------|
| MedQA (USMLE) | 64.4% | 69.0% | +4.6% |
| EHRQA | 85% | 90% | +5% |
| Histopathology ROUGE-L | 0.02 | 0.49 | +2350% |
| Bounding Box IoU | Baseline | +35% | Significant |

#### New Capabilities in 1.5

```
┌─────────────────────────────────────────────────────────────┐
│              MedGemma 1.5 Exclusive Features                 │
├─────────────────────────────────────────────────────────────┤
│ 🔬 3D Medical Imaging: CT volumes, MRI sequences            │
│ 🔍 Whole-Slide Pathology: Gigapixel histopathology images   │
│ 📊 Longitudinal Analysis: Compare CXRs over time            │
│ 📍 Anatomical Localization: Bounding box coordinates        │
│ 🏥 FHIR Navigation: Structured EHR interpretation           │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Inference Optimization

#### Quantization Options

```python
# 4-bit Quantization with bitsandbytes (reduces ~8GB → ~4GB VRAM)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="auto"
)
```

#### vLLM Serving Configuration

```python
# High-throughput inference with vLLM
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/medgemma-4b-it",
    tensor_parallel_size=1,  # Increase for multi-GPU
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    dtype="bfloat16"
)

sampling_params = SamplingParams(
    temperature=0.1,  # Low for medical accuracy
    top_p=0.95,
    max_tokens=2048
)
```

### 2.4 Fine-Tuning with QLoRA

> **Guide:** [Fine-Tune Gemma using QLoRA](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# QLoRA configuration for medical fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Training requires ~8GB VRAM (vs ~40GB for full fine-tuning)
```

---

## 3. Medical Datasets by Domain

### 3.1 Diagnostic Imaging Datasets

#### Chest X-Ray Datasets

| Dataset | Size | Labels | License | Access | Link |
|---------|------|--------|---------|--------|------|
| **MIMIC-CXR** | 377,110 images | CheXpert labels | PhysioNet | Credentialed | [PhysioNet](https://physionet.org/content/mimic-cxr/) |
| **CheXpert** | 224,316 images | 14 pathologies | Research only | Application | [Stanford ML](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **ChestX-ray14** | 112,120 images | 14 pathologies | CC0 Public | Open | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **VinDr-CXR** | 18,000 images | 28 findings | CC BY | Open | [PhysioNet](https://physionet.org/content/vindr-cxr/) |
| **PadChest** | 160,000 images | 174 labels | Research | Application | [BIMCV](https://bimcv.cipf.es/bimcv-projects/padchest/) |

#### CT/MRI Datasets

| Dataset | Modality | Size | Annotations | License | Link |
|---------|----------|------|-------------|---------|------|
| **LIDC-IDRI** | Lung CT | 1,018 cases | 4-radiologist consensus | CC BY 3.0 | [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) |
| **BraTS 2024** | Brain MRI | ~4,500 cases | Tumor segmentation | Competition | [Synapse](https://www.synapse.org/brats) |
| **DeepLesion** | Multi-organ CT | 32,735 lesions | Bounding boxes | CC BY | [NIH](https://nihcc.app.box.com/v/DeepLesion) |
| **RSNA Intracranial** | Head CT | 752,803 images | Hemorrhage labels | Kaggle | [Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) |
| **COVID-CT** | Chest CT | 746 images | COVID-19 labels | Open | [GitHub](https://github.com/UCSD-AI4H/COVID-CT) |

#### Pathology (Whole-Slide Imaging)

| Dataset | Organ | Size | Task | License | Link |
|---------|-------|------|------|---------|------|
| **CAMELYON16** | Breast | 400 WSIs | Metastasis detection | CC0 | [Grand Challenge](https://camelyon16.grand-challenge.org/) |
| **CAMELYON17** | Breast | 1,000 WSIs | pN-stage classification | CC0 | [Grand Challenge](https://camelyon17.grand-challenge.org/) |
| **PANDA** | Prostate | 10,616 WSIs | Gleason grading | Kaggle | [Kaggle](https://www.kaggle.com/c/prostate-cancer-grade-assessment) |
| **TCGA** | Multi-organ | 30,000+ WSIs | Various | Open | [GDC Portal](https://portal.gdc.cancer.gov/) |

### 3.2 Clinical Workflow Datasets

#### Electronic Health Records

| Dataset | Patients | Data Types | Access | Link |
|---------|----------|------------|--------|------|
| **MIMIC-IV** | 65,000+ ICU | Structured + Notes + ECG | PhysioNet Credentialed | [PhysioNet](https://physionet.org/content/mimiciv/3.1/) |
| **MIMIC-III** | 46,000+ ICU | Structured + Notes | PhysioNet Credentialed | [PhysioNet](https://physionet.org/content/mimiciii/) |
| **eICU** | 200,000+ ICU | Multi-center ICU | PhysioNet Credentialed | [PhysioNet](https://physionet.org/content/eicu-crd/) |
| **Synthea** | Synthetic | Full EHR simulation | Open Source | [Synthea](https://synthetichealth.github.io/synthea/) |

#### Clinical NLP Benchmarks

| Dataset | Task | Size | Access | Link |
|---------|------|------|--------|------|
| **n2c2 2022** | Clinical summarization | 1,000+ notes | DUA Required | [n2c2](https://n2c2.dbmi.hms.harvard.edu/data-sets) |
| **n2c2 2018** | Adverse drug events | 500+ notes | DUA Required | [n2c2](https://n2c2.dbmi.hms.harvard.edu/data-sets) |
| **MIMIC-III Notes** | Various NLP | 2M+ notes | PhysioNet | [PhysioNet](https://physionet.org/content/mimiciii/) |
| **MTSamples** | Clinical documentation | 5,000+ samples | Open | [MTSamples](https://www.mtsamples.com/) |

### 3.3 Patient Communication Datasets

| Dataset | Type | Size | License | Link |
|---------|------|------|---------|------|
| **MedDialog-EN** | Consultations | 257,454 dialogues | Research | [GitHub](https://github.com/UCSD-AI4H/Medical-Dialogue-System) |
| **MedQuAD** | Medical Q&A | 47,457 pairs | CC BY | [GitHub](https://github.com/abachaa/MedQuAD) |
| **PubMedQA** | Biomedical Q&A | 1,000 expert | MIT | [PubMedQA](https://pubmedqa.github.io/) |
| **MedMCQA** | Medical MCQ | 194,000 questions | Apache 2.0 | [MedMCQA](https://medmcqa.github.io/) |
| **HealthCareMagic** | Patient queries | 200,000+ | Research | [Kaggle](https://www.kaggle.com/datasets/itachi9604/healthcare-nlp) |

### 3.4 Dataset Access Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                   Dataset Access Timeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OPEN ACCESS (Immediate)                                         │
│  ├── ChestX-ray14, CAMELYON16/17, MedQuAD, Synthea              │
│  └── Download directly, no registration required                 │
│                                                                  │
│  KAGGLE DATASETS (1-2 days)                                      │
│  ├── PANDA, RSNA competitions                                    │
│  └── Kaggle account + competition rules acceptance               │
│                                                                  │
│  PHYSIONET CREDENTIALED (1-2 weeks)                              │
│  ├── MIMIC-IV, MIMIC-CXR, eICU                                   │
│  ├── 1. Create PhysioNet account                                 │
│  ├── 2. Complete CITI training course                            │
│  ├── 3. Sign Data Use Agreement                                  │
│  └── 4. Request access (reviewed by PhysioNet)                   │
│                                                                  │
│  INSTITUTIONAL DUA (2-4 weeks)                                   │
│  ├── n2c2, CheXpert                                              │
│  └── Requires institutional affiliation + signed DUA             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Multi-Agent System Architecture

### 4.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MedAI Compass Architecture                          │
│                     Production-Grade Multi-Agent Medical AI                  │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────┐
                                    │   PATIENT   │
                                    │   PORTAL    │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   Web/Mobile    │    │   EHR System    │    │   PACS/DICOM    │
         │    Interface    │    │  (FHIR R4)      │    │    Server       │
         └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (Kong/Traefik)                         │
│                    Rate Limiting │ Auth │ SSL Termination                    │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MASTER ORCHESTRATOR                                  │
│                    LangGraph StateGraph + Intent Router                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Intent Classification → Route to Domain → Aggregate Results        │    │
│  │  State Management → Checkpoint Persistence → Human Escalation       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└───────────────┬─────────────────────┬─────────────────────┬─────────────────┘
                │                     │                     │
                ▼                     ▼                     ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│   DIAGNOSTIC AGENT    │ │    WORKFLOW AGENT     │ │  COMMUNICATION AGENT  │
│      (LangGraph)      │ │      (CrewAI)         │ │      (AutoGen)        │
├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤
│                       │ │                       │ │                       │
│ ┌───────────────────┐ │ │ ┌───────────────────┐ │ │ ┌───────────────────┐ │
│ │  Image Analyzer   │ │ │ │  Scheduler Agent  │ │ │ │  Triage Agent     │ │
│ │  (MedGemma 1.5)   │ │ │ │                   │ │ │ │  (MedGemma 4B)    │ │
│ └───────────────────┘ │ │ └───────────────────┘ │ │ └───────────────────┘ │
│ ┌───────────────────┐ │ │ ┌───────────────────┐ │ │ ┌───────────────────┐ │
│ │  Report Generator │ │ │ │  Doc Writer Agent │ │ │ │  Educator Agent   │ │
│ │  (MedGemma 27B)   │ │ │ │  (MedGemma 27B)   │ │ │ │  (MedGemma 4B)    │ │
│ └───────────────────┘ │ │ └───────────────────┘ │ │ └───────────────────┘ │
│ ┌───────────────────┐ │ │ ┌───────────────────┐ │ │ ┌───────────────────┐ │
│ │  Pathology Agent  │ │ │ │  Prior Auth Agent │ │ │ │  Follow-up Agent  │ │
│ │  (Path Foundation)│ │ │ │                   │ │ │ │                   │ │
│ └───────────────────┘ │ │ └───────────────────┘ │ │ └───────────────────┘ │
│                       │ │                       │ │                       │
└───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHARED SERVICES LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   FHIR      │  │  Guardrails │  │   Human     │  │   Audit     │        │
│  │   Client    │  │   Service   │  │  Escalation │  │   Logger    │        │
│  │  (HAPI)     │  │  (NeMo)     │  │   Gateway   │  │  (HIPAA)    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Drug       │  │  Medical    │  │  Embedding  │  │  Cache      │        │
│  │  Checker    │  │  Knowledge  │  │   Store     │  │  (Redis)    │        │
│  │  (DrugBank) │  │  (PubMed)   │  │  (pgvector) │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL SERVING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    NVIDIA Triton Inference Server                    │    │
│  │              Dynamic Batching │ Multi-Model │ Ensemble               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ MedGemma 4B  │  │ MedGemma 27B │  │    Path      │  │     CXR      │    │
│  │   (vLLM)     │  │   (vLLM)     │  │  Foundation  │  │  Foundation  │    │
│  │  GPU 0-1     │  │  GPU 2-5     │  │   GPU 6      │  │   GPU 7      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  PostgreSQL  │  │    Redis     │  │   MinIO      │  │ Elasticsearch│    │
│  │  (Sessions,  │  │   (Cache,    │  │  (Medical    │  │   (Audit     │    │
│  │   Checkpts)  │  │   Queues)    │  │   Images)    │  │    Logs)     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                         All encrypted at rest (AES-256)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Framework Selection Rationale

> **Comparison Guide:** [CrewAI vs LangGraph vs AutoGen](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)

| Domain | Framework | Rationale |
|--------|-----------|-----------|
| **Diagnostic** | LangGraph | Complex multi-step workflows with conditional branching; checkpoint persistence for long diagnostic pipelines; human-in-the-loop gates at critical decisions |
| **Workflow** | CrewAI | Role-based agent teams (scheduler, documenter, authorizer); sequential task execution; clear responsibility delegation |
| **Communication** | AutoGen | Dynamic conversational patterns; adaptive responses; `human_input_mode` for clinical verification |

### 4.3 LangGraph Diagnostic Agent Implementation

> **Documentation:** [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# State definition for medical sessions
class DiagnosticState(TypedDict):
    patient_id: str
    session_id: str
    images: list[str]  # DICOM paths
    findings: list[dict]
    confidence_scores: list[float]
    requires_review: bool
    audit_trail: list[dict]
    fhir_context: dict

# Define the diagnostic workflow graph
def create_diagnostic_graph():
    workflow = StateGraph(DiagnosticState)
    
    # Add nodes for each diagnostic step
    workflow.add_node("preprocess_images", preprocess_images)
    workflow.add_node("analyze_with_medgemma", analyze_with_medgemma)
    workflow.add_node("pathology_analysis", pathology_analysis)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("confidence_check", confidence_check)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize", finalize)
    
    # Define edges with conditional routing
    workflow.set_entry_point("preprocess_images")
    workflow.add_edge("preprocess_images", "analyze_with_medgemma")
    workflow.add_edge("analyze_with_medgemma", "pathology_analysis")
    workflow.add_edge("pathology_analysis", "generate_report")
    workflow.add_edge("generate_report", "confidence_check")
    
    # Conditional routing based on confidence
    workflow.add_conditional_edges(
        "confidence_check",
        route_by_confidence,
        {
            "high_confidence": "finalize",
            "low_confidence": "human_review"
        }
    )
    workflow.add_edge("human_review", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile(
        checkpointer=PostgresSaver.from_conn_string(
            "postgresql://user:pass@localhost/medai_sessions"
        )
    )

def route_by_confidence(state: DiagnosticState) -> str:
    """Route based on minimum confidence score"""
    min_confidence = min(state["confidence_scores"]) if state["confidence_scores"] else 0
    
    # Clinical thresholds
    if min_confidence >= 0.95:
        return "high_confidence"
    else:
        return "low_confidence"
```

### 4.4 CrewAI Workflow Agent Implementation

```python
from crewai import Agent, Task, Crew, Process

# Define workflow agents
scheduler_agent = Agent(
    role="Medical Scheduler",
    goal="Optimize appointment scheduling and resource allocation",
    backstory="Expert in healthcare operations with deep knowledge of clinical workflows",
    llm="google/medgemma-4b-it",
    verbose=True
)

documenter_agent = Agent(
    role="Clinical Documentation Specialist",
    goal="Generate accurate, compliant clinical documentation",
    backstory="Experienced medical transcriptionist with expertise in clinical terminology",
    llm="google/medgemma-27b-it",
    verbose=True
)

prior_auth_agent = Agent(
    role="Prior Authorization Specialist",
    goal="Streamline insurance authorization processes",
    backstory="Expert in healthcare billing codes and insurance requirements",
    llm="google/medgemma-4b-it",
    verbose=True
)

# Define workflow tasks
documentation_task = Task(
    description="Generate discharge summary from clinical notes: {notes}",
    expected_output="Structured discharge summary following hospital template",
    agent=documenter_agent
)

scheduling_task = Task(
    description="Schedule follow-up appointment based on: {diagnosis}",
    expected_output="Appointment confirmation with optimal timing",
    agent=scheduler_agent
)

# Create the workflow crew
workflow_crew = Crew(
    agents=[scheduler_agent, documenter_agent, prior_auth_agent],
    tasks=[documentation_task, scheduling_task],
    process=Process.sequential,
    verbose=True
)
```

### 4.5 AutoGen Communication Agent Implementation

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configure MedGemma as the LLM backend
llm_config = {
    "model": "google/medgemma-4b-it",
    "api_type": "openai",
    "base_url": "http://localhost:8000/v1",  # vLLM endpoint
    "temperature": 0.1  # Low temperature for medical accuracy
}

# Patient-facing triage agent
triage_agent = AssistantAgent(
    name="TriageNurse",
    system_message="""You are an experienced triage nurse helping patients 
    understand their symptoms. Always recommend professional medical consultation 
    for serious symptoms. Never diagnose conditions.""",
    llm_config=llm_config
)

# Health education agent
educator_agent = AssistantAgent(
    name="HealthEducator",
    system_message="""You are a patient health educator. Explain medical 
    concepts in simple terms. Provide evidence-based health information 
    from reputable sources like CDC, WHO, and NIH.""",
    llm_config=llm_config
)

# Human proxy for clinical oversight
clinical_proxy = UserProxyAgent(
    name="ClinicalOversight",
    human_input_mode="ALWAYS",  # Require human approval for medical advice
    code_execution_config=False
)

# Group chat for complex patient interactions
group_chat = GroupChat(
    agents=[triage_agent, educator_agent, clinical_proxy],
    messages=[],
    max_round=10
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)
```

### 4.6 Master Orchestrator Integration

```python
from enum import Enum
from pydantic import BaseModel

class IntentType(Enum):
    DIAGNOSTIC = "diagnostic"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"
    HYBRID = "hybrid"

class UserIntent(BaseModel):
    intent_type: IntentType
    confidence: float
    entities: dict
    requires_phi: bool

class MasterOrchestrator:
    def __init__(self):
        self.diagnostic_graph = create_diagnostic_graph()
        self.workflow_crew = workflow_crew
        self.communication_manager = manager
        self.intent_classifier = self._load_intent_classifier()
        
    async def process_request(self, request: dict, session_id: str):
        # Classify intent
        intent = self._classify_intent(request)
        
        # Audit logging
        self._log_audit("request_received", request, session_id)
        
        # Route to appropriate domain
        if intent.intent_type == IntentType.DIAGNOSTIC:
            result = await self._handle_diagnostic(request, session_id)
        elif intent.intent_type == IntentType.WORKFLOW:
            result = await self._handle_workflow(request)
        elif intent.intent_type == IntentType.COMMUNICATION:
            result = await self._handle_communication(request)
        else:  # HYBRID
            result = await self._handle_hybrid(request, session_id)
            
        # Apply output guardrails
        result = self._apply_guardrails(result)
        
        return result
    
    async def _handle_diagnostic(self, request: dict, session_id: str):
        config = {"configurable": {"thread_id": session_id}}
        return await self.diagnostic_graph.ainvoke(request, config)
    
    async def _handle_workflow(self, request: dict):
        return self.workflow_crew.kickoff(inputs=request)
    
    async def _handle_communication(self, request: dict):
        return await self.communication_manager.a_initiate_chat(
            self.communication_manager,
            message=request["message"]
        )
```

---

## 5. On-Premises Deployment Strategy

### 5.1 Hardware Requirements

#### Minimum Configuration (Development/Demo)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | 1× NVIDIA RTX 4090 (24GB) | MedGemma 4B inference |
| **CPU** | AMD EPYC 7313 or Intel Xeon Gold | General processing |
| **RAM** | 64GB DDR4 ECC | Model loading, batching |
| **Storage** | 1TB NVMe SSD + 4TB HDD | Models + medical images |
| **Network** | 10GbE | PACS integration |

#### Production Configuration (Hospital Deployment)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | 4× NVIDIA A100 80GB or 8× NVIDIA L40S | MedGemma 27B + concurrent inference |
| **CPU** | 2× AMD EPYC 9654 (96 cores each) | Parallel preprocessing |
| **RAM** | 512GB DDR5 ECC | Large batch processing |
| **Storage** | 8TB NVMe RAID + 100TB enterprise SAN | High-throughput medical imaging |
| **Network** | 100GbE + InfiniBand | Low-latency GPU communication |
| **Backup** | Offsite encrypted backup | HIPAA disaster recovery |

### 5.2 Software Stack

```yaml
# docker-compose.yml for MedAI Compass
version: '3.8'

services:
  # Model Serving
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    runtime: nvidia
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    volumes:
      - ./models:/models
      - ./triton-config:/config
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8080:8000"
    volumes:
      - ./model-cache:/root/.cache/huggingface
    command: >
      --model google/medgemma-4b-it
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
      --max-model-len 8192
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

  # Databases
  postgres:
    image: postgres:16-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
      - POSTGRES_DB=medai_compass
    secrets:
      - db_password

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data

  # Object Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    environment:
      - MINIO_ROOT_USER=${MINIO_USER}
      - MINIO_ROOT_PASSWORD=${MINIO_PASSWORD}

  # Observability
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards

  # Application
  medai-api:
    build: ./api
    depends_on:
      - postgres
      - redis
      - vllm
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres/medai_compass
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - VLLM_ENDPOINT=http://vllm:8000/v1
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /encrypted-storage/postgres
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### 5.3 Triton Model Repository Structure

```
models/
├── medgemma_4b/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.py
│   └── tokenizer/
├── medgemma_27b/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
├── path_foundation/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── cxr_foundation/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── diagnostic_ensemble/
    └── config.pbtxt  # Combines multiple models
```

#### Triton Config for MedGemma 4B

```protobuf
# models/medgemma_4b/config.pbtxt
name: "medgemma_4b"
backend: "vllm"
max_batch_size: 8

input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [-1]
  },
  {
    name: "image_input"
    data_type: TYPE_FP16
    dims: [-1, 3, 896, 896]
  }
]

output [
  {
    name: "text_output"
    data_type: TYPE_STRING
    dims: [-1]
  },
  {
    name: "confidence"
    data_type: TYPE_FP32
    dims: [1]
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters: {
  key: "model_path"
  value: {string_value: "google/medgemma-4b-it"}
}
```

### 5.4 Air-Gapped Deployment

```bash
#!/bin/bash
# air-gap-setup.sh - Prepare for isolated deployment

# 1. Download all models on connected system
mkdir -p /transfer/models
huggingface-cli download google/medgemma-4b-it --local-dir /transfer/models/medgemma-4b-it
huggingface-cli download google/medgemma-27b-it --local-dir /transfer/models/medgemma-27b-it

# 2. Save Docker images
docker pull nvcr.io/nvidia/tritonserver:24.01-py3
docker pull vllm/vllm-openai:latest
docker pull postgres:16-alpine
docker pull redis:7-alpine

docker save -o /transfer/images/triton.tar nvcr.io/nvidia/tritonserver:24.01-py3
docker save -o /transfer/images/vllm.tar vllm/vllm-openai:latest
docker save -o /transfer/images/postgres.tar postgres:16-alpine
docker save -o /transfer/images/redis.tar redis:7-alpine

# 3. Generate checksums
sha256sum /transfer/models/*/* > /transfer/checksums.txt
sha256sum /transfer/images/* >> /transfer/checksums.txt

# 4. Transfer to air-gapped system via secure media
# On air-gapped system:
# docker load -i /transfer/images/triton.tar
# Verify: sha256sum -c /transfer/checksums.txt
```

---

## 6. Evaluation Framework

### 6.1 Multi-Domain Evaluation Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MedAI Compass Evaluation Framework                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    DIAGNOSTIC IMAGING METRICS                        │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  • AUC-ROC per pathology (Target: ≥0.85)                            │    │
│  │  • Sensitivity @ 95% specificity                                     │    │
│  │  • Dice coefficient for segmentation (Target: ≥0.75)                │    │
│  │  • Localization IoU (Target: ≥0.50)                                 │    │
│  │  • Inter-rater agreement vs radiologists (Cohen's κ)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CLINICAL NLP METRICS                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  • Entity Extraction F1 (Target: ≥0.90)                             │    │
│  │  • ROUGE-L for summarization (Target: ≥0.45)                        │    │
│  │  • BERTScore F1 (Target: ≥0.85)                                     │    │
│  │  • Factual accuracy (human eval) (Target: ≥95%)                     │    │
│  │  • Hallucination rate (Target: <2%)                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PATIENT COMMUNICATION METRICS                     │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  • Response appropriateness (human eval) (Target: ≥90%)             │    │
│  │  • Medical accuracy (clinician review) (Target: ≥95%)               │    │
│  │  • Harm potential score (Target: <1%)                               │    │
│  │  • Readability (Flesch-Kincaid Grade) (Target: 6-8)                 │    │
│  │  • Empathy score (human eval) (Target: ≥4/5)                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SYSTEM-LEVEL METRICS                              │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  • End-to-end latency P95 (Target: <5s diagnostic, <2s chat)        │    │
│  │  • Throughput (Target: ≥100 requests/min)                           │    │
│  │  • Human escalation rate (Target: 10-20%)                           │    │
│  │  • System availability (Target: ≥99.9%)                             │    │
│  │  • Security audit pass rate (Target: 100%)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Evaluation Implementation

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
from bert_score import score as bert_score

@dataclass
class DiagnosticEvaluation:
    """Evaluation results for diagnostic imaging"""
    auc_roc: dict[str, float]  # Per-pathology AUC
    sensitivity_at_95_spec: dict[str, float]
    dice_scores: Optional[dict[str, float]] = None
    localization_iou: Optional[float] = None
    
@dataclass  
class NLPEvaluation:
    """Evaluation results for clinical NLP"""
    entity_f1: float
    rouge_l: float
    bert_score_f1: float
    factual_accuracy: float  # Human-evaluated
    hallucination_rate: float

@dataclass
class CommunicationEvaluation:
    """Evaluation results for patient communication"""
    appropriateness_score: float  # Human-evaluated
    medical_accuracy: float  # Clinician-reviewed
    harm_potential: float
    readability_grade: float
    empathy_score: float  # Human-evaluated

class MedAIEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def evaluate_diagnostic(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray,
        pathology_names: list[str]
    ) -> DiagnosticEvaluation:
        """Evaluate diagnostic imaging performance"""
        auc_scores = {}
        sensitivity_scores = {}
        
        for i, pathology in enumerate(pathology_names):
            # AUC-ROC
            auc_scores[pathology] = roc_auc_score(
                labels[:, i], 
                predictions[:, i]
            )
            
            # Sensitivity at 95% specificity
            sensitivity_scores[pathology] = self._sensitivity_at_specificity(
                labels[:, i],
                predictions[:, i],
                target_specificity=0.95
            )
            
        return DiagnosticEvaluation(
            auc_roc=auc_scores,
            sensitivity_at_95_spec=sensitivity_scores
        )
    
    def evaluate_nlp(
        self,
        generated_texts: list[str],
        reference_texts: list[str],
        extracted_entities: list[dict],
        reference_entities: list[dict]
    ) -> NLPEvaluation:
        """Evaluate clinical NLP performance"""
        # ROUGE-L
        rouge_scores = [
            self.rouge_scorer.score(ref, gen)['rougeL'].fmeasure
            for ref, gen in zip(reference_texts, generated_texts)
        ]
        avg_rouge_l = np.mean(rouge_scores)
        
        # BERTScore
        P, R, F1 = bert_score(
            generated_texts, 
            reference_texts, 
            lang='en',
            model_type='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        )
        avg_bert_f1 = F1.mean().item()
        
        # Entity F1 (simplified)
        entity_f1 = self._calculate_entity_f1(extracted_entities, reference_entities)
        
        return NLPEvaluation(
            entity_f1=entity_f1,
            rouge_l=avg_rouge_l,
            bert_score_f1=avg_bert_f1,
            factual_accuracy=0.0,  # Requires human evaluation
            hallucination_rate=0.0  # Requires human evaluation
        )
    
    def _sensitivity_at_specificity(
        self, 
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        target_specificity: float
    ) -> float:
        """Calculate sensitivity at target specificity"""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # Find threshold where specificity >= target
        specificity = 1 - fpr
        idx = np.where(specificity >= target_specificity)[0]
        if len(idx) == 0:
            return 0.0
        return tpr[idx[-1]]
    
    def _calculate_entity_f1(
        self, 
        predicted: list[dict], 
        reference: list[dict]
    ) -> float:
        """Calculate entity extraction F1 score"""
        # Simplified implementation
        total_tp, total_fp, total_fn = 0, 0, 0
        for pred, ref in zip(predicted, reference):
            pred_set = set(pred.get('entities', []))
            ref_set = set(ref.get('entities', []))
            total_tp += len(pred_set & ref_set)
            total_fp += len(pred_set - ref_set)
            total_fn += len(ref_set - pred_set)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
```

### 6.3 Continuous Monitoring Dashboard

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
INFERENCE_LATENCY = Histogram(
    'medai_inference_latency_seconds',
    'Inference latency in seconds',
    ['model', 'domain'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'medai_prediction_confidence',
    'Model confidence scores',
    ['model', 'pathology'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

HUMAN_ESCALATIONS = Counter(
    'medai_human_escalations_total',
    'Total human escalations',
    ['domain', 'reason']
)

MODEL_DRIFT_SCORE = Gauge(
    'medai_model_drift_score',
    'Model drift detection score',
    ['model']
)

ACTIVE_SESSIONS = Gauge(
    'medai_active_sessions',
    'Number of active patient sessions'
)

class MetricsCollector:
    @staticmethod
    def record_inference(model: str, domain: str, latency: float, confidence: float):
        INFERENCE_LATENCY.labels(model=model, domain=domain).observe(latency)
        PREDICTION_CONFIDENCE.labels(model=model, pathology=domain).observe(confidence)
    
    @staticmethod
    def record_escalation(domain: str, reason: str):
        HUMAN_ESCALATIONS.labels(domain=domain, reason=reason).inc()
    
    @staticmethod
    def update_drift_score(model: str, score: float):
        MODEL_DRIFT_SCORE.labels(model=model).set(score)
```

### 6.4 Model Drift Detection

```python
import numpy as np
from scipy import stats

class DriftDetector:
    """Detect distribution shift in model inputs and outputs"""
    
    def __init__(self, reference_embeddings: np.ndarray, threshold: float = 0.1):
        self.reference = reference_embeddings
        self.threshold = threshold
        
    def calculate_psi(
        self, 
        current: np.ndarray, 
        bins: int = 10
    ) -> float:
        """Population Stability Index for drift detection"""
        # Create bins from reference distribution
        _, bin_edges = np.histogram(self.reference, bins=bins)
        
        # Calculate proportions
        ref_counts, _ = np.histogram(self.reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        ref_props = ref_counts / len(self.reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.clip(ref_props, 1e-10, 1)
        cur_props = np.clip(cur_props, 1e-10, 1)
        
        # PSI calculation
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return psi
    
    def ks_test(self, current: np.ndarray) -> tuple[float, float]:
        """Kolmogorov-Smirnov test for distribution comparison"""
        statistic, p_value = stats.ks_2samp(self.reference.flatten(), current.flatten())
        return statistic, p_value
    
    def check_drift(self, current: np.ndarray) -> dict:
        """Comprehensive drift check"""
        psi = self.calculate_psi(current)
        ks_stat, ks_pvalue = self.ks_test(current)
        
        drift_detected = psi > self.threshold or ks_pvalue < 0.05
        
        return {
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "drift_detected": drift_detected,
            "severity": "high" if psi > 0.25 else "medium" if psi > 0.1 else "low"
        }
```

---

## 7. Guardrails & Safety Implementation

### 7.1 Layered Safety Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MedAI Compass Safety Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      INPUT GUARDRAILS                                │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │ PHI/PII     │  │ Jailbreak   │  │   Scope     │  │  Content    │ │    │
│  │  │ Detection   │  │ Detection   │  │ Validation  │  │  Filtering  │ │    │
│  │  │ & Masking   │  │ (LlamaGuard)│  │             │  │             │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PROCESSING GUARDRAILS                             │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │ RAG with    │  │   Drug      │  │ Contra-     │  │ Uncertainty │ │    │
│  │  │ Medical KB  │  │ Interaction │  │ indication  │  │ Quantifi-   │ │    │
│  │  │ (PubMed)    │  │ Check       │  │ Detection   │  │ cation      │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     OUTPUT GUARDRAILS                                │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │ Medical     │  │ Citation    │  │Hallucination│  │ Confidence  │ │    │
│  │  │ Terminology │  │ Verification│  │ Detection   │  │ Scoring     │ │    │
│  │  │ Validation  │  │             │  │(SelfCheck)  │  │             │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   HUMAN-IN-THE-LOOP GATEWAY                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │ Automatic   │  │ Clinician   │  │   Audit     │                  │    │
│  │  │ Escalation  │  │ Review UI   │  │   Trail     │                  │    │
│  │  │ Triggers    │  │             │  │ Recording   │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 NeMo Guardrails Configuration

> **Documentation:** [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

```yaml
# config/guardrails.yaml
models:
  - type: main
    engine: vllm
    model: google/medgemma-4b-it
    
  - type: content_safety
    engine: nvidia_ai_endpoints
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
    
  - type: jailbreak_detection  
    engine: nvidia_ai_endpoints
    model: nvidia/llama-guard-3-8b

rails:
  input:
    flows:
      - check medical scope
      - content safety check input
      - jailbreak detection
      - phi detection and masking
      
  output:
    flows:
      - content safety check output
      - medical terminology validation
      - hallucination detection
      - confidence threshold check
      - add clinical disclaimer

prompts:
  - task: check_medical_scope
    content: |
      Determine if the following query is within the medical AI assistant's scope.
      Valid scopes: diagnostic imaging interpretation, clinical documentation, 
      patient health education, appointment scheduling.
      
      Query: {{ user_input }}
      
      Respond with: IN_SCOPE, OUT_OF_SCOPE, or REQUIRES_CLINICIAN
      
  - task: add_clinical_disclaimer
    content: |
      Add appropriate medical disclaimer to the response based on content type:
      - Diagnostic: "This analysis is for clinical decision support only..."
      - Treatment: "Please consult with your healthcare provider..."
      - General: "This information is educational and not medical advice..."
```

### 7.3 Guardrails Implementation

```python
from nemoguardrails import RailsConfig, LLMRails
from typing import Optional
import re

class MedicalGuardrails:
    def __init__(self, config_path: str = "config/guardrails.yaml"):
        self.config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(self.config)
        self.phi_patterns = self._compile_phi_patterns()
        
    def _compile_phi_patterns(self) -> dict:
        """Compile regex patterns for PHI detection"""
        return {
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "mrn": re.compile(r'\bMRN[:\s]*\d{6,10}\b', re.I),
            "phone": re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "email": re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b'),
            "dob": re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}\b'),
            "address": re.compile(r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', re.I)
        }
    
    def detect_and_mask_phi(self, text: str) -> tuple[str, list[str]]:
        """Detect and mask PHI in input text"""
        detected_phi = []
        masked_text = text
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                detected_phi.append(f"{phi_type}: {match}")
                masked_text = masked_text.replace(match, f"[{phi_type.upper()}_REDACTED]")
                
        return masked_text, detected_phi
    
    async def process_input(self, user_input: str) -> dict:
        """Apply input guardrails"""
        # PHI detection
        masked_input, detected_phi = self.detect_and_mask_phi(user_input)
        
        # Run through NeMo Guardrails
        result = await self.rails.generate_async(
            messages=[{"role": "user", "content": masked_input}]
        )
        
        return {
            "processed_input": masked_input,
            "detected_phi": detected_phi,
            "guardrails_response": result,
            "blocked": result.get("blocked", False),
            "block_reason": result.get("block_reason", None)
        }
    
    async def process_output(
        self, 
        response: str, 
        confidence: float,
        domain: str
    ) -> dict:
        """Apply output guardrails"""
        # Confidence check
        requires_review = confidence < 0.80
        
        # Add appropriate disclaimer
        disclaimer = self._get_disclaimer(domain, confidence)
        
        # Hallucination check (simplified)
        hallucination_score = await self._check_hallucination(response)
        
        return {
            "response": response,
            "disclaimer": disclaimer,
            "confidence": confidence,
            "requires_review": requires_review or hallucination_score > 0.3,
            "hallucination_score": hallucination_score
        }
    
    def _get_disclaimer(self, domain: str, confidence: float) -> str:
        """Generate appropriate medical disclaimer"""
        disclaimers = {
            "diagnostic": (
                "⚠️ This AI-assisted analysis is for clinical decision support only. "
                f"Confidence: {confidence:.1%}. "
                "All findings should be verified by a qualified radiologist."
            ),
            "treatment": (
                "⚠️ This information is not a substitute for professional medical advice. "
                "Please consult with your healthcare provider before making treatment decisions."
            ),
            "communication": (
                "ℹ️ This is general health information and not personalized medical advice. "
                "For specific concerns, please consult a healthcare professional."
            )
        }
        return disclaimers.get(domain, disclaimers["communication"])
    
    async def _check_hallucination(self, response: str) -> float:
        """Check for potential hallucinations using SelfCheckGPT approach"""
        # Simplified implementation - in production, use multiple samples
        # and check consistency
        return 0.1  # Placeholder
```

### 7.4 Uncertainty Quantification

```python
import torch
import numpy as np
from typing import Callable

class UncertaintyEstimator:
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model: torch.nn.Module, n_samples: int = 30):
        self.model = model
        self.n_samples = n_samples
        
    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
                
    def estimate_uncertainty(
        self, 
        inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate prediction uncertainty using MC Dropout
        
        Returns:
            mean_prediction: Average prediction across samples
            uncertainty: Standard deviation across samples
        """
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(inputs)
                predictions.append(output)
                
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_prediction, uncertainty
    
    def should_escalate(
        self, 
        uncertainty: torch.Tensor, 
        threshold: float = 0.15
    ) -> bool:
        """Determine if prediction should be escalated to human review"""
        max_uncertainty = uncertainty.max().item()
        return max_uncertainty > threshold


class EnsembleUncertainty:
    """Ensemble-based uncertainty for production systems"""
    
    def __init__(self, models: list[torch.nn.Module]):
        self.models = models
        
    def predict_with_uncertainty(
        self, 
        inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Get ensemble predictions with uncertainty metrics"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(inputs)
                predictions.append(output)
                
        predictions = torch.stack(predictions)
        
        # Metrics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)  # Model uncertainty
        
        # For classification, also compute predictive entropy
        if predictions.dim() > 2:  # Classification logits
            probs = torch.softmax(predictions, dim=-1).mean(dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        else:
            entropy = None
            
        metrics = {
            "epistemic_uncertainty": epistemic_uncertainty.mean().item(),
            "max_uncertainty": epistemic_uncertainty.max().item(),
            "entropy": entropy.mean().item() if entropy is not None else None,
            "model_agreement": self._calculate_agreement(predictions)
        }
        
        return mean_pred, epistemic_uncertainty, metrics
    
    def _calculate_agreement(self, predictions: torch.Tensor) -> float:
        """Calculate inter-model agreement"""
        if predictions.dim() > 2:  # Classification
            pred_classes = predictions.argmax(dim=-1)
            mode_class = torch.mode(pred_classes, dim=0).values
            agreement = (pred_classes == mode_class).float().mean().item()
        else:  # Regression
            std = predictions.std(dim=0)
            agreement = 1.0 / (1.0 + std.mean().item())
        return agreement
```

### 7.5 Human Escalation Triggers

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import re

class EscalationReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    CRITICAL_FINDING = "critical_finding"
    HIGH_UNCERTAINTY = "high_uncertainty"
    SAFETY_CONCERN = "safety_concern"
    OUT_OF_SCOPE = "out_of_scope"
    PATIENT_REQUEST = "patient_request"
    SYSTEM_ERROR = "system_error"

@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: Optional[EscalationReason]
    priority: str  # "immediate", "urgent", "routine"
    details: str

class HumanEscalationGateway:
    """Determine when to escalate to human clinicians"""
    
    # Critical findings requiring immediate escalation
    CRITICAL_PATTERNS = [
        r"pneumothorax",
        r"pulmonary\s+embol",
        r"aortic\s+dissection",
        r"stroke|CVA",
        r"myocardial\s+infarction|heart\s+attack|MI\b",
        r"intracranial\s+hemorrhage",
        r"tension\s+pneumo",
        r"cardiac\s+arrest",
        r"anaphyla",
    ]
    
    # Safety concern patterns in patient communication
    SAFETY_PATTERNS = [
        r"suicid",
        r"self[- ]?harm",
        r"kill\s+(myself|me)",
        r"end\s+(my|it)\s+life",
        r"want\s+to\s+die",
        r"overdose",
        r"abuse",
    ]
    
    def __init__(self):
        self.critical_regex = [re.compile(p, re.I) for p in self.CRITICAL_PATTERNS]
        self.safety_regex = [re.compile(p, re.I) for p in self.SAFETY_PATTERNS]
        
    def evaluate(
        self,
        response: str,
        confidence: float,
        uncertainty: float,
        domain: str,
        user_input: Optional[str] = None
    ) -> EscalationDecision:
        """Evaluate if human escalation is needed"""
        
        # Check for critical findings (highest priority)
        for pattern in self.critical_regex:
            if pattern.search(response):
                return EscalationDecision(
                    should_escalate=True,
                    reason=EscalationReason.CRITICAL_FINDING,
                    priority="immediate",
                    details=f"Critical finding detected: {pattern.pattern}"
                )
        
        # Check for safety concerns in patient communication
        if user_input:
            for pattern in self.safety_regex:
                if pattern.search(user_input):
                    return EscalationDecision(
                        should_escalate=True,
                        reason=EscalationReason.SAFETY_CONCERN,
                        priority="immediate",
                        details="Patient safety concern detected"
                    )
        
        # Check confidence threshold
        confidence_thresholds = {
            "diagnostic": 0.90,
            "workflow": 0.85,
            "communication": 0.80
        }
        threshold = confidence_thresholds.get(domain, 0.85)
        
        if confidence < threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                priority="urgent" if confidence < 0.70 else "routine",
                details=f"Confidence {confidence:.1%} below threshold {threshold:.1%}"
            )
        
        # Check uncertainty threshold
        if uncertainty > 0.20:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.HIGH_UNCERTAINTY,
                priority="urgent" if uncertainty > 0.30 else "routine",
                details=f"Uncertainty {uncertainty:.1%} exceeds threshold"
            )
        
        # No escalation needed
        return EscalationDecision(
            should_escalate=False,
            reason=None,
            priority="none",
            details="All checks passed"
        )
```

---

## 8. Security & HIPAA Compliance

### 8.1 HIPAA Technical Safeguards (2025 Updates)

> **Reference:** [HIPAA Security Rule Updates](https://www.hipaajournal.com/when-ai-technology-and-hipaa-collide/)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIPAA Compliance Requirements (2025)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ACCESS CONTROLS (§164.312(a)(1)) - REQUIRED                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ✓ Unique user identification for all system users                  │    │
│  │  ✓ Emergency access procedures documented and tested                │    │
│  │  ✓ Automatic logoff after 15 minutes of inactivity                 │    │
│  │  ✓ Encryption and decryption of ePHI                               │    │
│  │  ✓ Multi-factor authentication (NEW - 240-day deadline)            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  AUDIT CONTROLS (§164.312(b)) - REQUIRED                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ✓ Hardware, software, and procedural audit mechanisms              │    │
│  │  ✓ All AI inference requests logged with ePHI indicators           │    │
│  │  ✓ Minimum 6-year retention of audit logs                          │    │
│  │  ✓ Regular audit log review procedures                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  INTEGRITY CONTROLS (§164.312(c)(1)) - REQUIRED                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ✓ Mechanisms to authenticate ePHI                                  │    │
│  │  ✓ Digital signatures for AI-generated reports                      │    │
│  │  ✓ Checksums for model weights and configurations                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TRANSMISSION SECURITY (§164.312(e)(1)) - REQUIRED                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ✓ Integrity controls during transmission                           │    │
│  │  ✓ TLS 1.2+ (TLS 1.3 recommended) for all data in transit          │    │
│  │  ✓ End-to-end encryption for ePHI                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  AI-SPECIFIC REQUIREMENTS (2025 Updates)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ✓ Technology asset inventory including all AI systems              │    │
│  │  ✓ Network segmentation for AI workloads                            │    │
│  │  ✓ Vulnerability scanning and penetration testing                   │    │
│  │  ✓ Anti-malware on all AI infrastructure                            │    │
│  │  ✓ 72-hour incident notification to HHS                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Security Architecture Implementation

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
import jwt

class HIPAASecurityManager:
    """HIPAA-compliant security implementation"""
    
    def __init__(self, master_key: bytes, jwt_secret: str):
        self.cipher = Fernet(master_key)
        self.jwt_secret = jwt_secret
        self.session_timeout = timedelta(minutes=15)
        
    def encrypt_phi(self, data: str) -> str:
        """Encrypt PHI data at rest (AES-256 via Fernet)"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        """Decrypt PHI data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_user_token(
        self, 
        user_id: str, 
        roles: list[str],
        mfa_verified: bool = False
    ) -> str:
        """Generate JWT with MFA verification status"""
        payload = {
            "user_id": user_id,
            "roles": roles,
            "mfa_verified": mfa_verified,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.session_timeout
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def hash_for_audit(self, data: str) -> str:
        """Create SHA-256 hash for audit logging (non-reversible)"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def generate_encryption_key() -> bytes:
        """Generate a new Fernet encryption key"""
        return Fernet.generate_key()


class AuditLogger:
    """HIPAA-compliant audit logging"""
    
    def __init__(self, security_manager: HIPAASecurityManager):
        self.security = security_manager
        
    def log_access(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        phi_accessed: bool,
        outcome: str,
        details: Optional[dict] = None
    ):
        """Log PHI access event"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": secrets.token_hex(16),
            "user_id_hash": self.security.hash_for_audit(user_id),
            "action": action,
            "resource_type": resource_type,
            "resource_id_hash": self.security.hash_for_audit(resource_id),
            "phi_accessed": phi_accessed,
            "outcome": outcome,
            "details": details or {}
        }
        
        # In production: write to tamper-evident log store
        self._write_to_secure_log(audit_entry)
        
        return audit_entry["event_id"]
    
    def log_ai_inference(
        self,
        user_id: str,
        model_name: str,
        input_hash: str,
        output_hash: str,
        confidence: float,
        escalated: bool,
        processing_time_ms: float
    ):
        """Log AI inference event for compliance"""
        return self.log_access(
            user_id=user_id,
            action="ai_inference",
            resource_type="model",
            resource_id=model_name,
            phi_accessed=True,  # Assume PHI in medical AI
            outcome="success",
            details={
                "input_hash": input_hash,
                "output_hash": output_hash,
                "confidence": confidence,
                "escalated_to_human": escalated,
                "processing_time_ms": processing_time_ms
            }
        )
    
    def _write_to_secure_log(self, entry: dict):
        """Write to tamper-evident audit log"""
        # Implementation: Elasticsearch with immutable indices,
        # or append-only blockchain-style log
        pass
```

### 8.3 Data Encryption Configuration

```yaml
# encryption-config.yaml
encryption:
  at_rest:
    algorithm: AES-256-GCM
    key_management: HashiCorp Vault
    key_rotation_days: 90
    
  in_transit:
    protocol: TLS 1.3
    cipher_suites:
      - TLS_AES_256_GCM_SHA384
      - TLS_CHACHA20_POLY1305_SHA256
    certificate_rotation_days: 365
    
  database:
    postgres:
      ssl_mode: verify-full
      encryption: scram-sha-256
    redis:
      tls_enabled: true
      
  storage:
    minio:
      server_side_encryption: SSE-S3
      encryption_key_algorithm: AES256

access_control:
  authentication:
    methods:
      - saml_sso
      - mfa_totp
    session_timeout_minutes: 15
    max_failed_attempts: 5
    lockout_duration_minutes: 30
    
  authorization:
    model: RBAC
    roles:
      - name: radiologist
        permissions: [read_images, view_ai_analysis, approve_reports]
      - name: nurse
        permissions: [view_patient_info, use_communication_agent]
      - name: admin
        permissions: [manage_users, view_audit_logs, configure_system]
      - name: ai_system
        permissions: [process_phi, generate_reports, log_activity]
```

### 8.4 Business Associate Agreement Checklist

> **Reference:** [BAA Requirements](https://www.legalontech.com/contracts/business-associate-agreement-baa)

```markdown
## BAA Requirements for AI Vendors

### Required Clauses
- [ ] Definition of permitted uses/disclosures of PHI
- [ ] Prohibition on unauthorized use or disclosure
- [ ] Safeguards requirement (administrative, physical, technical)
- [ ] Reporting obligations for security incidents
- [ ] Subcontractor requirements (flow-down provisions)
- [ ] Access to PHI for covered entity audits
- [ ] Return/destruction of PHI upon termination
- [ ] Breach notification requirements (within 60 days)

### AI-Specific Additions
- [ ] Prohibition on using PHI for model training without consent
- [ ] Transparency requirements for AI decision-making
- [ ] Model audit rights for bias and accuracy
- [ ] Data localization requirements (on-premises)
- [ ] Encryption specifications (AES-256, TLS 1.2+)
- [ ] Multi-factor authentication requirements
- [ ] Incident response SLAs

### Vendor Certifications to Request
- [ ] HITRUST CSF certification
- [ ] SOC 2 Type II report
- [ ] ISO 27001 certification
- [ ] HIPAA compliance attestation
```

---

## 9. Data Flow Diagrams

### 9.1 Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MedAI Compass - Complete Data Flow                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Patient    │     │  Clinician   │     │    PACS      │
│   Portal     │     │  Workstation │     │   Server     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ HTTPS/TLS 1.3      │ HTTPS/TLS 1.3      │ DICOM TLS
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER (HAProxy)                              │
│                    SSL Termination │ Rate Limiting                           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (Kong)                                   │
│              Authentication │ Authorization │ Request Routing                │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Validate JWT token                                                       │
│  2. Check MFA verification status                                            │
│  3. Apply rate limits per user/role                                          │
│  4. Route to appropriate service                                             │
│  5. Log all requests to audit trail                                          │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│  DIAGNOSTIC SERVICE   │   │   WORKFLOW SERVICE    │   │ COMMUNICATION SERVICE │
├───────────────────────┤   ├───────────────────────┤   ├───────────────────────┤
│                       │   │                       │   │                       │
│ 1. Receive DICOM/     │   │ 1. Receive EHR data   │   │ 1. Receive patient    │
│    image request      │   │    (FHIR R4)          │   │    message            │
│                       │   │                       │   │                       │
│ 2. Input guardrails:  │   │ 2. Input guardrails:  │   │ 2. Input guardrails:  │
│    - PHI detection    │   │    - PHI masking      │   │    - Safety check     │
│    - Scope validation │   │    - Scope validation │   │    - Jailbreak detect │
│                       │   │                       │   │                       │
│ 3. Preprocess image   │   │ 3. Parse FHIR bundle  │   │ 3. Classify intent    │
│    - DICOM parsing    │   │    - Extract context  │   │    - Triage/educate   │
│    - Normalization    │   │                       │   │                       │
│                       │   │                       │   │                       │
│ 4. Model inference    │   │ 4. Model inference    │   │ 4. Model inference    │
│    - MedGemma 1.5 4B  │   │    - MedGemma 27B     │   │    - MedGemma 4B      │
│    - Path Foundation  │   │    - MedASR           │   │                       │
│                       │   │                       │   │                       │
│ 5. Output guardrails: │   │ 5. Output guardrails: │   │ 5. Output guardrails: │
│    - Confidence check │   │    - Terminology val  │   │    - Medical accuracy │
│    - Hallucination    │   │    - Completeness     │   │    - Harm potential   │
│                       │   │                       │   │                       │
│ 6. Human escalation   │   │ 6. Generate document  │   │ 6. Add disclaimer     │
│    decision           │   │                       │   │                       │
│                       │   │                       │   │                       │
└───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
            │                           │                           │
            └─────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED SERVICES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   FHIR CLIENT   │  │  GUARDRAILS SVC │  │   AUDIT LOGGER  │              │
│  │   (HAPI FHIR)   │  │   (NeMo)        │  │   (HIPAA)       │              │
│  │                 │  │                 │  │                 │              │
│  │ - Patient ctx   │  │ - PHI detection │  │ - Access logs   │              │
│  │ - Medications   │  │ - Jailbreak     │  │ - Inference log │              │
│  │ - Allergies     │  │ - Hallucination │  │ - Error logs    │              │
│  │ - Conditions    │  │ - Confidence    │  │ - 6-year retain │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  DRUG CHECKER   │  │   MEDICAL KB    │  │   EMBEDDING     │              │
│  │  (DrugBank API) │  │   (PubMed RAG)  │  │   STORE         │              │
│  │                 │  │                 │  │   (pgvector)    │              │
│  │ - Interactions  │  │ - Evidence      │  │                 │              │
│  │ - Contraindic.  │  │ - Guidelines    │  │ - Semantic      │              │
│  │ - Dosing        │  │ - Citations     │  │   search        │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL SERVING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TRITON INFERENCE SERVER                           │    │
│  │              Ensemble │ Dynamic Batching │ Multi-Model               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ MedGemma 4B  │  │ MedGemma 27B │  │ Path Found.  │  │ CXR Found.   │    │
│  │   (vLLM)     │  │   (vLLM)     │  │   (ONNX)     │  │   (ONNX)     │    │
│  │  A100 GPU 0  │  │ A100 GPU 1-3 │  │  A100 GPU 4  │  │  A100 GPU 5  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    ENCRYPTED STORAGE (AES-256)                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  PostgreSQL  │  │    Redis     │  │    MinIO     │  │Elasticsearch │    │
│  │              │  │              │  │              │  │              │    │
│  │ - Sessions   │  │ - Cache      │  │ - DICOM      │  │ - Audit logs │    │
│  │ - Checkpts   │  │ - Queues     │  │ - Reports    │  │ - Search     │    │
│  │ - Users      │  │ - Rate limit │  │ - WSI tiles  │  │ - Analytics  │    │
│  │              │  │              │  │              │  │              │    │
│  │ SSL: verify  │  │ TLS: enabled │  │ SSE-S3      │  │ TLS: enabled │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Diagnostic Imaging Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Diagnostic Imaging Pipeline Flow                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   PACS       │
│   Server     │
└──────┬───────┘
       │ DICOM TLS
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 1: IMAGE INGESTION                                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  • Parse DICOM headers (patient ID, study info, modality)                │
│  • Extract pixel data (Hounsfield units for CT, signal for MRI)          │
│  • Validate image quality (resolution, artifacts)                         │
│  • Log ingestion event (audit trail)                                      │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 2: PREPROCESSING                                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  • Resize to model input size (896×896 for MedGemma)                     │
│  • Normalize pixel values (model-specific normalization)                  │
│  • Window/level adjustment (for CT: lung/bone/soft tissue windows)       │
│  • For 3D: slice selection or volume preparation                          │
│  • For WSI: tile extraction at multiple magnifications                    │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 3: CONTEXT RETRIEVAL                                                │
├──────────────────────────────────────────────────────────────────────────┤
│  • Query FHIR server for patient context                                  │
│    - Active medications (drug interaction check)                          │
│    - Known allergies                                                      │
│    - Relevant medical history                                             │
│    - Prior imaging studies (for comparison)                               │
│  • Retrieve clinical indication from order                                │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 4: MODEL INFERENCE (MedGemma 1.5 4B)                               │
├──────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ <image>[preprocessed_image]</image>                              │     │
│  │ Clinical indication: [indication]                                │     │
│  │ Patient context: [relevant_history]                              │     │
│  │ Prior studies: [comparison_notes]                                │     │
│  │                                                                  │     │
│  │ Please analyze this [modality] image and provide:                │     │
│  │ 1. Key findings with anatomical localization                     │     │
│  │ 2. Differential diagnosis                                        │     │
│  │ 3. Comparison with prior studies if available                    │     │
│  │ 4. Recommendations for follow-up                                 │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                           │
│  PROCESSING:                                                              │
│  • MedSigLIP vision encoder → image embeddings                           │
│  • Gemma 3 language model → structured analysis                          │
│  • Monte Carlo dropout → uncertainty estimation                           │
│                                                                           │
│  OUTPUT:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ {                                                                │     │
│  │   "findings": [...],                                             │     │
│  │   "differential": [...],                                         │     │
│  │   "bounding_boxes": [...],  // anatomical localization           │     │
│  │   "comparison": "...",                                           │     │
│  │   "recommendations": [...],                                      │     │
│  │   "confidence": 0.92,                                            │     │
│  │   "uncertainty": 0.08                                            │     │
│  │ }                                                                │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 5: OUTPUT GUARDRAILS                                                │
├──────────────────────────────────────────────────────────────────────────┤
│  • Medical terminology validation (SNOMED, RadLex)                        │
│  • Hallucination detection (cross-reference with image content)          │
│  • Critical finding detection (immediate escalation triggers)             │
│  • Confidence threshold check (≥0.90 for diagnostic)                     │
│  • Generate clinical disclaimer                                           │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 6: ESCALATION DECISION                                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────┐     ┌─────────────────────┐                     │
│  │ CONFIDENCE ≥ 0.95   │ NO  │ CONFIDENCE 0.80-0.95│                     │
│  │ No critical finding │────▶│ OR high uncertainty │                     │
│  └──────────┬──────────┘     └──────────┬──────────┘                     │
│             │ YES                       │ YES                             │
│             ▼                           ▼                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐                     │
│  │   AUTO-APPROVE      │     │   SOFT REVIEW       │                     │
│  │   (with logging)    │     │   (flag for audit)  │                     │
│  └─────────────────────┘     └─────────────────────┘                     │
│                                         │                                 │
│                                         │ CONFIDENCE < 0.80              │
│                                         │ OR critical finding             │
│                                         ▼                                 │
│                              ┌─────────────────────┐                     │
│                              │   HUMAN REVIEW      │                     │
│                              │   (radiologist)     │                     │
│                              └─────────────────────┘                     │
│                                                                           │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 7: REPORT GENERATION                                                │
├──────────────────────────────────────────────────────────────────────────┤
│  • Structure findings in standard radiology report format                 │
│  • Include AI-generated analysis with confidence scores                   │
│  • Add clinical disclaimer and human review status                        │
│  • Generate DICOM SR (Structured Report) for PACS integration             │
│  • Log final report to audit trail                                        │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 8: OUTPUT DELIVERY                                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  • Store report in MinIO (encrypted)                                      │
│  • Push DICOM SR to PACS                                                  │
│  • Update EHR via FHIR (DiagnosticReport resource)                       │
│  • Notify ordering physician                                              │
│  • Update patient portal (if enabled)                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Implementation Roadmap

### 10.1 Project Timeline (12 Weeks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MedAI Compass Implementation Timeline                     │
│                    Competition Deadline: February 24, 2026                   │
└─────────────────────────────────────────────────────────────────────────────┘

WEEK 1-2: FOUNDATION
├── Day 1-3: Environment Setup
│   ├── [ ] Set up development GPU server (A100 or RTX 4090)
│   ├── [ ] Install CUDA, Docker, Kubernetes (kind for local)
│   ├── [ ] Configure HuggingFace access for HAI-DEF models
│   └── [ ] Set up version control (GitHub) and CI/CD (GitHub Actions)
│
├── Day 4-7: Model Exploration
│   ├── [ ] Run official MedGemma notebooks
│   │       - Quick start: https://developers.google.com/health-ai-developer-foundations/medgemma/quickstart
│   │       - Fine-tuning: https://developers.google.com/health-ai-developer-foundations/medgemma/finetune
│   │       - 3D imaging: https://developers.google.com/health-ai-developer-foundations/medgemma/3d
│   ├── [ ] Benchmark inference performance (latency, throughput)
│   ├── [ ] Test vLLM serving configuration
│   └── [ ] Document model capabilities and limitations
│
├── Day 8-10: Dataset Access
│   ├── [ ] Submit PhysioNet credentialing application (MIMIC-IV)
│   ├── [ ] Download open datasets (ChestX-ray14, CAMELYON16)
│   ├── [ ] Set up Synthea for synthetic EHR generation
│   └── [ ] Create data loading pipelines
│
└── Day 11-14: Architecture Design
    ├── [ ] Finalize multi-agent architecture document
    ├── [ ] Design API contracts between services
    ├── [ ] Create database schema for sessions and audit logs
    └── [ ] Set up project structure and coding standards

───────────────────────────────────────────────────────────────────────────────

WEEK 3-4: DIAGNOSTIC AGENT
├── [x] Implement LangGraph diagnostic workflow
│   ├── [x] Image preprocessing node
│   ├── [x] MedGemma inference node
│   ├── [x] Pathology analysis node (Path Foundation)
│   ├── [x] Report generation node
│   └── [x] Confidence check and routing
│
├── [x] Integrate CXR Foundation for chest X-ray analysis
├── [x] Implement 3D CT/MRI processing (MedGemma 1.5 feature)
├── [x] Add bounding box localization for findings
├── [x] Create evaluation pipeline for diagnostic metrics
└── [x] Unit tests for diagnostic pipeline

───────────────────────────────────────────────────────────────────────────────

WEEK 5-6: WORKFLOW AGENT
├── [x] Implement CrewAI workflow agents
│   ├── [x] Scheduler agent
│   ├── [x] Documentation agent (discharge summaries)
│   ├── [x] Prior authorization agent
│   └── [x] Agent coordination logic
│
├── [x] Integrate MedGemma 27B for complex documentation
├── [x] Implement FHIR R4 client for EHR integration
├── [x] Create clinical note summarization pipeline
├── [x] Add MedASR integration for dictation (stretch goal)
└── [x] Unit tests for workflow pipeline

───────────────────────────────────────────────────────────────────────────────

WEEK 7-8: COMMUNICATION AGENT
├── [x] Implement AutoGen communication agents
│   ├── [x] Triage agent
│   ├── [x] Health education agent
│   ├── [x] Follow-up scheduling agent
│   └── [x] Clinical oversight proxy
│
├── [ ] Build patient-facing chat interface (React/Next.js)
├── [x] Implement conversation history and context management
├── [x] Add multi-language support (stretch goal)
├── [x] Create patient communication evaluation metrics
└── [x] Unit tests for communication pipeline

───────────────────────────────────────────────────────────────────────────────

WEEK 9-10: SAFETY & INTEGRATION
├── [x] Deploy NeMo Guardrails
│   ├── [x] Configure medical-specific rails
│   ├── [x] Implement PHI detection and masking
│   ├── [x] Add jailbreak detection
│   └── [x] Implement hallucination detection
│
├── [x] Implement uncertainty quantification (MC Dropout)
├── [x] Build human escalation gateway
│   ├── [x] Critical finding detection
│   ├── [x] Safety concern triggers
│   ├── [ ] Clinician review UI
│   └── [x] Escalation routing logic
│
├── [x] Integrate master orchestrator
│   ├── [x] Intent classification
│   ├── [x] Cross-domain routing
│   └── [x] Response aggregation
│
├── [x] Comprehensive HIPAA audit logging
└── [x] Integration tests across all agents

───────────────────────────────────────────────────────────────────────────────

WEEK 11: PRODUCTION HARDENING
├── [x] Containerization with Docker
│   ├── [x] HIPAA-compliant configurations
│   ├── [x] Security scanning (Trivy)
│   └── [x] Multi-stage builds for optimization
│
├── [x] Deploy Triton Inference Server
│   ├── [x] Model repository setup
│   ├── [x] Ensemble pipeline configuration
│   └── [x] Performance optimization
│
├── [x] Implement observability stack
│   ├── [x] Prometheus metrics
│   ├── [x] Grafana dashboards
│   ├── [x] Alert rules
│   └── [x] Log aggregation (ELK)
│
├── [x] Security assessment
│   ├── [ ] Penetration testing
│   ├── [x] HIPAA compliance checklist
│   └── [x] Vulnerability scanning
│
└── [x] Load testing and optimization

───────────────────────────────────────────────────────────────────────────────

WEEK 12: COMPETITION SUBMISSION
├── Day 1-3: Demo Video Production
│   ├── [ ] Write script (3-minute max)
│   │       - 30 sec: Problem statement + elevator pitch
│   │       - 2 min: Live demo of all three domains
│   │       - 30 sec: Technical highlights + impact
│   ├── [ ] Record screen captures with voiceover
│   ├── [ ] Edit and polish video
│   └── [ ] Add captions and branding
│
├── Day 4-5: Technical Write-up
│   ├── [ ] Architecture documentation (3 pages max)
│   ├── [ ] Model usage and fine-tuning details
│   ├── [ ] Performance analysis and benchmarks
│   ├── [ ] Deployment and scalability discussion
│   └── [ ] Limitations and future work
│
├── Day 6: Code Repository
│   ├── [ ] Clean up codebase
│   ├── [ ] Write comprehensive README
│   ├── [ ] Add inline documentation
│   ├── [ ] Create reproducibility instructions
│   └── [ ] Add license (Apache 2.0)
│
└── Day 7: Final Submission
    ├── [ ] Final review of all materials
    ├── [ ] Submit to Kaggle before February 24, 2026 11:59 PM UTC
    └── [ ] Backup submission confirmation

───────────────────────────────────────────────────────────────────────────────
```

### 10.2 Key Milestones

| Milestone | Date | Deliverables |
|-----------|------|--------------|
| **M1: Foundation Complete** | Week 2 | Dev environment, model access, architecture design |
| **M2: Diagnostic MVP** | Week 4 | Working diagnostic pipeline with CXR analysis |
| **M3: Workflow MVP** | Week 6 | Working documentation generation from clinical notes |
| **M4: Communication MVP** | Week 8 | Patient-facing chat with health education |
| **M5: Integrated System** | Week 10 | All agents integrated with guardrails and orchestration |
| **M6: Production Ready** | Week 11 | Containerized, monitored, security-tested system |
| **M7: Competition Submission** | Week 12 | Video, write-up, code repository submitted |

---

## 11. Competition Submission Strategy

### 11.1 Evaluation Criteria Mapping

| Criterion | Weight | Our Strategy | Evidence |
|-----------|--------|--------------|----------|
| **Effective use of HAI-DEF** | 20% | Use MedGemma 1.5 4B (3D imaging, WSI, longitudinal), Path Foundation, CXR Foundation, MedASR | Multi-model integration, unique 1.5 features |
| **Problem Domain** | 15% | Target radiology workflow bottleneck: reporting delays impact patient care | WHO statistics on diagnostic imaging gaps |
| **Impact Potential** | 15% | Quantify: 40% reduction in report turnaround, 25% reduction in missed findings | Cite published radiologist productivity studies |
| **Product Feasibility** | 20% | Production-grade architecture, HIPAA compliance, on-premises deployment | Technical documentation, deployment guide |
| **Execution & Communication** | 30% | Polished demo video, clean code, comprehensive write-up | Professional production quality |

### 11.2 Demo Video Script Outline

```markdown
## MedAI Compass Demo Script (3 minutes)

### Opening Hook (0:00 - 0:30)
- Problem: "Every day, radiologists face a backlog of 50+ studies. 
  Critical findings get delayed. MedAI Compass changes that."
- Quick montage: Overwhelmed radiologist → MedAI Compass dashboard → happy outcome

### Live Demo: Diagnostic Agent (0:30 - 1:30)
- Upload chest X-ray (MIMIC-CXR sample)
- Show real-time MedGemma 1.5 analysis
- Highlight: Anatomical localization with bounding boxes
- Show: Confidence score and uncertainty visualization
- Demo: Human escalation trigger for low-confidence finding

### Live Demo: Workflow Agent (1:30 - 2:00)
- Input: Sample discharge note (MIMIC-III)
- Show: Auto-generated structured summary
- Highlight: FHIR integration for EHR update
- Demo: Prior authorization draft generation

### Live Demo: Communication Agent (2:00 - 2:20)
- Patient portal chat interface
- Sample query: "What does my X-ray result mean?"
- Show: Patient-friendly explanation with safety guardrails
- Highlight: Appropriate referral to clinician for follow-up

### Technical Highlights (2:20 - 2:45)
- Architecture diagram overlay
- Key differentiators:
  - MedGemma 1.5 multimodal capabilities
  - Multi-agent orchestration with LangGraph
  - Production-grade safety guardrails
  - HIPAA-compliant on-premises deployment

### Impact Statement (2:45 - 3:00)
- "MedAI Compass: Transforming radiology workflow with responsible AI"
- Call to action: "Check out our code on GitHub"
- End screen: Logo, team name, links
```

### 11.3 Technical Write-up Template

```markdown
# MedAI Compass: Production-Grade Multi-Agent Medical AI

## 1. Problem Statement (0.5 page)
- Radiology workflow challenges and impact on patient care
- Why AI is the right solution
- Target users and their improved journey

## 2. Solution Architecture (1 page)
- Multi-agent system overview
- HAI-DEF model selection and rationale
- Data flow and integration points

## 3. HAI-DEF Model Usage (0.75 page)
- MedGemma 1.5 4B: Diagnostic imaging (3D, WSI, longitudinal)
- MedGemma 27B: Complex documentation
- Path Foundation: Histopathology embeddings
- CXR Foundation: Chest X-ray classification
- Fine-tuning approach (QLoRA)

## 4. Safety and Compliance (0.5 page)
- Guardrails implementation (NeMo)
- Uncertainty quantification
- Human-in-the-loop design
- HIPAA compliance approach

## 5. Results and Analysis (0.25 page)
- Evaluation metrics and benchmarks
- Limitations and future work

## References
- Links to code repository, demo video, datasets used
```

---

## 12. References & Resources

### 12.1 Official HAI-DEF Resources

| Resource | Link |
|----------|------|
| **HAI-DEF Main Site** | https://developers.google.com/health-ai-developer-foundations |
| **MedGemma Documentation** | https://developers.google.com/health-ai-developer-foundations/medgemma |
| **MedGemma 1.5 Model Card** | https://developers.google.com/health-ai-developer-foundations/medgemma/model-card |
| **HAI-DEF HuggingFace Collection** | https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def |
| **MedGemma HuggingFace Collection** | https://huggingface.co/collections/google/medgemma-release |
| **HAI-DEF Concept Apps** | https://huggingface.co/collections/google/hai-def-concept-apps |
| **HAI-DEF Developer Forum** | https://discuss.ai.google.dev/c/hai-def/62 |
| **HAI-DEF FAQs** | https://developers.google.com/health-ai-developer-foundations/faqs |
| **TxGemma Documentation** | https://developers.google.com/health-ai-developer-foundations/txgemma |

### 12.2 Model Links

| Model | HuggingFace Link |
|-------|------------------|
| **MedGemma 4B IT** | https://huggingface.co/google/medgemma-4b-it |
| **MedGemma 27B IT** | https://huggingface.co/google/medgemma-27b-it |
| **TxGemma 2B** | https://huggingface.co/google/txgemma-2b |
| **TxGemma 9B** | https://huggingface.co/google/txgemma-9b |
| **TxGemma 27B** | https://huggingface.co/google/txgemma-27b |

### 12.3 Dataset Links

| Dataset | Link |
|---------|------|
| **MIMIC-IV** | https://physionet.org/content/mimiciv/3.1/ |
| **MIMIC-CXR** | https://physionet.org/content/mimic-cxr/ |
| **ChestX-ray14** | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| **CAMELYON16** | https://camelyon16.grand-challenge.org/ |
| **LIDC-IDRI** | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| **n2c2 Datasets** | https://n2c2.dbmi.hms.harvard.edu/data-sets |
| **Synthea** | https://synthetichealth.github.io/synthea/ |
| **MedQuAD** | https://github.com/abachaa/MedQuAD |
| **MedDialog** | https://github.com/UCSD-AI4H/Medical-Dialogue-System |

### 12.4 Framework Documentation

| Framework | Link |
|-----------|------|
| **LangGraph** | https://docs.langchain.com/oss/python/langgraph |
| **CrewAI** | https://docs.crewai.com/ |
| **AutoGen** | https://microsoft.github.io/autogen/ |
| **NeMo Guardrails** | https://github.com/NVIDIA/NeMo-Guardrails |
| **vLLM** | https://docs.vllm.ai/ |
| **Triton Inference Server** | https://docs.nvidia.com/deeplearning/triton-inference-server/ |

### 12.5 Compliance & Security

| Resource | Link |
|----------|------|
| **HIPAA Security Rule** | https://www.hhs.gov/hipaa/for-professionals/security/ |
| **HIPAA AI Guidance** | https://www.hipaajournal.com/when-ai-technology-and-hipaa-collide/ |
| **BAA Requirements** | https://www.legalontech.com/contracts/business-associate-agreement-baa |

### 12.6 Research Papers

| Topic | Reference |
|-------|-----------|
| **MedGemma Research** | Google Research Blog: https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/ |
| **MedGemma 1.5 & MedASR** | https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/ |
| **TxGemma Research** | https://developers.googleblog.com/en/introducing-txgemma-open-models-improving-therapeutics-development/ |
| **Medical AI Fairness** | https://pmc.ncbi.nlm.nih.gov/articles/PMC10632090/ |

### 12.7 Competition Resources

| Resource | Link |
|----------|------|
| **MedGemma Impact Challenge** | https://www.kaggle.com/competitions/medgemma-impact-challenge |
| **Demo Video Best Practices** | https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video |
| **Video Making Guide** | https://help.devpost.com/article/84-video-making-best-practices |

---

## Appendix A: Quick Start Commands

```bash
# Clone the repository
git clone https://github.com/yourusername/medai-compass.git
cd medai-compass

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download MedGemma 4B
huggingface-cli login
huggingface-cli download google/medgemma-4b-it --local-dir ./models/medgemma-4b-it

# Start the development stack
docker-compose up -d postgres redis minio

# Run the diagnostic agent
python -m medai_compass.agents.diagnostic --config config/dev.yaml

# Run tests
pytest tests/ -v --cov=medai_compass
```

---

## Appendix B: Project Structure

```
medai-compass/
├── README.md
├── LICENSE
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── config/
│   ├── dev.yaml
│   ├── prod.yaml
│   ├── guardrails.yaml
│   └── triton/
│       └── config.pbtxt
│
├── medai_compass/
│   ├── __init__.py
│   ├── orchestrator.py          # Master orchestrator
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── diagnostic/          # LangGraph diagnostic agent
│   │   │   ├── __init__.py
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   └── state.py
│   │   ├── workflow/            # CrewAI workflow agent
│   │   │   ├── __init__.py
│   │   │   ├── agents.py
│   │   │   ├── tasks.py
│   │   │   └── crew.py
│   │   └── communication/       # AutoGen communication agent
│   │       ├── __init__.py
│   │       ├── agents.py
│   │       └── manager.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── medgemma.py          # MedGemma wrapper
│   │   ├── path_foundation.py   # Path Foundation wrapper
│   │   └── cxr_foundation.py    # CXR Foundation wrapper
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_rails.py
│   │   ├── output_rails.py
│   │   └── escalation.py
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── encryption.py
│   │   ├── audit.py
│   │   └── hipaa.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── drift.py
│   │   └── benchmarks.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── dicom.py
│       ├── fhir.py
│       └── logging.py
│
├── tests/
│   ├── __init__.py
│   ├── test_diagnostic.py
│   ├── test_workflow.py
│   ├── test_communication.py
│   └── test_guardrails.py
│
├── notebooks/
│   ├── 01_medgemma_quickstart.ipynb
│   ├── 02_diagnostic_pipeline.ipynb
│   ├── 03_workflow_automation.ipynb
│   └── 04_evaluation.ipynb
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_models.sh
│   └── run_benchmarks.py
│
└── docs/
    ├── architecture.md
    ├── deployment.md
    └── api.md
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Author:** MedAI Compass Team  
**License:** Apache 2.0