"""
Fine Tuning - Detailed Breakdown
============================================

[Optional: longer overview paragraph you can fill in later]
"""

import os
import re
import sys
import subprocess
from pathlib import Path
import base64
import os


TOPIC_NAME = "Fine Tuning_Detailed Breakdown"

# ─────────────────────────────────────────────────────────────────────────────
# PATH TO THE PIPELINE SCRIPT
# Adjust this to match your actual project layout
# ─────────────────────────────────────────────────────────────────────────────

# This resolves relative to this file's location:
#   topics/08_a_FineTuning_FullFineTuning.py
#   Implementation/Full_Fine_Tuning_Implementation/scripts/Full_fine_tuning_main.py

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "Implementation" / "Full_Fine_Tuning_Implementation" / "scripts"
_MAIN_SCRIPT = _SCRIPTS_DIR / "Full_fine_tuning_main.py"


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPER — converts local images to base64 HTML for st.markdown()
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_html(path, alt="", width="100%"):
    """Convert a local image file to an HTML <img> tag with base64 data.
    This allows images to render inside st.markdown() with unsafe_allow_html=True.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(ext, "image/png")
        return f'<img src="data:{mime};base64,{b64}" alt="{alt}" style="width:{width}; border-radius:8px; margin:12px 0;">'
    return f'<p style="color:red;">️ Image not found: {path}</p>'

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

### Fine Tuning Detailed Breakdown
            
                                                                            FINE-TUNING METHODS HIERARCHY — LANDSCAPE VIEW
                                                
                                                ══════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        
                                                                                            ┌──────────────────────┐
                                                                                            │     FINE-TUNING      │
                                                                                            │  (Adapting a model   │
                                                                                            │   to a specific task)│
                                                                                            └──────────┬───────────┘
                                                                                                       │
                                  ┌────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┐
                                  │                                                                    │                                                                    │
                                  ▼                                                                    ▼                                                                    ▼
                  ┌───────────────────────────────┐                              ┌────────────────────────────────────────┐                             ┌─────────────────────────────────────┐
                  │       FULL FINE-TUNING        │                              │    PEFT (Parameter-Efficient           │                             │       ALIGNMENT TUNING              │
                  │                               │                              │    Fine-Tuning)                        │                             │    (Human Preference-Based)         │
                  │  • ALL params updated         │                              │                                        │                             │                                     │
                  │  • Best quality potential     │                              │  • Only a SUBSET of params updated     │                             │  • Aligns model behavior            │
                  │  • Highest cost (GPU/memory)  │                              │  • Lower cost (memory & compute)       │                             │    with human values                │
                  │  • Risk of catastrophic       │                              │  • Preserves pre-trained knowledge     │                             │  • Uses ranked preferences          │
                  │    forgetting                 │                              │  • Modular (swap adapters per task)    │                             │    or reward signals                │
                  └───────────┬───────────────────┘                              └──────────────────────┬─────────────────┘                             └─────────────────────┬───────────────┘
                              │                                                                         │                                                                     │
              ┌───────────────┼───────────────┐                                                         │                                                                     │
              ▼               ▼               ▼                                                         │                                                                     │
      ┌──────────────┐┌──────────────┐┌──────────────┐                                                  │                                                                     │
      │  Standard    ││  Feature     ││  Gradual     │                                                  │                                                                     │
      │  Full FT     ││  Extraction  ││  Unfreezing  │                                                  │                                                                     │
      │              ││              ││              │                                                  │                                                                     │
      │ All layers   ││ Freeze base, ││ Unfreeze     │                                                  │                                                                     │
      │ unlocked     ││ train new    ││ layers one   │                                                  │                                                                     │
      │ from start   ││ head only    ││ by one       │                                                  │                                                                     │
      └──────────────┘└──────────────┘└──────────────┘                                                  │                                                                     │
                                                                                                        │                                                                     │
                                                                                                        │                                                                     │
                    ┌──────────────────────────────┬──────────────────────────────┬─────────────────────┼──────────────────┬──────────────────────────────┐                   │
                    │                              │                              │                     │                  │                              │                   │
                    ▼                              ▼                              ▼                     │                  ▼                              ▼                   │
        ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐       │      ┌───────────────────────────┐ ┌───────────────────────────┐    │
        │    ADDITIVE METHODS       │ │   REPARAMETERIZATION      │ │    SELECTIVE METHODS      │       │      │     HYBRID METHODS        │ │     PROMPT METHODS        │    │
        │                           │ │                           │ │                           │       │      │                           │ │                           │    │
        │  Add NEW parameters       │ │  Transform existing       │ │  Select WHICH existing    │       │      │  Combine multiple PEFT    │ │  Learn soft prompts,      │    │
        │  to the model while       │ │  params via low-rank      │ │  params to train and      │       │      │  strategies (e.g.         │ │  NOT weights. Trainable   │    │
        │  freezing originals       │ │  decomposition            │ │  freeze the rest          │       │      │  quantization + adapters) │ │  tokens prepended to input│    │
        └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘       │      └─────────────┬─────────────┘ └──────────────┬────────────┘    │
                      │                             │                             │                     │                    │                              │                 │
            ┌─────────┼─────────┐         ┌─────────┼─────────┐          ┌────────┼────────┐            │          ┌─────────┼─────────┐          ┌─────────┼─────────┐       │
            ▼         ▼         ▼         ▼         ▼         ▼          ▼        ▼        ▼            │          ▼         ▼         ▼          ▼         ▼         ▼       │
       ┌──────────┐┌────────┐┌──────┐ ┌─────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐     │     ┌─────────┐┌────────┐┌────────┐┌─────────┐┌────────┐┌────────┐  │
       │Bottleneck││ Soft   ││ IA³  │ │  LoRA   ││  DoRA  ││ LoRA+  ││ rsLoRA ││BitFit  ││Fish    │     │     │ QLoRA   ││LongLoRA││LoRA-FA ││ Prefix  ││P-Tuning││Prompt  │  │
       │Adapters  ││Prompts ││      │ │         ││        ││        ││        ││        ││ Mask   │     │     │         ││        ││        ││ Tuning  ││  v2    ││Tuning  │  │
       │          ││        ││      │ │         ││        ││        ││        ││        ││        │     │     │         ││        ││        ││         ││        ││        │  │
       │Small FFN ││Learned ││Learns│ │W + A×B  ││Weight- ││Diff LR ││Rank    ││Train   ││Learned │ Freeze/   │4-bit    ││Shifted ││Frozen  ││Learned  ││Deep    ││Learned │  │
       │modules   ││vectors ││resca-│ │Low-rank ││Decomp- ││for A & ││stabili-││ONLY    ││binary  │  Thaw     │base +   ││sparse  ││activa- ││vectors  ││prompts ││tokens  │  │
       │inserted  ││added to││ling  │ │adapters ││osed    ││B matri-││zation  ││bias    ││masks on│           │16-bit   ││atten-  ││tion    ││prepended││across  ││at input│  │
       │between   ││input   ││vecto-│ │trainable││LoRA    ││ces     ││for high││terms   ││params  │ Select    │adapters ││tion    ││adapters││to K & V ││all     ││embed-  │  │
       │layers    ││embeddi-││rs    │ │         ││        ││        ││ranks   ││        ││        │ layers    │         ││        ││        ││at every ││layers  ││dings   │  │
       │          ││ngs     ││      │ │         ││        ││        ││        ││        ││        │ to train  │         ││        ││        ││layer    ││        ││only    │  │
       └──────────┘└────────┘└──────┘ └─────────┘└────────┘└────────┘└────────┘└────────┘└────────┘           └─────────┘└────────┘└────────┘└─────────┘└────────┘└────────┘  │
                                                                                                                                                                              │
                                                                                                                                                                              │
            ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════                               │
                                                                ALIGNMENT TUNING (DETAIL)                                                                                     │
            ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════                               │  
                                                                                                                                                                              │
                                                                                                                                                                              │
                                                                                                                                                                              │  
                                                                                                                                                                              │
                                                                                                                                                                              │
                                                                                                                                                                              │  
                                                                                                                                                                              │  
                                                                                                                                                                              │  
                         ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                                                                                          │
    │     Step 1: SFT                          Step 2: Reward Modeling                       Step 3: RL Optimization                                           │
    │     ┌──────────────────────┐              ┌──────────────────────────┐                  ┌─────────────────────────────────────────────────────────┐      │
    │     │ Supervised           │              │ Human ranks outputs:     │                  │                                                         │      │
    │     │ Fine-Tuning on       │  ──────────▶ │                          │    ──────────▶   │   ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌────────┐│      │
    │     │ (prompt, ideal       │              │ Response A > B > C       │                  │   │  RLHF    │  │  DPO      │  │  ORPO     │  │  KTO   ││      │
    │     │  response) pairs     │              │ Train reward model       │                  │   │  (PPO)   │  │           │  │           │  │        ││      │ 
    │     │                      │              │ on these preferences     │                  │   │          │  │ Direct    │  │ Combines  │  │ Only   ││      │
    │     └──────────────────────┘              └──────────────────────────┘                  │   │ Model    │  │ Preference│  │ SFT +     │  │ needs  ││      │
    │                                                                                         │   │ generates│  │ Optim.    │  │ preference│  │ thumbs ││      │
    │                                                                                         │   │ → reward │  │           │  │ in one    │  │ up/down││      │
    │                                                                                         │   │ scores → │  │ Skips     │  │ step      │  │        ││      │
    │                                                                                         │   │ PPO      │  │ reward    │  │           │  │ No     ││      │
    │                                                                                         │   │ updates  │  │ model     │  │           │  │ pairs  ││      │
    │                                                                                         │   └──────────┘  └───────────┘  └───────────┘  └────────┘│      │
    │                                                                                         └─────────────────────────────────────────────────────────┘      │
    │                                                                                                                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


---

##### LandScape Diagram of the fine-tuning process: Simplified 


        ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                    FINE-TUNING LANDSCAPE — HORIZONTAL VIEW 
        ════════════════════════════════════════════════════════════════════════════════════════════════════════════════


                                                                  ┌─────────────────────┐
                                                                  │   FOUNDATION MODEL  │
                                                                  │  (GPT, LLaMA, etc.) │
                                                                  └──────────┬──────────┘
                                                                             │
                         ┌───────────────────────────────────────────────────┼───────────────────────────────────┐
                         │                                                   │                                   │
                         ▼                                                   ▼                                   │
              ┌──────────────────────────┐                   ┌──────────────────────────────┐                    │
              │     FULL FINE-TUNING     │                   │   PEFT (Parameter-Efficient) │                    │   
              │   (all weights updated)  │                   │   (few weights updated)      │                    │
              └──────────┬───────────────┘                   └───────────────┬──────────────┘                    │
                         │                                                   │                                   │
            ┌────────────┼────────────┐                 ┌────────────────────┼───────────────────┐               │   
            ▼            ▼            ▼                 ▼                    ▼                   ▼               │
      ┌───────────┐┌──────────┐┌──────────┐         ┌──────────────┐  ┌──────────────┐  ┌───────────────┐        │
      │  Feature  ││ Gradual  ││ Standard │         │ ADAPTER-BASED│  │ SOFT PROMPT  │  │  SELECTIVE    │        │
      │Extraction ││Unfreezing││  (Full)  │         │              │  │   -BASED     │  │  / SPARSE     │        │
      └───────────┘└──────────┘└──────────┘         │              │  │              │  │               │        │
                                                    │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌───────────┐ │        │
                                                    │ │  LoRA    │ │  │ │ Prefix   │ │  │ │  BitFit   │ │        │   
                                                    │ │ W+A×B    │ │  │ │ Tuning   │ │  │ │(bias only)│ │        │
                                                    │ └────┬─────┘ │  │ └──────────┘ │  │ └───────────┘ │        │
                                                    │      │       │  │ ┌──────────┐ │  │ ┌───────────┐ │        │
                                                    │      ▼       │  │ │ Prompt   │ │  │ │ (IA)³     │ │        │
                                                    │ ┌──────────┐ │  │ │ Tuning   │ │  │ │(rescaling)│ │        │
                                                    │ │  QLoRA   │ │  │ └──────────┘ │  │ └───────────┘ │        │
                                                    │ │4-bit base│ │  │ ┌──────────┐ │  │ ┌──────────┐  │        │
                                                    │ │+16-bit   │ │  │ │ P-Tuning │ │  │ │   Diff   │  │        │   
                                                    │ │ adapters │ │  │ │    v2    │ │  │ │ Pruning  │  │        │
                                                    │ └──────────┘ │  │ └──────────┘ │  │ └──────────┘  │        │   
                                                    │ ┌──────────┐ │  └──────────────┘  └───────────────┘        │
                                                    │ │  DoRA    │ │                                             │
                                                    │ │ AdaLoRA  │ │                                             │
                                                    │ │ LoRA+    │ │                                             │   
                                                    │ └──────────┘ │                                             │   
                                                    └──────────────┘                                             │ 
                                                                                                                 │
                                                                                                                 │
                                                                                                                 │
                                                       ┌─────────────────────────────────────────────────────────┘ 
                                                       │
                                                       ▼                                                     
                                       ┌────────────────────────────────────┐
                                       │          ALIGNMENT TUNING          │
                                       │     (human preference-based)       │
                                       └────────────────────────────────────┘
                                                       │
                   ┌───────────────────────────────────┼────────────────────────────────┐   
                   ▼                                   ▼                                ▼
              ┌────────────┐                     ┌────────────┐                  ┌──────────────┐
              │    RLHF    │                     │    DPO     │                  │  ORPO / KTO  │
              └────────────┘                     └────────────┘                  └──────────────┘
                   │                                   │                                │
                   ▼                                   ▼                                ▼
            ┌───────────────────────────────────────────────────────────────────────────────────────┐
            │                              SFT  →  Reward Model  →  RL                              │
            │                                                                                       │                               
            │                              (or skip reward w/ DPO)                                  │
            │                              (or single-step w/ ORPO)                                 │
            └───────────────────────────────────────────────────────────────────────────────────────┘                                                             
                                                                                                            
---                                                                                                            
                                                                                                            
    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════   
                                LEARNING PARADIGMS  (Cross-Cutting — Apply to ANY method above)
    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    
    SUPERVISED                               UNSUPERVISED / SELF-SUPERVISED                   SEMI-SUPERVISED                           REINFORCEMENT LEARNING
    ┌──────────────────────────────┐         ┌──────────────────────────────────┐             ┌────────────────────────────────┐        ┌──────────────────────────────┐
    │ Labeled pairs:               │         │ Raw domain text:                 │             │ Small labeled set +            │        │ Human preferences:           │
    │ (input → desired output)     │         │ next-token prediction / MLM      │             │ large unlabeled set            │        │ (chosen vs rejected)         │
    │                              │         │                                  │             │                                │        │                              │
    │ Used in:                     │         │ Used in:                         │             │ Used in:                       │        │ Used in:                     │
    │ • SFT (instruction tuning)   │         │ • Continued pre-training         │             │ • Pseudo-labeling              │        │ • RLHF (PPO-based)           │
    │ • Classification, NER        │         │ • Domain adaptation              │             │ • Self-training                │        │ • DPO                        │
    │ • Summarization, translation │         │   (medical, legal, code corpora) │             │ • Co-training                  │        │ • ORPO, KTO                  │
    │ • Question answering         │         │ • Language model adaptation      │             │ • Teacher-student labeling     │        │ • RLAIF                      │
    └──────────────────────────────┘         └──────────────────────────────────┘             └────────────────────────────────┘        └──────────────────────────────┘
              │                                            │                                            │                                         │
              └────────────────────────────────────────────┴────────────────┬───────────────────────────┴─────────────────────────────────────────┘
                                                                            │
                                            These paradigms can be COMBINED with ANY fine-tuning method
                                            (e.g. QLoRA + SFT,  Full Fine Tuning + domain adaptation,  LoRA + DPO)


---

    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                        TYPICAL END-TO-END PRODUCTION PIPELINE
    ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    
    
      ┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐       ┌──────────────────────┐       ┌────────────────┐
      │  Pre-Trained    │       │  Continued            │       │  Supervised            │       │  Alignment            │       │  Evaluation &        │       │  Deployment    │
      │  Foundation     │──────▶│  Pre-Training         │──────▶│  Fine-Tuning (SFT)     │──────▶│  (RLHF / DPO)         │──────▶│  Merging             │──────▶│  & Serving     │
      │  Model          │       │                       │       │                        │       │                       │       │                      │       │                │
      │  (LLaMA,        │       │  Unsupervised on      │       │  Instruction-response  │       │  Preference-based     │       │  Benchmarks,         │       │  API, vLLM,    │
      │   Mistral,      │       │  domain-specific      │       │  pairs (labeled data)  │       │  optimization         │       │  merge LoRA into     │       │  TGI, GGUF,    │
      │   Qwen, etc.)   │       │  corpus (OPTIONAL)    │       │                        │       │                       │       │  base, A/B test      │       │  Ollama        │
      └─────────────────┘       └───────────────────────┘       └────────────────────────┘       └───────────────────────┘       └──────────────────────┘       └────────────────┘
                                  │                               │                               │                               │                              │                                   
                                  │ OPTIONAL                      │  Choose: Full FT or PEFT      │  Choose: Full FT or PEFT      │ Benchmarks,                  │ API,
                                  │ Domain corpus                 │  LoRA / QLoRA most popular    │  LoRA + DPO most popular      │ merge adapters,              │ vLLM,
                                  │ (medical, legal, code)        │  at this stage                │  at this stage                │ A/B test                     │ TGI
    
---

        ════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                          OTHER APPROACHES / SUPPLEMENTARY TECHNIQUES  (Can be applied at any stage)
        ════════════════════════════════════════════════════════════════════════════════════════════════════════════════


      ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
      │  Instruction       │  │  Knowledge         │  │  Multi-Task        │  │  Curriculum         │  │  NEFTune           │  │  Adapter Merging   │  │  Data Augmentation │
      │  Tuning            │  │  Distillation      │  │  Fine-Tuning       │  │  Learning           │  │                    │  │                    │  │                    │
      │                    │  │                    │  │                    │  │                     │  │                    │  │                    │  │                    │
      │  Diverse (instr,   │  │  Teacher model     │  │  Train on multiple │  │  Order training     │  │  Add noise to      │  │  Combine multiple  │  │  Synthetic data,   │
      │  output) datasets  │  │  → Student model   │  │  tasks jointly     │  │  data from          │  │  embeddings during │  │  LoRA adapters     │  │  paraphrasing,     │
      │  (FLAN, Alpaca,    │  │  via soft labels   │  │  with shared       │  │  easy → hard        │  │  training for      │  │  into single       │  │  back-translation  │
      │  OpenAssistant)    │  │  or logit matching │  │  backbone          │  │  for better         │  │  better            │  │  base model        │  │  for more          │
      │                    │  │                    │  │                    │  │  generalization     │  │  generalization    │  │  (TIES, DARE,      │  │  training data     │
      │                    │  │                    │  │                    │  │                     │  │                    │  │   linear merge)    │  │                    │
      └────────────────────┘  └────────────────────┘  └────────────────────┘  └─────────────────────┘  └────────────────────┘  └────────────────────┘  └────────────────────┘
      
---  

        ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                        PEFT METHODS — COMPARISON MATRIX
        ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
        
        
        ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬───────────────────┐
        │     CATEGORY        │    WHAT IT DOES  │  PARAMS TRAINED  │   MEMORY COST    │   PERFORMANCE    │   BEST FOR        │
        ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
        │                     │                  │                  │                  │                  │                   │
        │  ADDITIVE           │  Inserts new     │  0.1% – 3%       │       Low        │  Good            │  Task-specific    │
        │  (Adapters, IA³)    │  modules/params  │  of original     │                  │                  │  adaptation       │
        │                     │                  │                  │                  │                  │                   │
        ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
        │                     │                  │                  │                  │                  │                   │
        │  REPARAMETERIZATION │  Low-rank        │  0.1% – 1%       │    Low-Medium    │  Very Good       │  General-purpose  │
        │  (LoRA, DoRA)       │  matrix decomp   │  of original     │                  │  (near full FT)  │  fine-tuning      │
        │                     │                  │                  │                  │                  │                   │
        ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
        │                     │                  │                  │                  │                  │                   │
        │  SELECTIVE          │  Picks specific  │  0.01% – 0.1%    │    Very Low      │  Moderate        │  Minimal compute  │
        │  (BitFit, Freeze)   │  existing params │  of original     │                  │                  │  budget           │
        │                     │                  │                  │                  │                  │                   │
        ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
        │                     │                  │                  │                  │                  │                   │
        │  HYBRID             │  Combines (e.g.  │  0.1% – 1%       │    Very Low      │  Very Good       │  Large models     │
        │  (QLoRA)            │  quantize+LoRA)  │  of original     │  (4-bit base!)   │                  │  on single GPU    │
        │                     │                  │                  │                  │                  │                   │
        ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼───────────────────┤
        │                     │                  │                  │                  │                  │                   │
        │  PROMPT-BASED       │  Learns virtual  │  0.001% – 0.01%  │      Lowest      │  Moderate        │  Quick task       │
        │  (Prefix, P-Tuning) │  tokens/prompts  │  of original     │                  │                  │  switching        │
        │                     │                  │                  │                  │                  │                   │
        └─────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴───────────────────┘

---
### Fine Tuning Detailed Breakdown: How its done ! 

#####  Textual diagram of the fine-tuning process: 

            ┌─────────────────────────────────────────────────────────┐
            │                  FINE-TUNING PIPELINE                   │
            └─────────────────────────────────────────────────────────┘
            
                ┌──────────────┐             ┌──────────────────────┐
                │  Pre-trained │             │  Task-Specific Data  │
                │    Model     │             │  (labeled examples)  │
                │  (e.g. LLM)  │             │                      │
                └──────┬───────┘             └──────────┬───────────┘
                       │                                │
                       ▼                                ▼
                ┌───────────────────────────────────────────────┐
                │              DATA PREPARATION                 │
                │  ┌─────────┐  ┌──────────┐  ┌──────────────┐  │
                │  │ Clean & │→ │ Format   │→ │ Train / Val  │  │
                │  │ Filter  │  │ (prompt/ │  │    Split     │  │
                │  │         │  │ response)│  │              │  │
                │  └─────────┘  └──────────┘  └──────────────┘  │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                ┌───────────────────────────────────────────────┐
                │              TRAINING LOOP                    │
                │                                               │
                │   ┌────────────┐    ┌──────────────────────┐  │
                │   │ Forward    │───▶│ Compute Loss         │  │
                │   │ Pass       │    │ (cross-entropy, etc.)│  │
                │   └────────────┘    └──────────┬───────────┘  │
                │         ▲                      │              │
                │         │                      ▼              │
                │   ┌─────┴──────┐    ┌──────────────────────┐  │
                │   │ Update     │◀───│ Backpropagation      │  │
                │   │ Weights    │    │ (gradients)          │  │
                │   └────────────┘    └──────────────────────┘  │
                │                                               │
                │   Repeat for N epochs                         │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                ┌───────────────────────────────────────────────┐
                │              EVALUATION                       │
                │  ┌────────────┐  ┌───────────┐  ┌──────────┐  │
                │  │ Validation │→ │ Metrics   │→ │ Hyper-   │  │
                │  │ Set        │  │ (acc,     │  │ param    │  │
                │  │            │  │  loss, F1)│  │ Tuning   │  │
                │  └────────────┘  └───────────┘  └──────────┘  │
                └───────────────────────┬───────────────────────┘
                                        │
                                        ▼
                                ┌──────────────────┐
                                │  Fine-Tuned      │
                                │  Model           │
                                │  (deploy/serve)  │
                                └──────────────────┘
---


         
### The Big Picture 

Imagine you hired a brilliant generalist — someone who has read millions of books, articles, and websites. 
They know a little about everything. That's your pre-trained foundation model.

Now imagine you want this generalist to become a specialist — say, a medical diagnostician. 
You sit them down and train them intensively on thousands of medical cases, patient records, and diagnostic reports.

Full fine-tuning is that intensive retraining. You take the entire model — every single connection, every weight, 
every parameter — and allow ALL of them to be updated based on your new, task-specific data.

##### Why It Exists
Pre-trained models learn general language patterns during pre-training 
(which costs millions of dollars and takes weeks/months on huge GPU clusters). 
They learn grammar, facts, reasoning patterns, and general knowledge from massive internet-scale datasets.

But this general knowledge isn't enough for specific tasks. 
A model that can write poetry might be terrible at classifying legal contracts. 
A model that can summarize news might fail at generating SQL queries.

Full fine-tuning bridges this gap: take the general knowledge and reshape ALL of it toward your specific goal.

---

### How It Works Conceptually

* Step 1: Start With a Pre-Trained Model

    Think of the model as a massive web of interconnected numbers (called weights or parameters).
    A model like LLaMA-7B has 7 billion of these numbers. GPT-3 has 175 billion.
    Each number contributes to how the model processes and generates language.
    
    During pre-training, these billions of numbers were carefully tuned so the model could predict the next word in a
    sentence across trillions of words of text.
    The result is a model that "understands" language at a deep statistical level.

* Step 2: Prepare Your Task-Specific Dataset: 

    You gather data specific to your goal. Examples:
    
    - Sentiment analysis: thousands of movie reviews labeled "positive" or "negative"
    - Medical Q&A: thousands of (medical question → accurate answer) pairs
    - Code generation: thousands of (natural language description → working code) pairs
    - Summarization: thousands of (long article → concise summary) pairs
    
    The key is that this dataset is much smaller than the pre-training data.
    Pre-training might use trillions of tokens. Fine-tuning might use thousands to millions of examples.
    
    **Data Formatting Matters:**
    For modern LLM fine-tuning, the format of your data is just as important as the content.
    
    Common formats include:

    - Instruction-tuning format: structured as (system prompt → user instruction → assistant response) triples.
    This is the standard for chat/instruction models.
    
    - Completion-only format: simple (input → output) pairs without role structure.
    
    - Chat templates: model-specific formatting (e.g., ChatML, LLaMA chat format) that the model was pre-trained with.
    Using the wrong template can significantly degrade performance.
    
    **Loss Masking:**
    During training, you typically only compute the loss on the output/response tokens,
    not on the input/instruction tokens. This tells the model "learn to generate better responses"
    rather than "learn to memorize the questions." This is sometimes called "completion-only loss"
    and is controlled by flags like **train_on_input=False** in most fine-tuning frameworks.

* Step 3: The Training Loop (What Actually Happens)

    Here's where the magic happens. For each example in your dataset, the model goes through a cycle:
    
    - A)  Forward Pass — The Model Makes a Prediction: 
    
        You feed an input into the model. 
        The input flows through all the layers — embedding layer, attention layers, feed-forward layers — and the model produces an output (a prediction).
        
        Think of it like water flowing through a complex network of pipes. 
        Each pipe has a valve (a weight) that controls how much water flows through. 
        The water enters at one end (input) and comes out the other end (output/prediction).
        
            
    - B) Loss Calculation — How Wrong Was It?
        
        You compare the model's prediction to the correct answer from your dataset. 
        The difference is called the loss. A high loss means the model was very wrong. A low loss means it was close.
        
        For example, if the model predicted "negative" for a movie review that was actually "positive," the loss would be high.
        
    - C) Backpropagation — Tracing the Blame 
    
        This is the critical step. 
        The algorithm works backward through the entire network, asking: "Which weights contributed to this mistake, and by how much?"
        
        Imagine the water came out the wrong pipe at the end. 
        Backpropagation traces backward through every pipe junction, calculating exactly which valves need to be 
        adjusted and by how much to redirect the water correctly.
        
        This produces a gradient for every single weight in the model — 
        —a mathematical direction that says "adjust this weight by this amount to reduce the error."

    - D) Weight Update — Adjusting Everything
        
        Using an optimizer (like Adam or SGD), every weight in the model is nudged in the direction that reduces the error. 
        The size of each nudge is controlled by the learning rate — a hyperparameter you set.
    
            - Learning rate too high → model changes too aggressively, overshoots, becomes unstable
            - Learning rate too low → model changes too slowly, takes forever, might get stuck
            - Just right → model gradually improves

        In full fine-tuning, ALL billions of weights are updated in step D. This is the defining characteristic.
    
    - E) Repeat

        This cycle (forward → loss → backward → update) repeats for every batch of examples, 
        across multiple passes through the entire dataset (called epochs).

    **Batch Size and Gradient Accumulation:**
    
    In practice, you can rarely fit your ideal batch size into GPU memory all at once.
    The solution is gradient accumulation: process several smaller micro-batches,
    accumulate the gradients from each, and only perform the weight update after N micro-batches.
    For example, if your target batch size is 32 but you can only fit 4 examples in GPU memory at once,
    you set 
    
    gradient_accumulation_steps = 8 (4 × 8 = 32 effective batch size).
    
    The model sees the same total gradient as if you processed all 32 at once — it just takes 8 micro-steps to get there.
    This is critical for full fine-tuning where memory is already tight.

---

### The Learning Rate Dilemma

This is one of the most important concepts in full fine-tuning. 
The learning rate for fine-tuning is typically much smaller (often 10x to 100x smaller) than what was used during pre-training.

Why? Because the pre-trained weights already encode valuable knowledge. 
You don't want to destroy that knowledge — you want to gently reshape it.

Think of it like this: the pre-trained model is a beautifully sculpted statue. 
Fine-tuning is like carefully chiseling new details onto it. 
If you chisel too aggressively (high learning rate), you destroy the original sculpture. 
If you're too gentle (low learning rate), you never finish the new details.

Typical learning rate ranges for full fine-tuning: 1e-6 to 5e-5 (compare this to LoRA which uses 5e-5 to 5e-4 —
PEFT methods can afford higher learning rates because they're only updating a tiny fraction of parameters).

### Catastrophic Forgetting

This is the biggest danger of full fine-tuning. 
When you update all the weights for your specific task, the model can "forget" the general knowledge it learned during pre-training.

Imagine teaching your medical specialist so intensively that they forget how to speak grammatically, 
or forget basic common sense, or forget everything about every other topic.

This happens because the new task-specific gradients can overwrite the patterns stored in the weights from pre-training. 
The model becomes narrow — excellent at your specific task but degraded at everything else.

Mitigation strategies:

- Small learning rate: gentle updates preserve more original knowledge
- Early stopping: stop training before the model overfits and forgets
- Regularization: mathematically penalize weights that drift too far from their pre-trained values
- Data mixing: include some general-purpose data alongside your task-specific data
- Gradual unfreezing (a variant — see below)

---

#### Evaluation and Knowing When to Stop

Training without proper evaluation is flying blind. Here's what to monitor:

**Validation Loss vs. Training Loss:**
Split your dataset into training (~90%) and validation (~10%) sets.
Track both losses during training. The critical signal is divergence:

- Training loss keeps dropping, validation loss also drops → model is learning, keep going
- Training loss keeps dropping, validation loss plateaus → model is starting to memorize, be cautious
- Training loss keeps dropping, validation loss rises → model is overfitting, stop training

**Early Stopping:**
In practice, you don't train for a fixed number of epochs and hope for the best.
You monitor validation loss and stop when it hasn't improved for N evaluation steps (called "patience").
Save checkpoints at each evaluation step so you can roll back to the best one.

**Task-Specific Metrics:**
Loss alone doesn't tell the whole story. Track metrics relevant to your task:

- Classification: F1 score, precision, recall, accuracy
- Generation/Summarization: BLEU, ROUGE, BERTScore
- Q&A: Exact match, F1 on answer spans
- Code generation: pass@k (does the generated code actually run and pass tests?)

These metrics sometimes diverge from loss — a model might have slightly higher loss
but produce more practically useful outputs.

---

### Variants of Full Fine-Tuning

- A) Standard Full Fine-Tuning
        All layers, all weights, updated from the start. Maximum flexibility, maximum risk of forgetting, maximum compute cost.
    
- B) Feature Extraction
        
    You freeze the entire pre-trained model and only train a new classification head (a small layer added on top). 
    The pre-trained model acts as a fixed feature extractor.
    
    Think of it as: you don't retrain the generalist at all. 
    You just put a specialist translator at the end who interprets what the generalist says and converts it into task-specific answers.
    
    This is technically not "full" fine-tuning, but it's on the spectrum. 
    It's the safest (no forgetting) but least flexible (the model can't adapt its internal representations).

- C) Gradual Unfreezing
        
    You start by freezing most layers and only training the top layers. 
    Then, epoch by epoch, you unfreeze deeper layers, allowing more of the model to adapt.
    
    Think of it as: first train the specialist translator (top layers), 
    then gradually allow the generalist to adapt their thinking (deeper layers) once the translator is stable.
    
    This balances adaptation and preservation beautifully.

- D) Layer-Selective Fine-Tuning

    Rather than all-or-nothing, you choose specific layers to train while freezing the rest.
    Common strategies include: 
    
    - Last N layers — most common, works well for classification tasks
    - First + Last layers — captures both low-level and task-specific features
    - Every Nth layer — distributes adaptation throughout the model

    This sits between Feature Extraction and Full Fine-Tuning on the spectrum,
    and connects directly to PEFT selective methods like BitFit (which only trains bias terms — ~0.1% of parameters —
    and is surprisingly effective).
    
    **The full spectrum looks like:**
    
        |Feature Extraction → Layer-Selective → Gradual Unfreezing → Full Fine-Tuning |
        |                   |                 |                    |                  |
        |(least flexible,   |  (moderate)     | (balanced)         | (most flexible,  |  
        |safest)            |                 |                    |      most risk)  |
    
    
    
---

### The Cost Problem

Full fine-tuning is expensive. Here's why:
Memory: During training, you need to store in GPU memory:

- The model weights themselves (e.g., 7B parameters × 4 bytes = 28 GB for a 7B model in float32)
- The gradients for every weight (same size as the weights — another 28 GB)
- The optimizer states (Adam stores 2 extra values per weight — another 56 GB)
- The activations from the forward pass (needed for backpropagation — variable, can be huge)


**Realistic Memory Math (Mixed Precision — How It's Actually Done):**
In practice, almost nobody fine-tunes in full float32 anymore.
Mixed precision training (BF16/FP16) is standard and roughly halves the memory for weights and gradients:

For a 7B parameter model with BF16 mixed precision + Adam optimizer:

    Model weights (BF16):           7B × 2 bytes  = ~14     GB
    Gradients (BF16):               7B × 2 bytes  = ~14     GB
    Optimizer states (FP32):        7B × 8 bytes  = ~56     GB    ← Adam keeps momentum + variance in FP32 for stability
    Activations (variable):                       = ~10-30  GB    (depends on batch size and sequence length)
    ─────────────────────────────────────────────────────────────
    Total:                                            ~94-114 GB


Note: Even with mixed precision, the optimizer states dominate memory.
This is why the Adam optimizer's FP32 states are often the bottleneck,
and why techniques like 8-bit Adam (from bitsandbytes) exist to compress them.

For a 70B parameter model: scale everything by 10× → ~1 TB+, requiring multiple GPUs.

So for a 7B parameter model, you might need 120+ GB of GPU memory just for training. 
For a 70B model, you're looking at over 1 TB — requiring multiple expensive GPUs.

---

### Gradient Checkpointing (Trading Compute for Memory):

The activations from the forward pass can consume enormous memory (they're needed for backpropagation).
Gradient checkpointing is a technique that discards most intermediate activations during the forward pass
and recomputes them on-the-fly during the backward pass.

- Without checkpointing: store all activations → high memory, normal speed
- With checkpointing: store only some activations, recompute the rest → ~30-40% less memory, ~20-30% slower

This is almost always enabled for full fine-tuning of large models. 
Think of it as choosing to re-derive a formula during the exam rather than memorizing it — 
saves brain space (memory) but costs time (compute).

Compute: Every training step requires a forward pass AND a backward pass through the ENTIRE model, updating ALL parameters. 
This is slow and costly.

Storage: You save a complete copy of all parameters. 
Every fine-tuned version is as large as the original model. 
If you fine-tune for 10 different tasks, you store 10 full copies.
This cost problem is exactly why PEFT methods like LoRA were invented — but that's for a later step.

---

### Distributed Training — How You Actually Scale

When a model doesn't fit on a single GPU (which is most full fine-tuning scenarios),
you distribute the work across multiple GPUs. 

Two dominant approaches:

- **DeepSpeed ZeRO (Zero Redundancy Optimizer):**
By default, each GPU holds a full copy of weights, gradients, and optimizer states — massively redundant.
ZeRO eliminates this redundancy in stages:

- ZeRO Stage 1: Shard optimizer states across GPUs (biggest memory win, ~4× reduction)
- ZeRO Stage 2: Also shard gradients across GPUs
- ZeRO Stage 3: Also shard model weights across GPUs (most memory efficient, enables training 
                                                      models that don't fit on any single GPU)
                 
**PyTorch FSDP (Fully Sharded Data Parallel):**
PyTorch's native equivalent of ZeRO Stage 3. Shards weights, gradients, and optimizer states across GPUs
and only gathers the full parameters when needed for computation.    

    - Practical example:
    A 70B model in BF16 needs ~140 GB just for weights. 
    With 8× A100 80GB GPUs and DeepSpeed ZeRO-3 or FSDP, each GPU only holds ~17.5 GB of sharded weights, 
    leaving room for gradients, optimizer states, and activations.
 
These tools are what make full fine-tuning of 70B+ models physically possible.
Without them, you'd need GPUs with terabytes of memory that don't exist.

---
This Cost Problem is Exactly Why PEFT Methods Like LoRA Were Invented
Every pain point listed above has a corresponding PEFT solution:

    Full Fine-Tuning Problem            →  PEFT Solution
    ────────────────────────────────────────────────────────────────────────────────────
    120+ GB memory for 7B model         →  LoRA: ~16 GB (trains only ~0.3% of params)
    1 TB+ for 70B model                 →  QLoRA: ~36-48 GB (4-bit base + LoRA adapters)
    Full copy per task (14-140 GB each) →  LoRA adapters: ~20-200 MB per task, swappable
    Catastrophic forgetting             →  Base model frozen, only adapters updated
    Hours/days of training              →  Much faster convergence with fewer parameters
    Multi-GPU clusters required         →  Single GPU feasible for most model sizes

---

When Full Fine-Tuning Makes Sense
Despite its costs, full fine-tuning is the right choice when:

- You have enough data (tens of thousands to millions of examples)
- You have enough compute (multiple high-end GPUs)
- You need maximum performance on your specific task
- Your task is significantly different from what the model learned in pre-training
- You're building a production system where even small accuracy gains matter
- You're a large organization with the infrastructure (Google, Meta, OpenAI, etc.)

When It Doesn't Make Sense

- Limited compute budget → use PEFT
- Small dataset (hundreds of examples) → high risk of overfitting with full fine-tuning
- Need to serve multiple tasks from one model → PEFT adapters are modular
- Rapid experimentation → full fine-tuning is too slow for quick iterations

---

### Summary Mental Model

    PRE-TRAINED MODEL                    FULL FINE-TUNING                        RESULT
    ┌─────────────────┐                                                      ┌──────────────────┐
    │ ░░░░░░░░░░░░░░░ │   All 7B+ weights           Every weight nudged      │ ████████████████ │
    │ ░░░░░░░░░░░░░░░ │   are UNLOCKED       ──▶    toward your task    ──▶  │ ████████████████ │
    │ ░░░░░░░░░░░░░░░ │   and trainable             via gradient descent     │ ████████████████ │
    │ ░ General       │                                                      │ █ Specialized    │
    │ ░ Knowledge ░░░ │                                                      │ █ Knowledge ████ │
    └─────────────────┘                                                      └──────────────────┘
      (knows everything                                                         (expert at your
       a little bit)                                                            task, may lose
                                                                                some generality)

## Solution to the cost Problem

PEFT and LoRA, which were invented specifically to solve the cost and forgetting problems described above.

---

### How is Full Fine Tuning Carried Out - Under the hood

Let me walk through the full data pipeline from raw data to what actually hits the model.

**Input Data:**

The input data is NOT pre-computed embedding vectors. 
This is a common misconception. You feed the model raw text, and the model's own embedding layer converts it to vectors
on the fly during training. The embedding layer itself is part of the model and gets updated during full fine-tuning. 
    
## What the Raw Data Looks Like
Your fine-tuning dataset is just text — stored in simple, standard file formats:
JSONL (JSON Lines) — the most common format:

JSON:
    {"instruction": "Classify the sentiment", "input": "This movie was breathtaking", "output": "positive"}
    {"instruction": "Classify the sentiment", "input": "Terrible waste of time", "output": "negative"}
    {"instruction": "Summarize this article", "input": "The Federal Reserve announced...", "output": "The Fed raised rates by..."}


Each line is one training example. A fine-tuning dataset might be a single .jsonl file that's a few hundred MB to a few GB.

Other common formats:

- CSV/TSV — simple tabular format
- Parquet — columnar format, compressed, used by Hugging Face datasets
- JSON — nested structures for multi-turn conversations
- Plain text — for continued pre-training (just raw text, no instruction structure)


#### For chat/instruction models, the data often looks like:
    
    {
        "conversations": 
        [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": "What causes migraines?"},
            {"role": "assistant", "content": "Migraines are caused by..."}
        ]
    }
```

That's it. No vectors. No embeddings. Just text in files.

---

#### How Text Becomes Numbers (Tokenization → Embedding)

Before the model can process text, two things happen — and this is where the vectors come in:

---

**Step 1: Tokenization (text → integer IDs)**

A tokenizer breaks text into subword tokens and maps each to an integer ID. This happens *before* the model, as a preprocessing step.

```
    "This movie was breathtaking"
        ↓ tokenizer
    [1552, 5765, 471, 4800, 28107]
    
```

These are just integer IDs — a lookup index. 
The tokenizer is fixed and never changes during fine-tuning. 
It's stored as a separate vocabulary file (usually `tokenizer.json` or `tokenizer.model`, a few MB).

---

**Step 2: Embedding Lookup (integer IDs → vectors)**

The model's first layer is an embedding table — essentially a giant matrix where each row corresponds to one token ID. 
When token ID 1552 enters the model, it looks up row 1552 and pulls out a dense vector (e.g., 4096 dimensions for a 7B model).


```
Token ID 1552 → [0.023, -0.891, 0.445, ..., 0.112]  (4096 floats)
Token ID 5765 → [0.771, 0.034, -0.562, ..., -0.338]  (4096 floats)
```

This happens *inside the model* during the forward pass, on the GPU, in real time. 
You never pre-compute or store these embedding vectors. 
The embedding table is part of the model's weights and gets updated during full fine-tuning like everything else.

---

### How the Data Is Actually Stored and Loaded

**Storage — it's simple files on disk:**
```
my_dataset/
├── train.jsonl          # 500K examples, ~2 GB
├── validation.jsonl     # 50K examples, ~200 MB
└── metadata.json        # dataset info
```

Or if using Hugging Face datasets (the most common approach):
```
dataset/
├── data/
│   ├── train-00000-of-00004.parquet
│   ├── train-00001-of-00004.parquet
│   ├── train-00002-of-00004.parquet
│   └── train-00003-of-00004.parquet
└── dataset_info.json
```

Parquet files are split into shards for parallel loading. 
Hugging Face datasets use Apache Arrow format under the hood — memory-mapped files that let you work with datasets larger than RAM without loading everything at once.

**No database involved.** You're not querying a database during training. 
The data sits on disk (local SSD or cloud storage like S3) and gets streamed into memory in batches. 
Training data pipelines are optimized for sequential throughput, not random access — the opposite of what databases are designed for.

---

### How Data Flows During Training — Not One Line, Not All at Once

This is the key part. It's neither one example at a time nor everything at once. It's **batches**.

**The DataLoader pipeline:**

    ```
    Disk (JSONL/Parquet files)
        ↓
    DataLoader reads a batch of raw examples (e.g., 8 examples)
        ↓
    Tokenizer converts text → token IDs for each example
        ↓
    Padding/Truncation — all sequences in the batch must be the same length
        (shorter ones get padded with a special [PAD] token,
         longer ones get truncated to max_seq_length, e.g., 2048 or 4096 tokens)
        ↓
    Collation — stack into a single tensor of shape [batch_size, seq_length]
        e.g., [8, 2048] — 8 examples, each 2048 tokens
        ↓
    Move to GPU memory
        ↓
    Model's embedding layer converts token IDs → vectors
        Now shape is [8, 2048, 4096] — 8 examples, 2048 positions, 4096-dim vectors
        ↓
    Forward pass through all transformer layers
        ↓
    Loss, backprop, weight update
        ↓
    Next batch
```
**Crucially, the DataLoader:**
- Shuffles the dataset at the start of each epoch (so the model sees examples in different order)
- Loads batches in parallel using multiple CPU workers while the GPU is processing the current batch
- Pre-fetches the next batch so the GPU is never waiting on data

**What "batch" means in practice with gradient accumulation:**

Say you want an effective batch size of 32 but can only fit 4 examples on the GPU:
```
Micro-batch 1:  4 examples → forward → loss → backward → accumulate gradients
Micro-batch 2:  4 examples → forward → loss → backward → accumulate gradients
Micro-batch 3:  4 examples → forward → loss → backward → accumulate gradients
...
Micro-batch 8:  4 examples → forward → loss → backward → accumulate gradients
                                                            ↓
                                              Weight update (using accumulated gradients from all 32 examples)
```

### The DataLoader pipeline Breakdown :

---
**What's sitting in your JSONL file:**

{"instruction": "Classify the sentiment", "input": "This movie was breathtaking", "output": "positive"}
{"instruction": "Classify the sentiment", "input": "Terrible waste of time", "output": "negative"}
{"instruction": "Translate to French", "input": "The cat sat on the mat", "output": "Le chat était assis sur le tapis"}


Just text. Nothing has happened yet.

---

### Step 1: Template Formatting (Text → Structured Text)
---

The DataLoader first applies a **chat template** to combine the fields into a single string. 
This is model-specific — different models expect different formats.

**For a LLaMA-style model:**

    Example 1: "<s>[INST] Classify the sentiment: This movie was breathtaking [/INST] positive</s>"
    Example 2: "<s>[INST] Classify the sentiment: Terrible waste of time [/INST] negative</s>"
    Example 3: "<s>[INST] Translate to French: The cat sat on the mat [/INST] Le chat était assis sur le tapis</s>"


**For a ChatML-style model (like Mistral/OpenChat):**
    
    Example 1: "<|im_start|>user\nClassify the sentiment: This movie was breathtaking<|im_end|>\n<|im_start|>assistant\npositive<|im_end|>"

This is where **special tokens** enter the picture:

```
Special Token    Purpose                          When Added
─────────────────────────────────────────────────────────────
<s> / BOS        Beginning of sequence             Start of the whole sequence
</s> / EOS       End of sequence                   End of the whole sequence
[INST] [/INST]   Instruction boundaries            Around the instruction/input
[CLS]            Classification token (BERT-era)   NOT used in modern LLM fine-tuning
[SEP]            Separator (BERT-era)              NOT used in modern LLM fine-tuning
[PAD]            Padding (comes later)             During batching — Step 4

```

##### **Important distinction:** 

**`[CLS]`** and **`[SEP]`** are BERT-family tokens. Modern decoder-only LLMs (LLaMA, Mistral, GPT) don't use them.
They use BOS, EOS, and model-specific instruction markers instead. 
If you see **`[CLS]`** and **`[SEP]`**, you're looking at older encoder-model fine-tuning (BERT, RoBERTa).

Still just text at this point. No numbers yet.

---

### Step 2: Tokenization (Text → Integer IDs)

The tokenizer breaks each formatted string into subword tokens and maps each to an integer ID from the vocabulary.


Example 1 (short):

    "&lt;s&gt;[INST] Classify the sentiment: This movie was breathtaking [/INST] positive&lt;/s&gt;"
        
                ↓ tokenizer
                
    [1, 518, 25580, 29962, 4134, 1598, 278, 19688, 29901, 910, 14064, 471, 4800, 28107, 518, 29914, 25580, 29962, 6374, 2]

That's 20 tokens.

Example 2 (longer):

    "&lt;s&gt;[INST] Translate to French: The cat sat on the mat [/INST] Le chat était assis sur le tapis&lt;/s&gt;"
        
        ↓ tokenizer
    
    [1, 518, 25580, 29962, 4103, 9632, 304, 5765, 29901, 450, 6635, 3290, 373, 278, 1775, 518, 29914, 25580, 29962, 997, 13563, 4496, 465, 275, 1190, 454, 260, 11690, 2]

That's 29 tokens.

===

Example 1 (short):
"<s>[INST] Classify the sentiment: This movie was breathtaking [/INST] positive</s>"
    ↓ tokenizer
[1, 518, 25580, 29962, 4134, 1598, 278, 19688, 29901, 910, 14064, 471, 4800, 28107, 518, 29914, 25580, 29962, 6374, 2]

That's 20 tokens.

Example 3 (longer):
"<s>[INST] Translate to French: The cat sat on the mat [/INST] Le chat était assis sur le tapis</s>"
    ↓ tokenizer
[1, 518, 25580, 29962, 4103, 9632, 304, 5765, 29901, 450, 6635, 3290, 373, 278, 1775, 518, 29914, 25580, 29962, 997, 13563, 4496, 465, 275, 1190, 454, 260, 11690, 2]

That's 29 tokens.
===

Notice the problem: **different examples have different lengths** (20 vs 29 tokens). 
GPUs need rectangular tensors — every row must be the same length. This is where padding comes in.

What we have now: a list of integer arrays of varying lengths. No vectors yet — just integer IDs.

---

---

### Step 3: Create the Label/Target Array and Loss Mask

Before batching, the DataLoader creates the **labels** — what the model should predict — and the **loss mask** that tells training which tokens to grade.
```
Example 1:
Input IDs: [1, 518, 25580, 29962, 4134, 1598, 278, 19688, 29901, 910, 14064, 471, 4800, 28107, 518, 29914, 25580, 29962, 6374, 2]

Labels:     [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 6374, 2]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^
             instruction + input tokens: IGNORED by loss                                                                  output tokens: GRADED
```

`-100` is a magic number in PyTorch that means "ignore this token when computing loss." The loss function skips any position with -100.

This is the **loss masking** we discussed earlier. The model sees the full sequence during the forward pass, but only gets graded on the output portion.

---

### Step 4: Padding and Attention Masking (This Is Where Padding Happens)

Now the DataLoader collects a batch of examples (say batch_size=4) and must make them all the same length.

```

Before padding (variable lengths):
Example 1: [1, 518, 25580, ..., 6374, 2]             → 20 tokens
Example 2: [1, 518, 25580, ..., 8178, 2]             → 18 tokens
Example 3: [1, 518, 25580, ..., 11690, 2]            → 29 tokens  ← longest
Example 4: [1, 518, 25580, ..., 1781, 2]             → 24 tokens

After RIGHT-PADDING to length 29 (length of longest example):
Example 1: [1, 518, 25580, ..., 6374, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]          → 29 tokens
Example 2: [1, 518, 25580, ..., 8178, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    → 29 tokens
Example 3: [1, 518, 25580, ..., 11690, 2]                                    → 29 tokens (no padding needed)
Example 4: [1, 518, 25580, ..., 1781, 2, 0, 0, 0, 0, 0]                      → 29 tokens

```

The `0` here is the PAD token ID. **This is what padding is** — adding filler tokens so all sequences are the same length. 
It has nothing to do with CLS or SEP or EOS — those are content tokens with meaning. Padding is meaningless filler.

**But there's a problem:** you don't want the model to pay attention to padding tokens. So the DataLoader also creates an **attention mask**:

```

Attention Mask (1 = real token, 0 = padding, ignore me):
Example 1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Example 2: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Example 3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Example 4: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

Labels are also padded with -100:
Example 1: [-100, ..., -100, 6374, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100]

```

Padding tokens get -100 in labels (ignored by loss) AND 0 in attention mask (ignored by attention). 
They're invisible to the model in every way.

---

### Step 5: Truncation (If Needed)

If any example exceeds `max_seq_length` (e.g., 2048 or 4096), it gets chopped:

---

Example with 5000 tokens, max_seq_length=2048:

    [1, 518, 25580, ..., token_2046, token_2047, 2]
     ↑                                           ↑
     kept: first 2048 tokens                     EOS forced at the end
     
     Tokens 2049-5000: DISCARDED

**This is a hard cutoff.** 

**Information beyond the max length is simply lost.** 

**This is why choosing the right max_seq_length matters —** 
**too short and you lose data, too long and you waste GPU memory (memory scales quadratically with sequence length in standard attention).**

---

### Step 6: Collation into Tensors (What Actually Goes to GPU)

The DataLoader's collator stacks everything into rectangular tensors:


    batch = {
        "input_ids"     :      tensor of shape [4, 29],    # 4 examples × 29 tokens each
        "attention_mask":      tensor of shape [4, 29],    # which tokens are real
        "labels"        :      tensor of shape [4, 29],    # what model should predict (-100 = ignore)
    }

---

These three tensors get moved from CPU to GPU. That's everything the model needs.

Visually as matrices:

input_ids [4 × 29]:

    ┌─────────────────────────────────────────────────────────┐
    │  1  518  25580  29962  4134  ...  6374  2   0   0   0   │  Example 1
    │  1  518  25580  29962  7301  ...  8178  2   0   0   0   │  Example 2
    │  1  518  25580  29962  4103  ... 11690  2               │  Example 3
    │  1  518  25580  29962  2431  ...  1781  2   0   0   0   │  Example 4
    └─────────────────────────────────────────────────────────┘
    
    attention_mask [4 × 29]:
    ┌─────────────────────────────────────────────────────────┐
    │  1   1   1   1   1   ...   1   1   0   0   0            │
    │  1   1   1   1   1   ...   1   1   0   0   0            │
    │  1   1   1   1   1   ...   1   1                        │
    │  1   1   1   1   1   ...   1   1   0   0   0            │
    └─────────────────────────────────────────────────────────┘
    
    labels [4 × 29]:
    ┌─────────────────────────────────────────────────────────┐
    │ -100 -100 -100 -100 -100 ... 6374  2  -100 -100 -100    │
    │ -100 -100 -100 -100 -100 ... 8178  2  -100 -100 -100    │
    │ -100 -100 -100 -100 -100 ...11690  2                    │
    │ -100 -100 -100 -100 -100 ... 1781  2  -100 -100 -100    │
    └─────────────────────────────────────────────────────────┘

---


### Step 7: Into the Model (Token IDs → Embeddings → Forward Pass)

NOW the model's embedding layer converts integer IDs to vectors:

---

    input_ids [4, 29]     →   Embedding Layer   →   hidden_states [4, 29, 4096]
    (integers)                 (lookup table)         (dense vectors)

Each integer ID gets replaced by a 4096-dimensional vector:
Token ID 518 → [0.023, -0.891, 0.445, ..., 0.112]  (4096 floats)

---

This 3D tensor `[batch_size, seq_length, hidden_dim]` is what flows through all 32 transformer layers, getting transformed at each step.

---

### The Full Picture

---

    DISK          STEP 1         STEP 2          STEP 3         STEP 4          STEP 5          STEP 6         STEP 7
    Raw JSONL  →  Template    →  Tokenize     →  Create      →  Pad +        →  Truncate     →  Stack into →  Embed +
                  Formatting     to IDs          Labels +       Attention        (if needed)     Tensors       Forward Pass
                                                 Loss Mask      Mask
    
    "Classify   "<s>[INST]     [1, 518,        labels:        input_ids       (chop to        [4, 29]       [4, 29, 4096]
    sentiment:   Classify..."   25580, ...]     [-100,...,     padded to       max_seq_len)    tensor         3D tensor
    This movie                                  6374, 2]      same length                     moved to       flows through
    was great"                                                + attn_mask                      GPU            32 layers
    → positive              

**Note:** **padding (Step 4) is adding meaningless filler zeros so all examples in a batch have equal length.** 
**Special tokens like BOS/EOS (Step 1) are meaningful markers that tell the model where sequences begin and end.**

---

{{FFT_IMAGE}}

---

### What Lives on GPU vs. CPU vs. Disk

```
DISK (SSD/Cloud Storage)              CPU RAM                         GPU VRAM
─────────────────────────        ─────────────────────          ─────────────────────
Full dataset files                Loaded batches being           Current micro-batch
(JSONL, Parquet)                  tokenized/preprocessed         (as tensors)
                                                                
Model checkpoints                 DataLoader workers             Model weights
(saved periodically)              prefetching data               Gradients
                                                                 Optimizer states
                                  Memory-mapped dataset          Activations
                                  index (Arrow/Parquet)          (for backpropagation)
                                  
                                  

```

"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY / COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY = """
| Aspect          | Detail          |
|-----------------|-----------------|
| Parameters      |                 |
| Training Time   |                 |
| Inference Time  |                 |
"""



# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
#
# Each operation IS a real pipeline step.
# "pipeline_cmd" → what --run argument to pass to Full_fine_tuning_main.py
# "code"         → shows the user what the step does under the hood
# "runnable"     → enables the Run button
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    "1. Verify HuggingFace Token": {
        "description": "Validates your HuggingFace credentials and checks model access permissions",
        "runnable": True,
        "pipeline_cmd": "token",
        "code": '''# What this step does (from HF_Token_Verification.py):
# =============================================
# 1. Checks for HF_TOKEN in system environment variables
# 2. Checks for HF_TOKEN in Keys.env file
# 3. Warns if both exist and conflict
# 4. Validates the token against HuggingFace API
# 5. Checks if the token has access to the target model
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run token --yes
'''
    },

    "2. Check VRAM Requirements": {
        "description": "Estimates GPU memory needed for training based on your config (model size, batch size, precision)",
        "runnable": True,
        "pipeline_cmd": "vram",
        "code": '''# VRAM Estimation Formula
# =======================
# For a model with P parameters in bf16:
#
# Model Weights:     P x 2 bytes  (bf16 = 2 bytes per param)
# Gradients:         P x 2 bytes  (same dtype as weights)
# Optimizer (AdamW): P x 8 bytes  (2 states x 4 bytes each, fp32)
# Activations:       ~1-4 GB      (depends on seq_len, batch_size)
#
# Example: Llama-3.2-1B (1.24B params)
#   Weights:    1.24B x 2 = 2.48 GB
#   Gradients:  1.24B x 2 = 2.48 GB
#   Optimizer:  1.24B x 8 = 9.92 GB
#   Activations: ~1.5 GB (with gradient checkpointing)
#   TOTAL:      ~16.4 GB
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run vram --yes
'''
    },

    "3. Prepare Dataset": {
        "description": "Downloads the dataset from HuggingFace, applies chat template formatting, and tokenizes",
        "runnable": True,
        "pipeline_cmd": "prepare",
        "code": '''# Data Preparation Pipeline (from prepare_data.py):
# ================================================
# 1. Authenticate with HuggingFace using your token
# 2. Download dataset (default: yahma/alpaca-cleaned, ~52K examples)
# 3. Apply Llama chat template to each example:
#       {"instruction": "...", "input": "...", "output": "..."}
# 4. Tokenize all examples using the model's tokenizer
# 5. Create train/eval split
# 6. Return tokenized datasets ready for the Trainer
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run prepare --yes
'''
    },

    "4. Train Model (Full Fine-Tuning)": {
        "description": "TAKES HOURS — Trains ALL model parameters on the prepared dataset",
        "runnable": True,
        "pipeline_cmd": "train",
        "needs_confirmation": True,
        "code": '''# Full Fine-Tuning Training Loop:
# ================================
# 1. Load model in bf16 with device_map="auto"
# 2. Enable gradient checkpointing (saves VRAM)
# 3. Configure TrainingArguments:
#       - AdamW optimizer (fused), Cosine LR scheduler
#       - bf16 mixed precision
# 4. Run trainer.train()
#       - ~17,000 optimizer steps across 3 epochs
#       - Logs every 10 steps, evals every 200, saves every 500
# 5. Save final model + tokenizer
#
# ESTIMATED TIME:
#   RTX 3090:  ~3-6 hours
#   RTX 4090:  ~2-4 hours
#   A100:      ~1-2 hours
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run train --yes
'''
    },

    "5. Test Inference": {
        "description": "Generate text with the fine-tuned model to verify it works",
        "runnable": True,
        "pipeline_cmd": "inference",
        "code": '''# Inference Testing (from inference.py):
# ======================================
# 1. Load the fine-tuned model from outputs/final/
# 2. Send a test prompt through the model
# 3. Display the generated response
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run inference --yes \\
#       --prompt "What is machine learning?"
'''
    },

    "6. Compare Original vs Fine-Tuned": {
        "description": "Side-by-side comparison of the base model vs your fine-tuned version",
        "runnable": True,
        "pipeline_cmd": "compare",
        "code": '''# Model Comparison (from compare.py):
# ====================================
# 1. Load the ORIGINAL pre-trained model
# 2. Load YOUR fine-tuned model from outputs/final/
# 3. Send identical prompts to both
# 4. Display side-by-side responses
#
# Requires both models to fit in VRAM simultaneously.
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run compare --yes
'''
    },

    "7. Run Full Pipeline (1 to 6)": {
        "description": "Runs ALL steps sequentially — Token, VRAM, Data, Train, Inference, Compare",
        "runnable": True,
        "pipeline_cmd": "all",
        "needs_confirmation": True,
        "code": '''# Full Pipeline (all steps in sequence):
# ========================================
# Step 1: Verify HuggingFace token
# Step 2: Check VRAM requirements
# Step 3: Prepare dataset (download + tokenize)
# Step 4: Train (FULL fine-tuning takes HOURS)
# Step 5: Quick inference test
# Step 6: Compare original vs fine-tuned
#
# Each step must succeed before the next one starts.
#
# CLI equivalent:
#   python Full_fine_tuning_main.py --run all --yes
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM STREAMLIT RENDERER
#
# This replaces the default "show code in expander" behavior for this topic.
# Each operation gets: description + code viewer + RUN button + live output.
# ─────────────────────────────────────────────────────────────────────────────

def render_operations():
    """
    Custom Streamlit UI for the Operations tab.

    Called by app_Testing.py instead of the default code-in-expander rendering.
    Provides:
    - Configuration panel (editable training hyperparameters)
    - Run buttons for each pipeline step
    - Real-time streaming output from subprocess
    - Step status tracking (pending / running / success / failed)
    """
    import streamlit as st

    # ── Session State Init ──
    if "fft_step_outputs" not in st.session_state:
        st.session_state.fft_step_outputs = {}
    if "fft_step_status" not in st.session_state:
        st.session_state.fft_step_status = {}

    # ── Verify Script Exists ──
    script_path = _MAIN_SCRIPT
    scripts_dir = _SCRIPTS_DIR

    if not script_path.exists():
        st.error(
            f"**Pipeline script not found!**\n\n"
            f"Expected at:\n`{script_path}`\n\n"
            f"Edit the path variables `_SCRIPTS_DIR` / `_MAIN_SCRIPT` at the "
            f"top of `08_a_FineTuning_FullFineTuning.py` to match your layout."
        )
        st.markdown("---")
        st.caption("Falling back to code-only view:")
        _render_code_only(st)
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION PANEL (collapsible at top)
    # ═══════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Training Configuration — Edit before running", expanded=False):
        st.caption("These values will be used when running the pipeline steps.")

        col1, col2 = st.columns(2)

        with col1:
            model_name = st.selectbox(
                "Model",
                options=[
                    "unsloth/Llama-3.2-1B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "HuggingFaceTB/SmolLM2-360M-Instruct",
                    "openai-community/gpt2",
                ],
                index=0,
                key="fft_model_name",
            )
            max_seq_length = st.select_slider(
                "Max Sequence Length",
                options=[128, 256, 512, 1024, 2048],
                value=512,
                key="fft_seq_len",
            )
            batch_size = st.number_input(
                "Per-Device Batch Size",
                min_value=1, max_value=16, value=1,
                key="fft_batch_size",
            )
            grad_accum = st.number_input(
                "Gradient Accumulation Steps",
                min_value=1, max_value=64, value=8,
                key="fft_grad_accum",
            )

        with col2:
            num_epochs = st.number_input(
                "Epochs", min_value=1, max_value=10, value=3,
                key="fft_epochs",
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                value=2e-5,
                format_func=lambda x: f"{x:.0e}",
                key="fft_lr",
            )
            use_bf16 = st.checkbox("Use bf16", value=True, key="fft_bf16")
            use_grad_ckpt = st.checkbox(
                "Gradient Checkpointing", value=True, key="fft_grad_ckpt"
            )

        effective_bs = batch_size * grad_accum
        st.info(
            f"**Effective batch size:** {batch_size} x {grad_accum} = **{effective_bs}**"
        )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE STEPS — Each Operation = Expander with Run + Code + Output
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("### Pipeline Steps")
    st.caption("Click **Run** to execute. Output streams in real-time.")
    st.markdown("")

    for op_name, op_data in OPERATIONS.items():
        pipeline_cmd = op_data.get("pipeline_cmd", "")
        needs_confirm = op_data.get("needs_confirmation", False)
        step_key = pipeline_cmd

        # ── Status Icon ──
        status = st.session_state.fft_step_status.get(step_key, "pending")
        icon = {"pending": "⬜", "running": "🔄", "success": "✅", "failed": "❌"}.get(status, "⬜")

        # Auto-expand if there's output to show
        has_output = step_key in st.session_state.fft_step_outputs
        with st.expander(f"{icon} {op_name}", expanded=has_output):

            st.markdown(f"**{op_data['description']}**")

            # ── Code Details (toggle) ──
            if st.checkbox("Show code details", key=f"fft_showcode_{step_key}", value=False):
                st.code(op_data["code"], language="python")

            st.markdown("---")

            # ── Confirmation for dangerous steps ──
            run_disabled = False
            if needs_confirm:
                confirmed = st.checkbox(
                    "⚠️ I understand this takes **several hours** and I'm ready",
                    key=f"fft_confirm_{step_key}",
                    value=False,
                )
                run_disabled = not confirmed

            # ── Custom prompt for inference ──
            custom_prompt = None
            if step_key == "inference":
                custom_prompt = st.text_input(
                    "Test prompt:",
                    value="What is machine learning? Explain in 2 sentences.",
                    key="fft_inference_prompt",
                )

            # ── Buttons row ──
            col_run, col_clear = st.columns([3, 1])

            with col_run:
                if st.button(
                    f"▶️ Run",
                    key=f"fft_run_{step_key}",
                    disabled=run_disabled,
                    use_container_width=True,
                    type="primary" if step_key in ("train", "all") else "secondary",
                ):
                    _run_pipeline_step(
                        st, step_key, op_name,
                        script_path, scripts_dir,
                        prompt=custom_prompt,
                    )

            with col_clear:
                if has_output:
                    if st.button(
                        "Clear", key=f"fft_clear_{step_key}", use_container_width=True
                    ):
                        del st.session_state.fft_step_outputs[step_key]
                        st.session_state.fft_step_status[step_key] = "pending"
                        st.rerun()

            # ── Live Output Area ──
            if has_output:
                output = st.session_state.fft_step_outputs[step_key]
                if status == "success":
                    st.success("Completed successfully")
                elif status == "failed":
                    st.error("Step failed — see output below")
                st.code(output, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# SUBPROCESS RUNNER — Executes a pipeline step and streams output
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_step(st, step_key, step_label, script_path, scripts_dir, prompt=None):
    """
    Run a pipeline step via subprocess, streaming stdout to Streamlit.

    Launches:
        python Full_fine_tuning_main.py --run <step_key> --yes

    Output is read line-by-line and displayed in a live-updating st.code() block.
    """
    st.session_state.fft_step_status[step_key] = "running"

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--run", step_key,
        "--yes",
    ]

    if step_key == "inference" and prompt:
        cmd.extend(["--prompt", prompt])

    output_lines = []
    output_placeholder = st.empty()

    try:
        output_placeholder.info(f"🔄 Starting: {step_label} ...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(scripts_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            encoding="utf-8",
            errors="replace",
        )

        for line in process.stdout:
            clean = _strip_ansi(line)
            output_lines.append(clean)
            output_placeholder.code("".join(output_lines), language="text")

        process.wait()

        if process.returncode == 0:
            st.session_state.fft_step_status[step_key] = "success"
            output_lines.append(
                f"\n{'='*50}\n"
                f"  {step_label} — completed successfully.\n"
            )
        else:
            st.session_state.fft_step_status[step_key] = "failed"
            output_lines.append(
                f"\n{'='*50}\n"
                f"  {step_label} — failed (exit code {process.returncode}).\n"
            )

    except FileNotFoundError:
        st.session_state.fft_step_status[step_key] = "failed"
        output_lines.append(
            f"Could not find Python or script.\n"
            f"  Python: {sys.executable}\n"
            f"  Script: {script_path}\n"
        )
    except Exception as e:
        st.session_state.fft_step_status[step_key] = "failed"
        output_lines.append(f"Unexpected error: {e}\n")

    final_output = "".join(output_lines)
    st.session_state.fft_step_outputs[step_key] = final_output
    output_placeholder.code(final_output, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK — Code-only view (if script path is wrong)
# ─────────────────────────────────────────────────────────────────────────────

def _render_code_only(st):
    """Render operations as plain code expanders (no run buttons)."""
    for op_name, op_data in OPERATIONS.items():
        with st.expander(f"▶️ {op_name}", expanded=False):
            st.markdown(f"**Description:** {op_data['description']}")
            st.markdown("---")
            st.code(op_data["code"], language="python")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(text):
    """Remove ANSI color/formatting escape codes from terminal output."""
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this topic module."""
    from Required_Images.Fft_visual import FFT_VISUAL_HEIGHT, FFT_VISUAL_HTML

    # Fallback static image (still used if interactive component fails)
    mlp_img = _image_to_html(
        "Required_Images/FFT_Breakdown.png",
        alt="Full Finr Tuning Architecture",
        width="50%"
    )
    theory_with_images = THEORY.replace("{{FFT_IMAGE}}", mlp_img)

    # Interactive component data — app.py splits theory_raw at each
    # placeholder and injects an st.components.v1.html() iframe instead.
    interactive_components = [
        {
            "placeholder": "{{FFT_IMAGE}}",
            "html": FFT_VISUAL_HTML,
            "height": FFT_VISUAL_HEIGHT,
        }
    ]
    return {
        "theory": theory_with_images,
        "theory_raw": THEORY,
        "interactive_components": interactive_components,
        "complexity": COMPLEXITY,
        "operations": OPERATIONS,
        "render_operations": render_operations,
    }

