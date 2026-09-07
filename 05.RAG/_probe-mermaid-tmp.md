# TEMPORARY Mermaid probe (will be deleted after diagnosis)

## P1 trivial

```mermaid
flowchart LR
    A[Start] --> B[Finish]
```

## P2 copy of 11.1 diagram

```mermaid
flowchart LR
    D[Chunks] --> T[Tokenize and normalize]
    T --> I[(Inverted index)]
    Q[Query] --> QT[Tokenize]
    QT --> B[BM25 scoring]
    I --> B
    B --> K[Top-k lexical hits]
```

## P3 copy of chapter 3 diagram

```mermaid
flowchart TD
    A[Need external knowledge?] -->|No| P[Prompt or fine-tune behavior]
    A -->|Yes| B{All required evidence fits?}
    B -->|Yes| C{Need high-volume selective lookup?}
    C -->|No| L[Use long context]
    C -->|Yes| R[Use RAG]
    B -->|No| R
    R --> D{Need stable specialized behavior?}
    D -->|Yes| F[RAG plus fine-tuned model]
    D -->|No| E[RAG plus base model]
```

## P4 bulk trivial diagrams

```mermaid
flowchart LR
    X1[Node 1] --> Y1[Done 1]
```

```mermaid
flowchart LR
    X2[Node 2] --> Y2[Done 2]
```

```mermaid
flowchart LR
    X3[Node 3] --> Y3[Done 3]
```

```mermaid
flowchart LR
    X4[Node 4] --> Y4[Done 4]
```

```mermaid
flowchart LR
    X5[Node 5] --> Y5[Done 5]
```

```mermaid
flowchart LR
    X6[Node 6] --> Y6[Done 6]
```

```mermaid
flowchart LR
    X7[Node 7] --> Y7[Done 7]
```

```mermaid
flowchart LR
    X8[Node 8] --> Y8[Done 8]
```

```mermaid
flowchart LR
    X9[Node 9] --> Y9[Done 9]
```

```mermaid
flowchart LR
    X10[Node 10] --> Y10[Done 10]
```

```mermaid
flowchart LR
    X11[Node 11] --> Y11[Done 11]
```

```mermaid
flowchart LR
    X12[Node 12] --> Y12[Done 12]
```

```mermaid
flowchart LR
    X13[Node 13] --> Y13[Done 13]
```

```mermaid
flowchart LR
    X14[Node 14] --> Y14[Done 14]
```

```mermaid
flowchart LR
    X15[Node 15] --> Y15[Done 15]
```

```mermaid
flowchart LR
    X16[Node 16] --> Y16[Done 16]
```

```mermaid
flowchart LR
    X17[Node 17] --> Y17[Done 17]
```

```mermaid
flowchart LR
    X18[Node 18] --> Y18[Done 18]
```

```mermaid
flowchart LR
    X19[Node 19] --> Y19[Done 19]
```

```mermaid
flowchart LR
    X20[Node 20] --> Y20[Done 20]
```

## P5 copy of 27.1 diagram

```mermaid
flowchart TD
    B[Bad answer] --> S{Correct source exists and is authorized?}
    S -->|No| C[Fix corpus/scope or abstain]
    S -->|Yes| I{Correctly parsed and indexed?}
    I -->|No| P[Fix ingestion and re-index]
    I -->|Yes| R{In candidate set?}
    R -->|No| Q[Fix query/search/filter/ANN recall]
    R -->|Yes| K{Survived rerank and packing?}
    K -->|No| X[Fix rerank/dedupe/token budget]
    K -->|Yes| G{Answer supported?}
    G -->|No| V[Fix prompt/model/validation]
    G -->|Yes| U[Check UI, citation rendering, user expectation]
```
