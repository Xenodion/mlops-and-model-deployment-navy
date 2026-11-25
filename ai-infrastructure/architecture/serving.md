# Joint inference service

## Measured path on Vertex

```mermaid
flowchart LR
  G[Private GCS benchmark bundle] --> W[Vertex custom job]
  W --> C[Replay client]
  C -->|localhost HTTP| A[FastAPI worker]
  A --> R[Ride classifier]
  A --> F[Fall risk model]
  R --> O[Joint HTTP response]
  F --> O
  O --> C
  C --> B[Raw timing and quality artifacts]
```

The client sends a whole ride or a window from the same sensor stream. In live mode it passes the returned state with the next window. The response contains the ride classification and a separate fall-risk output. The measured path excludes external network latency.

## Proposed deployment

```mermaid
flowchart LR
  Client --> Ingress
  Ingress --> Pool[Replicas of the joint API]
  Registry[Pinned image and model artifacts] --> Pool
  Pool --> Metrics[Request rate, errors, route latency and memory]
  Metrics --> Operator
  Operator -->|adjust replicas or restore prior release| Pool
```

Keeping both models in each replica preserves the joint response with one request. Readiness should require both models. Version the image and both model references together so recovery restores a tested combination. During a rollout, keep successive live windows on a compatible release until the ride ends. Determine the replica count from concurrent mixed-traffic tests with a target p95 for each route. Sequential loopback latency alone cannot establish capacity.
