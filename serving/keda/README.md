# KEDA

Engine autoscaling is a `ScaledObject` rendered by each engine chart when
`autoscaling.enabled=true` (see `serving/helm/mock-engine/templates/scaledobject.yaml`).
It scales on `sum(vllm:num_requests_waiting)` from Prometheus. The logic is exercised on CPU
in kind with the mock engine and validated once on 2x L4 (ADR D5).

`keda-values.yaml` pins the KEDA Helm chart configuration used by `make up-autoscale`.
