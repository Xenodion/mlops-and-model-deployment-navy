# ADR 005 - The live API is stateless: the client carries the running state

**Status:** accepted, 20/11/2026.

## Context

A ride streams its windows to the service while it happens. The live score is a running mean over the windows
seen so far, so something must remember the sums between two requests.

## Decision

`POST /v1/predict/window` receives one window plus the client's state (sum of the category probabilities, sum of
the danger scores, number of windows) and returns the window's values, the updated cumulative values and the new
state. The server keeps nothing per ride.

## Alternatives rejected

- Server-side sessions keyed by ride id: need sticky routing, memory per ride, and are lost on a pod restart.
- WebSocket streaming: harder to load-test with Locust and to show in Swagger.

## Consequences

- Any replica can answer any window: horizontal scaling and the load test need no session affinity, which is what
  the AI-infrastructure scaling plan relies on.
- The capacity of one pod in live rides is measured directly: requests per second x window stride.
- The state is 7 numbers; a client could send a wrong state, which only affects its own ride.

## Evidence

`specs/001-final-deliverable/contracts/api.md`, research R4, commit 8248e55, issue #15.
