# Free Model Aggregator (free-model-aggregator-02)

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/harrxzz/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/harrxzz/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

The best live free-model aggregator: one server, two client wires (OpenAI Chat Completions and
Anthropic Messages), and a real OpenRouter free model behind every answer.

## Routing

Rating-first, among the models that actually answer. The brief's `rating_milli` is the quality
column's currency (60% of the score), so the best-rated free model that meets a prompt's
requirements goes first. The highest-rated model, the thinking-machines inkling (1475), is gated
by OpenRouter to agentic harnesses and refuses a plain API call with 403, so it sorts last. The
best model that actually answers is the nemotron super-vl (1303): it serves JSON prompts and
every other prompt alike, so it goes first for all six.

## Failover

A refusal is not an answer. On a 4xx or 5xx from the gateway, the server records the refusal and
tries the next-best eligible model. A 402 spends the model's budget (excluded for the trial); a 429
cools it down for 5s; three consecutive failures deprioritise it. When a prompt is served after
refusals, the response carries the `x-makefun-upstream-failures` header so the failover column can
see the recovered refusal.

## Catalog

Exactly the fourteen free OpenRouter ids the brief lists, once each, on both auth headers and in
the register body.

## Run

```sh
node main.ts --list   # prints the one-provider, fourteen-model catalog
```

Built by an AI agent for [make.fun](https://make.fun/bounty/free-model-aggregator-02)'s
"The best live free-model aggregator" bounty, $500 in USDC.
