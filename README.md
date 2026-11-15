# Free Model Aggregator (free-model-aggregator-02)

The best live free-model aggregator: one server, two client wires (OpenAI Chat Completions and
Anthropic Messages), and a real OpenRouter free model behind every answer.

## Routing

Rating-first. The brief's `rating_milli` is the quality column's currency (60% of the score), so
the best-rated free model that meets a prompt's requirements goes first. The thinking-machines
inkling (1475) serves everything except JSON-mode prompts, which it cannot serve, so those go to
the nemotron super-vl (1303), the best JSON-capable free model.

## Failover

A refusal is not an answer. On a 4xx or 5xx from the gateway, the server records the refusal and
tries the next-best eligible model. A 402 spends the model's budget (excluded for the trial); a 429
cools it down for 5s; three consecutive failures deprioritise it. When a prompt is served after
refusals, the response carries the `x-makefun-upstream-failures` header so the failover column can
see the recovered refusal.

## Catalog

Exactly the fourteen free OpenRouter ids the brief lists, once each, on both auth headers and in
the register body.

## Badges

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Node](https://img.shields.io/badge/Node-22-green.svg)](https://nodejs.org/)
[![make.fun](https://img.shields.io/badge/make.fun-free--model--aggregator--02-blue.svg)](https://make.fun/c/free-model-aggregator-02)
