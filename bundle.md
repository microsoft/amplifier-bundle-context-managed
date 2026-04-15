---
bundle:
  name: context-managed
  version: 0.1.0
  description: LLM-powered rolling context summarization with persistent transcript and budget-aware tracking

includes: []
---

# Context Managed Bundle

This bundle provides intelligent context management for Amplifier, replacing mechanical truncation with LLM-powered rolling summaries, persistent message history, and budget-aware context tracking.

## Components

### Context Manager Module

The core module implementing the `ContextManager` protocol. Manages message lifecycle: accepting messages, persisting to disk, maintaining rolling summary tiers, tracking budget, and assembling optimized message lists.

### Transcript Tool

A lightweight tool giving the LLM on-demand access to full-fidelity past messages that have been compressed into summaries. Reads from the same `transcript.jsonl` that the context module writes.

### Context Instructions

@context-managed:context/summary-instructions.md

Teaches the model how to interpret summaries and when to use the transcript tool.
