---
title: "Why Compute Governance Might Be Our Best Lever"
date: 2025-02-14
categories: [ai-safety]
tags: [governance, compute, policy, alignment]
description: "Hardware is the one chokepoint in the AI supply chain we can actually regulate. Here's why that matters."
featured: false
draft: false
---

The history of nuclear weapons contains a useful lesson: you cannot easily enrich uranium in a garage. The physics and engineering constraints created natural chokepoints that made governance tractable — not easy, but tractable.

AI training at frontier scale has an analogous property: you cannot train a GPT-4-class model without substantial compute, and that compute is produced by a small number of fabs and sold by an even smaller number of cloud providers.

## The concentration argument

As of 2025, the vast majority of frontier AI training happens on NVIDIA H100s and their successors. TSMC manufactures the chips. AWS, GCP, and Azure are the primary cloud providers. This is a very short supply chain.

This concentration is temporary — it will erode over years as competitors emerge — but in the window where it exists, it creates genuine policy leverage.

## What governance could look like

Three mechanisms seem most tractable:

1. **Know Your Customer (KYC) requirements** for large compute purchases
2. **International compute reporting standards** (analogous to export controls)
3. **"Compute permits"** for training runs above a certain FLOP threshold

None of these are sufficient on their own. All of them are more tractable than trying to regulate model weights or capabilities directly.

## The objection

Critics argue that compute governance is already outdated — that algorithmic efficiency improvements mean you can train increasingly powerful models on less compute. This is true. But the frontier still moves with compute, and governance that catches 80% of risk is better than governance that catches 0%.
