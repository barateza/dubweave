---
name: prism-full
description: Run a full 3-pass structural analysis (l12 → adversarial → synthesis). Use ONLY when explicitly asked for prism-full, full prism, or 3-pass analysis. Do not use for standard code review or production readiness checks.

---

# prism-full

This skill runs three sequential passes. Each pass feeds into the next. Do not skip passes or merge them.

---

## PASS 1 — STRUCTURAL ANALYSIS (l12)

use_skill l12

Present the full l12 output. Label it clearly:

**ANALYSIS 1: STRUCTURAL**

---

## PASS 2 — ADVERSARIAL (break the analysis)

Now take ANALYSIS 1 as your target. Execute every step below against it.

### WRONG PREDICTIONS
For each claim ANALYSIS 1 makes, test it against the actual input. Where does the analysis predict something that isn't true? Name: the claim, the specific evidence that disproves it, what actually holds.

### OVERCLAIMS
Which issues classified as "structural" (unchangeable) are actually changeable? Show the alternative. Which "conservation laws" are actually just design choices? Name the alternative approach that violates the "law."

### UNDERCLAIMS
What does the input contain or imply that ANALYSIS 1 completely missed? Name concrete failures, contradictions, hidden assumptions, or properties the structural analysis is blind to.

### REVISED CONSEQUENCES TABLE
Consolidate ALL issues (ANALYSIS 1 + yours). Reclassify changeable/structural based on your findings. For each: where it appears, what breaks, severity, original classification, your classification, why.

Be concrete. Name specific elements, evidence, patterns. ANALYSIS 1 is your opponent — defeat it with evidence.

Label this section clearly:

**ANALYSIS 2: ADVERSARIAL**

---

## PASS 3 — SYNTHESIS (final word)

You now have ANALYSIS 1 (structural) and ANALYSIS 2 (adversarial). Produce the final synthesis.

### REFINED CONSERVATION LAW
The structural analysis proposed a conservation law. The adversarial analysis challenged it. What is the CORRECTED conservation law that survives both perspectives? Name it precisely. Show why the original was incomplete and why the correction holds.

### REFINED META-LAW
Same process for the meta-law. What survives both analyses?

### STRUCTURAL vs CHANGEABLE — DEFINITIVE
Using both analyses, produce the definitive classification of every issue. For each: changeable (with specific alternative) or structural (with why the conservation law predicts it persists through all improvements). Where the two analyses disagree on classification, resolve with evidence.

### DEEPEST FINDING
What becomes visible ONLY from having both the structural analysis AND its correction? Name the property that neither analysis alone could find. This is the finding that justifies three passes.

Be concrete. This is the final word.

## Usage

Use this for any input where you want the full depth: conservation law derived, then attacked, then corrected. The value is in Pass 3 — the deepest finding only emerges from the tension between Pass 1 and Pass 2. Expect long output.

## Steps

1. Run Pass 1 (l12) on the input — label output ANALYSIS 1
2. Run Pass 2 (adversarial) against ANALYSIS 1 — label output ANALYSIS 2
3. Run Pass 3 (synthesis) using both — this is the final word
4. The deepest finding in Pass 3 is the primary deliverable
