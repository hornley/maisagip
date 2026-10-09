# Inspection Variety Display Names Design

## Goal

Show friendly variety names in the inspection results while preserving the classifier's canonical class IDs.

## Design

Add a small frontend display-name mapping for `yellow_sweet_corn` → `Sweet Fortune` and `white_corn` → `Sweet Pearl`. Use it in the inspection summary badge and per-view captions. Keep API values, model classes, Dataset labels, annotations, and stored reports unchanged.

## Verification

Check that both mapped names appear in the inspection summary and per-view captions, unknown class IDs still render as readable text, and existing API values remain canonical.

## Scope

This changes only user-facing variety text in the Inspect results. It does not rename dataset classes or alter training or inference.
