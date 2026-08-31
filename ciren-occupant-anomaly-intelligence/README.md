# CIREN Occupant Positioning & Anomaly Intelligence

A dependency-aware occupant-state anomaly intelligence pipeline built from CIREN public crash data and operationalized in Splunk.

## Pipeline

CIREN data
-> normalization
-> occupant-state modeling
-> dependency modeling
-> conditional probability estimation
-> information-theoretic surprisal
-> anomaly scoring
-> explainability
-> vehicle aggregation
-> Splunk ingestion
-> detection
-> investigation dashboard
-> alerting

## Capabilities

- CIREN data preparation and normalization
- Occupant positioning and state modeling
- Dependency-aware conditional probability modeling
- Information-theoretic anomaly detection
- Surprisal-based anomaly scoring
- Contribution-based explainability
- Vehicle-level anomaly aggregation
- Splunk event generation
- Scheduled anomaly detection
- Investigation workflow
- Operational dashboarding
- Validation and temporal auditing

## Repository Structure

ciren-occupant-anomaly-intelligence/
|
+-- src/
|   +-- 01_data_preparation/
|   +-- 02_state_modeling/
|   +-- 03_dependency_model/
|   +-- 04_information_theory/
|   +-- 05_anomaly_detection/
|   +-- 06_vehicle_analysis/
|   +-- 07_splunk/
|
+-- validation/
+-- splunk/
|   +-- alerts/
|   +-- searches/
|   +-- dashboards/
|   +-- configuration/
|
+-- docs/
+-- evidence/

## Detection Method

For an observed state x, surprisal is defined as:

S(x) = -log2 P(x)

The dependency-aware formulation evaluates conditional state probability:

S(x_i | parents(x_i)) = -log2 P(x_i | parents(x_i))

This allows unusual occupant configurations to be evaluated while accounting for learned dependencies between state variables.

## Splunk

The project includes the exported CIREN Splunk dashboard:

CIREN Occupant Positioning & Anomaly Intelligence

The dashboard provides anomaly KPIs, severity distributions, occupant positioning analysis, explainability, vehicle-level intelligence, and investigation queues.

## Validation

Final Splunk evidence contains:

- 10,000 occupant-state events
- 10,000 unique event IDs
- 10,000 unique occupants
- 9,939 unique vehicles
- 283 HIGH occupant events
- 33 CRITICAL occupant events
- 33 CRITICAL vehicles
- maximum vehicle anomaly index: 100
- maximum observed surprisal: approximately 26.75 bits

## Limitations

This is an analytical and detection prototype based on CIREN public data. Statistical anomaly detection should not be interpreted as causal inference or as a production automotive safety determination.

Production deployment would require additional domain validation, governance, model monitoring, streaming infrastructure, security controls, and safety validation.

## Status

Completed analytical prototype with Splunk operationalization, detection, alerting, and investigation dashboard.
