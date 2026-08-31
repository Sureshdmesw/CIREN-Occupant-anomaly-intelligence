# Splunk Operationalization

## Dashboard

The project contains the exported Splunk dashboard:

splunk/dashboards/CIREN_Occupant_Positioning_Anomaly_Intelligence.xml

The dashboard provides:

- global severity filtering
- occupant-role filtering
- seat-position filtering
- anomaly KPIs
- severity distribution
- anomaly magnitude analysis
- dominant contributor analysis
- surprisal analysis
- occupant positioning
- posture analysis
- vehicle anomaly severity
- vehicle dominant contributors
- HIGH/CRITICAL investigation queue

## Detection

The Splunk environment contains CIREN scheduled detections for anomalous occupant and vehicle conditions.

The implementation uses scheduled searches and the Splunk triggered-alert mechanism.

## Investigation Workflow

1. Detect an anomalous event.
2. Review anomaly severity and magnitude.
3. Examine occupant positioning and state.
4. Inspect the dominant contributor.
5. Correlate the event at vehicle level.
6. Prioritize the vehicle/event for investigation.

Splunk is the operational detection and investigation layer. The underlying statistical anomaly model is implemented in the Python analytical pipeline.
