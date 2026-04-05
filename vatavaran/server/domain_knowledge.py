"""Bank-domain schema and candidate values for Vatavaran tasks."""

from __future__ import annotations

POSSIBLE_REASONS = [
    "high CPU usage",
    "high memory usage",
    "network latency",
    "network packet loss",
    "high disk I/O read usage",
    "high disk space usage",
    "high JVM CPU load",
    "JVM Out of Memory (OOM) Heap",
]

POSSIBLE_COMPONENTS = [
    "apache01",
    "apache02",
    "Tomcat01",
    "Tomcat02",
    "Tomcat03",
    "Tomcat04",
    "MG01",
    "MG02",
    "IG01",
    "IG02",
    "Mysql01",
    "Mysql02",
    "Redis01",
    "Redis02",
]

SCHEMA_TEXT = """## TELEMETRY DIRECTORY STRUCTURE

- Telemetry root: `data/telemetry/Bank/`
- Date partitions: `.../Bank/{YYYY_MM_DD}/`
- Per-date folders: `metric/`, `trace/`, `log/`
- CSV files:
  - metric_app.csv
  - metric_container.csv
  - trace_span.csv
  - log_service.csv

## DATA SCHEMA

1) Metric files:
- metric_app.csv: `timestamp,rr,sr,cnt,mrt,tc`
- metric_container.csv: `timestamp,cmdb_id,kpi_name,value`

2) Trace files:
- trace_span.csv: `timestamp,cmdb_id,parent_id,span_id,trace_id,duration`

3) Log files:
- log_service.csv: `log_id,timestamp,cmdb_id,log_name,value`

## TIMESTAMP NOTES

- metric timestamps are in seconds.
- trace timestamps are in milliseconds.
- log timestamps are in seconds.
- Use UTC+8 (`Asia/Shanghai`) when interpreting wall-clock time.
"""


def get_domain_knowledge() -> str:
    """Return a single prompt-style domain knowledge block."""

    reasons = "\n".join([f"- {item}" for item in POSSIBLE_REASONS])
    components = "\n".join([f"- {item}" for item in POSSIBLE_COMPONENTS])
    return (
        f"{SCHEMA_TEXT}\n\n## POSSIBLE ROOT CAUSE REASONS\n{reasons}\n\n"
        f"## POSSIBLE ROOT CAUSE COMPONENTS\n{components}"
    )
