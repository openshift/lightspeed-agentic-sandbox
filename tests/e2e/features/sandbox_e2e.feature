Feature: Sandbox E2E contract
  Verifies: .ai/spec/what/run-api.md (context), .ai/spec/what/audit-logging.md (OTLP traces and audit logs)
  Harness: .ai/spec/what/e2e-testing.md
  Live batch Job tests on a cluster (see scripts/e2e-containers.sh). Exact context prefix
  formatting: test_run_agent.py. Run error rules 22–23: structured_output.feature.

  Scenario: Target namespaces from context reach the model
    Given the sandbox service is running
    And a context with target namespaces and an echo output schema have been prepared
    When I run the agent with the prepared context and schema
    Then the run completes successfully
    And success is true
    And the response namespaces field matches the prepared context

  Scenario: Previous attempts from context reach the model
    Given the sandbox service is running
    And a context with previous attempts and an echo output schema have been prepared
    When I run the agent with the prepared context and schema
    Then the run completes successfully
    And success is true
    And the response first failure reason matches the prepared context

  Scenario: Batch run exports traces and audit logs to OTEL
    Given provider credentials are configured
    And the sandbox service is running
    And the OTEL collector is available for telemetry verification
    And a simple non-skill query has been prepared
    When I run the agent with the prepared query and no output schema
    Then the run completes successfully
    And the OTEL collector received traces for the batch run
    # Also validates paired LLM requests and delivery of auth JSON records from pod stdout.
    And the OTEL collector received audit logs with agenticrun attributes

  Scenario: Approved option from context reaches the model
    Given the sandbox service is running
    And a context with approved option and an echo output schema have been prepared
    When I run the agent with the prepared context and schema
    Then the run completes successfully
    And success is true
    And the response approved option fields match the prepared context
