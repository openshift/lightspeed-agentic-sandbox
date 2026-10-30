Feature: Tool-result safety inspection
  Verifies that a focused batch e2e run can enable tool-result inspection without
  changing the default for the rest of the e2e suite.

  Scenario: MCP tool output is inspected before the model uses it
    Given tool-result inspection is enabled for the batch Job
    And the OTEL collector is available for telemetry verification
    And the sandbox service is running with MCP servers configured
    And an MCP tool invocation query has been prepared
    When I run the agent with the prepared schema and query
    Then the run completes successfully
    And success is true
    And the response JSON validates against the output schema
    And the response summary contains the sentinel namespace from the tool
    And the OTEL collector received a tool-result inspection span

  Scenario: Malicious MCP tool output is rejected before model use
    Given tool-result inspection is enabled for the batch Job
    And the OTEL collector is available for telemetry verification
    And the sandbox service is running with MCP servers configured
    And a malicious MCP tool-result query has been prepared
    When I run the agent with the prepared schema and query
    Then the run fails closed with a tool-result safety inspection error
    And no Result CR was published for the run
    And the OTEL collector received a malicious tool-result inspection span
