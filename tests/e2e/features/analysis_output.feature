Feature: Operator analysis output with find-token skill
  Verifies: operator AnalysisResult status shape with components (eval parity)
  The agent discovers the find-token skill, runs its script, and returns
  structured analysis with tokens under options[].components[].

  Scenario: Find-token skill returns analysis with component tokens
    Given the sandbox service is running with skills
    And the find-token analysis query and schema have been prepared
    When I run the agent with the prepared find-token analysis query
    Then the run completes successfully
    And the analysis status validates against the operator components schema
    And actionRequired is True
    And the response contains DIAG and VERIFY tokens in component tokens
    And the first analysis option has remediation and component audit fields
