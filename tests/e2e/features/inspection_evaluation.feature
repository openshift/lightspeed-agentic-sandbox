Feature: Classifier-only corpus evaluation
  Measures the production tool-result classifier inside the sandbox image,
  separately from main-agent MCP enforcement and without an accuracy threshold.

  Scenario: Evaluate the labeled corpus using only the production classifier
    Given provider credentials are configured
    When I evaluate the labeled corpus with only the production classifier
    Then the corpus evaluation completes without operational errors
