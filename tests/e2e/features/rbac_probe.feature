Feature: Sandbox ServiceAccount RBAC probe
  Validates effective authorization of the sandbox e2e ServiceAccount with
  real Kubernetes API calls under that account's credentials.  No RBAC object
  inspection or SubjectAccessReview — only real create/read/denied outcomes.
  Harness: .ai/spec/what/e2e-testing.md

  Scenario: Allowed operation — SA creates and reads a Result CR
    Given the sandbox service account is available
    When the RBAC probe creates and reads an AnalysisResult using SA credentials
    Then the allowed Result CR operations succeed

  Scenario: Denied operation — SA cannot create Secrets
    Given the sandbox service account is available
    When the RBAC probe attempts to create a Secret using SA credentials
    Then the operation is denied with HTTP 403 Forbidden
