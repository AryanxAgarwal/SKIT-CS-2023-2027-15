\# SYSTEM ARCHITECTURE AND QUALITY ASSURANCE REPORT



\*\*Project:\*\* SKIT-CS-2023-2027-15  

\*\*Project Title:\*\* Driver Drowsiness Detection System  

\*\*Prepared by:\*\* Chiranshu Kumar Mudgal  

\*\*Role:\*\* Systems Architecture and Quality Assurance  

\*\*Reporting Period:\*\* Sprint 1 – Sprint 3 (Up to 12 November 2026)  

\*\*Report Date:\*\* 16 November 2026



\---



\## 1. Introduction



This report documents the system architecture, validation, model conversion, runtime testing and quality assurance activities performed for the Driver Drowsiness Detection System.



The primary objective was to verify the reliability of data processing, evaluate model performance, validate deployment formats and develop independent runtime quality assurance components.



\## 2. Sprint 1 – Baseline Integration and Data Flow Verification



\### Activities performed



\- Reviewed the existing frame ingestion validation pipeline.

\- Examined the automated GitHub Actions testing workflow.

\- Verified dataset frame consistency and validation checks.

\- Executed existing unit tests for EAR and MAR calculations.

\- Reviewed the feature and label data pipeline.



\### Verification results



\- Frame ingestion tests: 4 passed.

\- EAR and MAR tests: 6 passed.

\- Total existing tests: 10 passed.



\*\*Status:\*\* Verified through local test execution. Latest GitHub Actions execution was not independently confirmed.



\## 3. Sprint 2 – Model Validation and Deployment Evaluation



\### 3.1 Benchmark Data Batch Management



\- Executed benchmark dataset validation.

\- Verified 383 dataset samples.

\- Checked manifest consistency and input dimensions.

\- Checked duplicate records, label mismatches and split leakage.

\- Verified hold-out clips and class representation.



\### 3.2 Model Validation Framework



\- Executed the evaluation framework.

\- Verified model input compatibility.

\- Reviewed training loss history and evaluation metrics.

\- Generated validation logs and reports.



\### 3.3 Model Serialization and Conversion



Validated the availability and loading of:



\- Keras model checkpoint.

\- ONNX converted model.

\- TFLite Float32 model.

\- TFLite INT8 model.



ONNX and TFLite model input compatibility was checked using the available eight-feature training input format.



\### 3.4 Quantization Evaluation



The existing quantization report showed the following results:



| Model | Accuracy | F1 Score |

|---|---:|---:|

| Keras Float32 | 100% | 100% |

| TFLite Float32 | 100% | 100% |

| TFLite INT8 | 81.82% | 81.82% |



The INT8 model showed an 18.18 percentage-point reduction in accuracy and F1 score compared with Float32.



\*\*Deployment decision:\*\* Float32 is the preferred candidate. INT8 requires further optimization and validation before deployment approval.



Note: These reported evaluation metrics use the existing limited test dataset and should not be interpreted as evidence of general performance across diverse drivers.



\## 4. Sprint 3 – Runtime Architecture and Quality Assurance



\### 4.1 Runtime State Machine



Developed an independent state machine with four states:



\- INIT

\- MONITORING

\- ALERT

\- ERROR



Implemented transition validation, state history and error recovery.



\*\*Verification:\*\* 10 unit tests passed.



\### 4.2 Parallel Processing Architecture



Developed an independent runtime pipeline containing:



\- Video capture worker.

\- Model inference worker.

\- Bounded frame queue.

\- Bounded result queue.

\- Error reporting.

\- Controlled thread shutdown.



\*\*Verification:\*\* 9 unit tests passed after an additional post-deadline validation improvement.



\### 4.3 Runtime Helper and Alarm Validation



Tested numerical input handling, eight-feature vector generation and background alarm execution.



\*\*Verification:\*\* 5 unit tests passed.



\### 4.4 Thread Stability Testing



Performed simulated high-load and repeated lifecycle tests.



Verified:



\- Bounded queue capacity.

\- Repeated start-stop cycles.

\- Worker exception handling.

\- Thread termination.



\*\*Verification:\*\* 4 unit tests passed.



\### 4.5 End-to-End Software Integration Testing



Executed simulated integration tests covering:



\- Frame generation.

\- Frame queue processing.

\- Inference result generation.

\- State transition to ALERT.

\- Error reporting and transition to ERROR.

\- Controlled shutdown.



\*\*Verification:\*\* 2 integration tests passed.



\## 5. Overall Test Summary



The following results were obtained through separate local test suite executions.



| Test suite | Passed |

|---|---:|

| Runtime helper and alarm | 5 |

| State machine | 10 |

| Threading pipeline | 9 |

| Thread stability | 4 |

| Software integration | 2 |

| Existing project tests | 10 |

| \*\*Total\*\* | \*\*40\*\* |



All 40 tests passed across the recorded local executions, with no test failures.



\## 6. Post-Deadline QA Improvement – 14 November 2026



After the reporting deadline, an additional improvement was made to the independent runtime threading module.



\### 6.1 Capture Worker Optimization



The capture worker was updated to introduce a short 10 ms wait when no video frame is available.



This prevents unnecessary continuous looping when the capture function returns None and helps reduce avoidable CPU usage.



\### 6.2 Additional Test Validation



A dedicated test, `test\_none\_frame\_does\_not\_busy\_spin`, was added to verify this behaviour.



After the update:



\- Runtime threading tests: 9 passed.

\- Complete evaluation test suite: 30 passed.

\- Existing project test suite: 10 passed.



\*\*Status:\*\* Post-deadline improvement verified through local automated tests.



\## 7. Known Limitations and Pending Validation



The following activities require further validation:



1\. Integration of the independent threading pipeline with the actual live application.

2\. Connection of the new state machine to the production control loop.

3\. End-to-end testing using an actual webcam and trained CNN model with the integrated threading pipeline.

4\. Prolonged hardware stability testing.

5\. Memory usage and deadlock profiling under real runtime load.

6\. Further investigation of INT8 accuracy degradation.

7\. Verification of the latest GitHub Actions workflow execution.



The simulated integration tests do not establish successful integration with the actual camera, CNN and user interface.



\## 8. Conclusion



The baseline data validation, model evaluation and conversion checks have been executed. Independent runtime architecture and quality assurance modules have also been developed and tested.



A total of 40 local unit and integration tests passed across separate test suite executions.



An additional capture worker optimization was implemented and verified after the reporting deadline.



Further production integration and hardware-level validation remain necessary before declaring the complete runtime architecture fully validated.



\---



\*\*Prepared by:\*\* Chiranshu Kumar Mudgal  

\*\*Role:\*\* Systems Architecture and Quality Assurance  

\*\*Project:\*\* Driver Drowsiness Detection System  

\*\*Institution:\*\* Swami Keshvanand Institute of Technology, Management \& Gramothan, Jaipur

