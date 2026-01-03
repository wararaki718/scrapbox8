# Guiding Principles for AI Agents

This repository is a collection of self-contained code samples, organized by topic into top-level directories (e.g., `aws_sample`, `ml_sample`, `ir_sample`, `py_sample`). Each subdirectory is an independent project.

## Key Concepts

- **Project Structure**: Each sample resides in its own directory and is self-contained. There is no shared code or single overarching architecture across the different samples.
- **Primary Documentation**: The `README.md` file within each sample's directory is the most important source of information. It contains specific instructions on how to set up, run, and use the sample.
- **Execution**: Most Python-based samples are executed via simple command-line instructions, such as `python main.py` or `python batch.py`. Always refer to the `README.md` for the exact command.
- **Dependencies**: Dependencies are managed on a per-sample basis. Some samples include a `requirements.txt` file or specify `pip install` commands in their `README.md`.

## Developer Workflow

When working on a specific sample, follow this workflow:

1.  **Navigate to the Sample Directory**: All operations should be performed from within the relevant sample's directory.
2.  **Consult the README**: Before making any changes, thoroughly read the `README.md` file in that directory. It will provide context and instructions.
3.  **Install Dependencies**: If the `README.md` or a `requirements.txt` file specifies dependencies, install them first. For example, in `ir_sample/simple-qdrant-sparse-retrieval`, you need to run `pip install qdrant-client==1.16.1 git+https://github.com/bizreach-inc/light-splade.git`.
4.  **Run the Sample**: Use the command specified in the `README.md` to run the code. For example, to run the `sample-hybrid-hyde-mqd` sample, you might use `python batch.py` for batch processing or `python app.py` to run the application.

## Examples

-   **AWS Samples (`aws_sample`)**: These samples demonstrate how to use AWS services, often with LocalStack for local development. They include detailed setup instructions for tools like the AWS SAM CLI and commands for interacting with services. See [aws_sample/use-localstack/README.md](aws_sample/use-localstack/README.md).
-   **Machine Learning Samples (`ml_sample`)**: These samples cover a wide range of ML topics. Each is a focused implementation of a specific algorithm or technique.
-   **Information Retrieval Samples (`ir_sample`)**: These samples explore different information retrieval methods. For instance, `sample-hybrid-hyde-mqd` in [ir_sample/sample-hybrid-hyde-mqd/README.md](ir_sample/sample-hybrid-hyde-mqd/README.md) provides instructions for running a web application and a batch process.

By treating each sample as a distinct mini-project and prioritizing its `README.md`, you will be able to navigate and contribute to this repository effectively.
