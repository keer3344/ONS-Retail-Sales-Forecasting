# Contributing to ONS Retail Sales Forecasting

Thank you for your interest in contributing to this project! We welcome contributions from the community to help improve this time series forecasting analysis.

## Code of Conduct

Please be respectful and constructive in all interactions with other contributors.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ONS-Retail-Sales-Forecasting.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (on Windows: `venv\\Scripts\\activate`)
5. Install dependencies: `pip install -r requirements.txt`

## Development Workflow

### Making Changes

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes to the code
3. Write or update tests to cover your changes
4. Run tests: `python -m pytest test_models.py`
5. Ensure code follows PEP 8 standards

### Commit Messages

Write clear and descriptive commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- Keep messages concise but informative
- Reference issue numbers if applicable: "Fix #123"

### Submitting a Pull Request

1. Push your branch to GitHub
2. Create a Pull Request with a clear title and description
3. Link any related issues
4. Ensure all tests pass
5. Request review from maintainers

## Code Guidelines

### Python Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

### Model Development

- Document model parameters and assumptions
- Include evaluation metrics (MAE, MSE, RMSE, MAPE)
- Add comments explaining complex algorithms
- Test with multiple datasets when possible
- Validate results against established baselines

### Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for model pipelines

## Types of Contributions

### Bug Reports

Report bugs by creating an issue with:
- Clear title describing the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and system information
- Error messages and traceback

### Feature Requests

Suggest features by creating an issue with:
- Clear description of the feature
- Use case and motivation
- Possible implementation approach
- Any relevant research or references

### Documentation

- Improve README clarity
- Add examples and tutorials
- Document model architectures
- Fix typos and improve language
- Add API documentation

### Model Improvements

- Implement new forecasting models
- Improve existing model performance
- Add hyperparameter tuning methods
- Enhance evaluation metrics
- Optimize computational efficiency

## Project Structure

```
.
├── README.md                 # Project documentation
├── CONTRIBUTING.md          # This file
├── requirements.txt         # Python dependencies
├── main.py                 # Main forecasting models
├── test_models.py          # Unit tests
└── .gitignore             # Git ignore file
```

## Common Issues

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Data Loading Issues

The project attempts to fetch data from ONS API. If this fails, synthetic data is generated for testing.

### Model Training Issues

Ensure you have sufficient memory and time for model training. Start with smaller datasets or reduced parameters.

## Getting Help

- Check existing issues and discussions
- Review the project README
- Create a new issue with detailed information
- Reach out to maintainers

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to this project!
