# Modern Portfolio Theory Based Portfolio Optimization and Analysis

The project is about the **Modern Portfolio Theory (MPT)** based analysis and optimization of a given portfolio’s returns based on individual investors’ risk tolerance. The project is based on an assumption of a real world utility of **Modern Portfolio Theory (MPT)** for a Portfolio Manager, for example.

**Assume**:
As a Portfolio Manager of a Fintech firm, he/she should be able to:
- Construct a portfolio of assets based on market index to maximize profit returns.
- Identify the optimal weights or mix (percentage of investment) of constituent assets such that risk can be minimized.
- Analyze the market return metrics through statistical visualizations and evidence.

To measure the functioning of the application, let's assume further to check the following conditions.

**Acceptance Criteria**:
- Allow the user to run the application from a Jupyter Notebook.
- Show maximum possible returns, given a certain risk tolerance as defined by the individual investor.
- Show minimum possible risk, given a certain risk tolerance for a target return as desired by the individual investor.

The **Modern Portfolio Theory** has two main fundamentals:
- a portfolio’s total risk and return profile is more important than the risk/return profile of any individual investment, and
- it is possible for an investor to build a diversified portfolio comprising of multiple assets or investments that will maximize returns while limiting risk.

The above points are built upon an understanding of three key concepts:
- the risk-return relationship,
- the role of diversification in building an investment portfolio, and
- the efficient frontier representing the combination offering the best possible expected return for given risk level.

---

## Technologies

This project leverages python 3.7.* with the following additional packages:
* [Jupyter Notebook](https://jupyter.org/) - The main module of the Financial Planner is written in Jupyter Notebook.
* [Conda](https://docs.conda.io/projects/conda/en/latest/) - Conda environment is recommended to have Pandas library and other dependencies pre-installed.

**Required Framework:**

- [??](??) - ??.

**Required Libraries:**

You may need the following libraries to work with the application.

- [Pandas](https://pandas.pydata.org/docs/reference/index.html) - pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive.
- [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html) - NumPy is the fundamental package for scientific computing in Python.
- [yfinance](https://pypi.org/project/yfinance/) - Download market data from Yahoo! Finance's API
- [SQLAlchemy](https://www.sqlalchemy.org/) - SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL.

---

## Usage

To use the application, clone the repository and run the **mpt.ipynb** with Jupyter notebook.

From the Git Bash terminal, make sure to 'activate conda' and appropriate virtual enivorment. Next, launch the JupyterLab web-based interactive development environment (IDE) interface by typing at the prompt:

```python
  > jupyter lab
```

Then, browse to **mpt.ipynb** starter code file to run the program. See the image below for a quick hint.

![Jupyter Notebook](images/app_usage.png)

**Note:** 

## Contributors
Peers of FinTech Blended Boot Camp, Columbia Engineering (2022-23 Batch)

- [Conyea, Will]()
- [Gavnoudias, Stratis](https://github.com/sgavnoudias)
- [Huang, Yan](https://github.com/Shiroyana)
- [Mandal, Dinesh](https://github.com/dinesh-m)
- [Skylizard, Loki 'billie'](https://github.com/Billie-LS)

---

## License

None