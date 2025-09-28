# 🚚 Supply Chain Analytics & Machine Learning

A comprehensive supply chain analytics framework featuring machine learning models, interactive visualizations, and optimization tools for demand forecasting, supplier performance, inventory management, and logistics efficiency.

## 📊 Features

### 🤖 Machine Learning Models
- **Demand Forecasting** - Random Forest, Gradient Boosting, Linear Regression
- **Supplier Optimization** - Performance scoring and clustering
- **Inventory Management** - ABC/XYZ analysis and stock optimization
- **Logistics Analysis** - Route efficiency and delivery performance

### 📈 Analytics Capabilities
- **Exploratory Data Analysis** - Comprehensive EDA on all datasets
- **Interactive Visualizations** - Plotly charts and dashboards
- **Performance Metrics** - KPI tracking and insights generation
- **Strategic Recommendations** - Actionable optimization insights

### 🛠️ Technical Stack
- **Python 3.11+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Plotly & Matplotlib** - Interactive and static visualizations
- **Jupyter Notebook** - Interactive analysis environment

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Azeemshah99/project-predictive.git
cd project-predictive
```

2. **Create virtual environment**
```bash
python -m venv supply_chain_env
source supply_chain_env/bin/activate  # On Windows: supply_chain_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run analysis**
```bash
python supply_chain_analysis.py
```

### Interactive Analysis
```bash
jupyter notebook supply_chain_analysis.ipynb
```

## 📁 Project Structure

```
project-predictive/
├── datasets/                          # Data files (excluded from git)
│   ├── supply_chain_sample.csv       # Sample order transactions
│   ├── supplier_performance.csv      # Supplier evaluation data
│   ├── demand_forecasting.csv        # Time series demand data
│   ├── logistics_transport.csv       # Shipping and delivery data
│   ├── inventory_management.csv      # Stock levels and ABC/XYZ data
│   └── README.md                     # Dataset documentation
├── supply_chain_analysis.py          # Main analysis script
├── supply_chain_analysis.ipynb       # Interactive Jupyter notebook
├── run_analysis.py                   # Quick execution script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 📊 Datasets

### 1. **Orders Data** (`supply_chain_sample.csv`)
- Transaction records with costs, suppliers, and warehouse info
- 20 sample orders with 22 features
- Includes product details, pricing, and delivery metrics

### 2. **Supplier Performance** (`supplier_performance.csv`)
- Quality ratings, delivery performance, and risk assessments
- 10 suppliers across multiple countries
- Cost competitiveness and sustainability metrics

### 3. **Demand Forecasting** (`demand_forecasting.csv`)
- Time series data with actual vs forecasted demand
- 25 time periods with seasonal and economic factors
- Marketing spend and competitor pricing data

### 4. **Logistics** (`logistics_transport.csv`)
- Transportation and delivery performance data
- 20 shipments with route efficiency metrics
- Weather conditions and customer satisfaction scores

### 5. **Inventory Management** (`inventory_management.csv`)
- Stock levels, reorder points, and turnover rates
- ABC/XYZ classification for optimization
- 19 product-warehouse combinations

## 🔧 Usage Examples

### Basic Analysis
```python
from supply_chain_analysis import SupplyChainAnalyzer

# Initialize analyzer
analyzer = SupplyChainAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()
```

### Demand Forecasting
```python
# Load demand data
import pandas as pd
demand = pd.read_csv('datasets/demand_forecasting.csv')

# Train ML models
analyzer.demand_forecasting_analysis()
```

### Supplier Optimization
```python
# Analyze supplier performance
analyzer.supplier_optimization_analysis()
```

## 📈 Key Insights Generated

### Financial Metrics
- Total order value and average costs
- Cost analysis by category and supplier
- Inventory value and carrying costs

### Performance Indicators
- On-time delivery rates
- Forecast accuracy percentages
- Customer satisfaction scores
- Supplier quality ratings

### Optimization Opportunities
- Overstocked and understocked items
- High-risk suppliers identification
- Route efficiency improvements
- Cost reduction opportunities

## 🎯 Use Cases

### For Data Scientists
- **Machine Learning** - Implement and compare ML models
- **Feature Engineering** - Create new predictive features
- **Model Evaluation** - Assess model performance and accuracy

### For Supply Chain Managers
- **Demand Planning** - Improve forecast accuracy
- **Supplier Management** - Optimize supplier selection
- **Inventory Optimization** - Reduce stockouts and overstock
- **Cost Reduction** - Identify cost-saving opportunities

### For Business Analysts
- **Performance Monitoring** - Track KPIs and metrics
- **Strategic Planning** - Data-driven decision making
- **Risk Assessment** - Identify and mitigate risks

## 🔍 Analysis Components

### 1. **Exploratory Data Analysis (EDA)**
- Dataset overview and statistics
- Missing value analysis
- Data type validation
- Key insights extraction

### 2. **Demand Forecasting**
- Multiple ML model comparison
- Feature importance analysis
- Forecast accuracy evaluation
- Time series analysis

### 3. **Supplier Optimization**
- Composite scoring system
- Risk assessment and clustering
- Cost vs quality analysis
- Performance ranking

### 4. **Inventory Management**
- ABC/XYZ classification
- Stock optimization recommendations
- Turnover rate analysis
- Reorder point optimization

### 5. **Logistics Analysis**
- Delivery performance metrics
- Route efficiency evaluation
- Cost analysis by transport mode
- Weather impact assessment

## 📊 Visualizations

The framework generates comprehensive visualizations including:
- Cost distribution histograms
- Supplier performance scatter plots
- Demand forecast accuracy charts
- ABC/XYZ classification pie charts
- Delivery performance box plots
- Route efficiency analysis

## 🛠️ Customization

### Adding New Datasets
1. Place CSV files in the `datasets/` directory
2. Update the `load_data()` method in `SupplyChainAnalyzer`
3. Add analysis methods for new data types

### Modifying ML Models
1. Edit the `demand_forecasting_analysis()` method
2. Add new algorithms to the models dictionary
3. Adjust feature selection and preprocessing

### Creating Custom Visualizations
1. Add new plotting functions to the class
2. Use Plotly for interactive charts
3. Customize styling and layout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Azeemshah Amiruddin**
- GitHub: [@Azeemshah99](https://github.com/Azeemshah99)
- Email: azeemshah.amiruddin@gmail.com

## 🙏 Acknowledgments

- Supply chain datasets for analysis
- Open source Python libraries
- Machine learning community resources

## 📞 Support

If you have any questions or need help with the project, please:
- Open an issue on GitHub
- Contact the author directly
- Check the documentation in the `datasets/README.md`

---

**Happy Analyzing! 🚀📊**
