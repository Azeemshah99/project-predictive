#!/usr/bin/env python3
"""
Supply Chain Analytics and Machine Learning
Comprehensive analysis of supply chain datasets for optimization and forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SupplyChainAnalyzer:
    def __init__(self, data_path='datasets/'):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.data = {}
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load all supply chain datasets"""
        print("Loading supply chain datasets...")
        
        # Load all CSV files
        datasets = {
            'orders': 'supply_chain_sample.csv',
            'suppliers': 'supplier_performance.csv', 
            'demand': 'demand_forecasting.csv',
            'logistics': 'logistics_transport.csv',
            'inventory': 'inventory_management.csv'
        }
        
        for name, filename in datasets.items():
            try:
                self.data[name] = pd.read_csv(f"{self.data_path}{filename}")
                print(f"✓ Loaded {name}: {len(self.data[name])} records")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        
        print(f"\nTotal datasets loaded: {len(self.data)}")
        return self.data
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA on all datasets"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        for name, df in self.data.items():
            print(f"\n📊 {name.upper()} DATASET ANALYSIS")
            print("-" * 40)
            
            # Basic info
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Missing values: {df.isnull().sum().sum()}")
            
            # Data types
            print(f"\nData Types:")
            print(df.dtypes)
            
            # Statistical summary
            print(f"\nStatistical Summary:")
            print(df.describe())
            
            # Key insights
            if 'total_cost' in df.columns:
                print(f"\n💰 Cost Analysis:")
                print(f"Total cost: ${df['total_cost'].sum():,.2f}")
                print(f"Average cost per order: ${df['total_cost'].mean():,.2f}")
                print(f"Cost range: ${df['total_cost'].min():,.2f} - ${df['total_cost'].max():,.2f}")
            
            if 'quality_rating' in df.columns:
                print(f"\n⭐ Quality Analysis:")
                print(f"Average quality rating: {df['quality_rating'].mean():.2f}")
                print(f"Best supplier: {df.loc[df['quality_rating'].idxmax(), 'supplier_name']}")
            
            if 'actual_demand' in df.columns:
                print(f"\n📈 Demand Analysis:")
                print(f"Average demand: {df['actual_demand'].mean():.2f}")
                print(f"Demand variance: {df['actual_demand'].var():.2f}")
                if 'forecasted_demand' in df.columns:
                    print(f"Forecast accuracy: {1 - abs(df['actual_demand'] - df['forecasted_demand']).mean() / df['actual_demand'].mean():.2%}")
    
    def demand_forecasting_analysis(self):
        """Build demand forecasting ML models"""
        print("\n" + "="*60)
        print("DEMAND FORECASTING ANALYSIS")
        print("="*60)
        
        if 'demand' not in self.data:
            print("❌ Demand dataset not available")
            return
        
        df = self.data['demand'].copy()
        
        # Prepare features for ML
        feature_columns = ['price', 'competitor_price', 'marketing_spend', 'economic_index', 
                          'weather_factor', 'trend_factor', 'seasonality_factor']
        
        # Handle missing values
        df = df.dropna()
        
        X = df[feature_columns]
        y = df['actual_demand']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        print("🤖 Training demand forecasting models...")
        
        for name, model in models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n{name} Performance:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  R²: {r2:.3f}")
            
            self.models[f'demand_{name.lower().replace(" ", "_")}'] = model
        
        # Feature importance
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Feature Importance (Random Forest):")
        print(feature_importance)
        
        return models
    
    def supplier_optimization_analysis(self):
        """Analyze supplier performance and optimization opportunities"""
        print("\n" + "="*60)
        print("SUPPLIER OPTIMIZATION ANALYSIS")
        print("="*60)
        
        if 'suppliers' not in self.data:
            print("❌ Supplier dataset not available")
            return
        
        df = self.data['suppliers'].copy()
        
        # Supplier scoring
        print("📊 Supplier Performance Scoring:")
        
        # Calculate composite score
        df['composite_score'] = (
            df['on_time_delivery_rate'] * 0.3 +
            df['quality_rating'] * 0.25 +
            df['cost_competitiveness'] * 0.2 +
            df['reliability_score'] * 0.15 +
            df['sustainability_rating'] * 0.1
        )
        
        # Rank suppliers
        df_sorted = df.sort_values('composite_score', ascending=False)
        
        print(f"\n🏆 Top 3 Suppliers:")
        for i, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
            print(f"{i}. {row['supplier_name']} - Score: {row['composite_score']:.3f}")
        
        print(f"\n⚠️ Risk Analysis:")
        high_risk = df[df['risk_level'] == 'High']
        print(f"High-risk suppliers: {len(high_risk)}")
        if len(high_risk) > 0:
            print("High-risk suppliers:", high_risk['supplier_name'].tolist())
        
        # Cost vs Quality analysis
        print(f"\n💰 Cost vs Quality Analysis:")
        cost_quality = df.groupby('country').agg({
            'cost_competitiveness': 'mean',
            'quality_rating': 'mean'
        }).round(3)
        print(cost_quality)
        
        return df_sorted
    
    def inventory_optimization_analysis(self):
        """Analyze inventory management and optimization"""
        print("\n" + "="*60)
        print("INVENTORY OPTIMIZATION ANALYSIS")
        print("="*60)
        
        if 'inventory' not in self.data:
            print("❌ Inventory dataset not available")
            return
        
        df = self.data['inventory'].copy()
        
        # ABC Analysis
        print("📊 ABC Analysis:")
        abc_counts = df['abc_classification'].value_counts()
        print(abc_counts)
        
        # XYZ Analysis
        print(f"\n📈 XYZ Analysis:")
        xyz_counts = df['xyz_classification'].value_counts()
        print(xyz_counts)
        
        # Combined ABC-XYZ Analysis
        print(f"\n🎯 ABC-XYZ Matrix:")
        abc_xyz = pd.crosstab(df['abc_classification'], df['xyz_classification'])
        print(abc_xyz)
        
        # Stock optimization recommendations
        print(f"\n💡 Stock Optimization Recommendations:")
        
        # Overstocked items
        overstocked = df[df['current_stock'] > df['max_stock_level']]
        if len(overstocked) > 0:
            print(f"⚠️ Overstocked items: {len(overstocked)}")
            print(overstocked[['product_name', 'current_stock', 'max_stock_level']].head())
        
        # Understocked items
        understocked = df[df['current_stock'] < df['min_stock_level']]
        if len(understocked) > 0:
            print(f"\n📉 Understocked items: {len(understocked)}")
            print(understocked[['product_name', 'current_stock', 'min_stock_level']].head())
        
        # High turnover items
        high_turnover = df[df['stock_turnover_rate'] > df['stock_turnover_rate'].quantile(0.75)]
        print(f"\n🚀 High turnover items: {len(high_turnover)}")
        print(high_turnover[['product_name', 'stock_turnover_rate']].head())
        
        return df
    
    def logistics_efficiency_analysis(self):
        """Analyze logistics and transportation efficiency"""
        print("\n" + "="*60)
        print("LOGISTICS EFFICIENCY ANALYSIS")
        print("="*60)
        
        if 'logistics' not in self.data:
            print("❌ Logistics dataset not available")
            return
        
        df = self.data['logistics'].copy()
        
        # Delivery performance
        print("📦 Delivery Performance:")
        on_time = df[df['delay_days'] <= 0]
        print(f"On-time deliveries: {len(on_time)}/{len(df)} ({len(on_time)/len(df)*100:.1f}%)")
        
        avg_delay = df['delay_days'].mean()
        print(f"Average delay: {avg_delay:.1f} days")
        
        # Cost analysis by transport mode
        print(f"\n💰 Cost Analysis by Transport Mode:")
        cost_by_mode = df.groupby('transport_mode').agg({
            'total_cost': ['mean', 'sum'],
            'distance_km': 'mean',
            'route_efficiency': 'mean'
        }).round(2)
        print(cost_by_mode)
        
        # Carrier performance
        print(f"\n🚚 Carrier Performance:")
        carrier_perf = df.groupby('carrier_name').agg({
            'delay_days': 'mean',
            'customer_satisfaction': 'mean',
            'route_efficiency': 'mean',
            'total_cost': 'mean'
        }).round(2)
        print(carrier_perf.sort_values('customer_satisfaction', ascending=False))
        
        # Weather impact
        print(f"\n🌤️ Weather Impact Analysis:")
        weather_impact = df.groupby('weather_conditions').agg({
            'delay_days': 'mean',
            'route_efficiency': 'mean',
            'customer_satisfaction': 'mean'
        }).round(2)
        print(weather_impact)
        
        return df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Cost distribution
        if 'orders' in self.data:
            plt.subplot(3, 3, 1)
            self.data['orders']['total_cost'].hist(bins=20, alpha=0.7)
            plt.title('Order Cost Distribution')
            plt.xlabel('Total Cost ($)')
            plt.ylabel('Frequency')
        
        # 2. Supplier quality vs cost
        if 'suppliers' in self.data:
            plt.subplot(3, 3, 2)
            plt.scatter(self.data['suppliers']['cost_competitiveness'], 
                       self.data['suppliers']['quality_rating'], 
                       alpha=0.7, s=100)
            plt.xlabel('Cost Competitiveness')
            plt.ylabel('Quality Rating')
            plt.title('Supplier: Cost vs Quality')
        
        # 3. Demand vs Forecast
        if 'demand' in self.data:
            plt.subplot(3, 3, 3)
            plt.scatter(self.data['demand']['forecasted_demand'], 
                       self.data['demand']['actual_demand'], 
                       alpha=0.7)
            plt.plot([0, self.data['demand']['actual_demand'].max()], 
                    [0, self.data['demand']['actual_demand'].max()], 
                    'r--', alpha=0.8)
            plt.xlabel('Forecasted Demand')
            plt.ylabel('Actual Demand')
            plt.title('Demand Forecast Accuracy')
        
        # 4. Inventory turnover by category
        if 'inventory' in self.data:
            plt.subplot(3, 3, 4)
            self.data['inventory'].boxplot(column='stock_turnover_rate', by='category')
            plt.title('Inventory Turnover by Category')
            plt.suptitle('')
        
        # 5. Delivery delays by transport mode
        if 'logistics' in self.data:
            plt.subplot(3, 3, 5)
            self.data['logistics'].boxplot(column='delay_days', by='transport_mode')
            plt.title('Delivery Delays by Transport Mode')
            plt.suptitle('')
        
        # 6. ABC classification distribution
        if 'inventory' in self.data:
            plt.subplot(3, 3, 6)
            self.data['inventory']['abc_classification'].value_counts().plot(kind='bar')
            plt.title('ABC Classification Distribution')
            plt.xticks(rotation=45)
        
        # 7. Cost trends over time
        if 'orders' in self.data and 'order_date' in self.data['orders'].columns:
            plt.subplot(3, 3, 7)
            orders_with_date = self.data['orders'].copy()
            orders_with_date['order_date'] = pd.to_datetime(orders_with_date['order_date'])
            daily_costs = orders_with_date.groupby('order_date')['total_cost'].sum()
            daily_costs.plot()
            plt.title('Daily Order Costs')
            plt.xlabel('Date')
            plt.ylabel('Total Cost ($)')
        
        # 8. Supplier risk distribution
        if 'suppliers' in self.data:
            plt.subplot(3, 3, 8)
            self.data['suppliers']['risk_level'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Supplier Risk Distribution')
        
        # 9. Route efficiency vs distance
        if 'logistics' in self.data:
            plt.subplot(3, 3, 9)
            plt.scatter(self.data['logistics']['distance_km'], 
                       self.data['logistics']['route_efficiency'], 
                       alpha=0.7)
            plt.xlabel('Distance (km)')
            plt.ylabel('Route Efficiency')
            plt.title('Route Efficiency vs Distance')
        
        plt.tight_layout()
        plt.savefig('supply_chain_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualizations saved as 'supply_chain_analysis.png'")
        
        return fig
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*60)
        print("SUPPLY CHAIN INSIGHTS REPORT")
        print("="*60)
        
        insights = []
        
        # Cost insights
        if 'orders' in self.data:
            total_cost = self.data['orders']['total_cost'].sum()
            avg_cost = self.data['orders']['total_cost'].mean()
            insights.append(f"💰 Total order value: ${total_cost:,.2f}")
            insights.append(f"💰 Average order value: ${avg_cost:,.2f}")
        
        # Supplier insights
        if 'suppliers' in self.data:
            high_quality = len(self.data['suppliers'][self.data['suppliers']['quality_rating'] > 4.5])
            insights.append(f"⭐ High-quality suppliers (>4.5 rating): {high_quality}")
            
            high_risk = len(self.data['suppliers'][self.data['suppliers']['risk_level'] == 'High'])
            insights.append(f"⚠️ High-risk suppliers: {high_risk}")
        
        # Demand insights
        if 'demand' in self.data:
            forecast_accuracy = 1 - abs(self.data['demand']['actual_demand'] - 
                                       self.data['demand']['forecasted_demand']).mean() / \
                               self.data['demand']['actual_demand'].mean()
            insights.append(f"📈 Forecast accuracy: {forecast_accuracy:.1%}")
        
        # Inventory insights
        if 'inventory' in self.data:
            overstocked = len(self.data['inventory'][self.data['inventory']['current_stock'] > 
                                                   self.data['inventory']['max_stock_level']])
            understocked = len(self.data['inventory'][self.data['inventory']['current_stock'] < 
                                                    self.data['inventory']['min_stock_level']])
            insights.append(f"📦 Overstocked items: {overstocked}")
            insights.append(f"📉 Understocked items: {understocked}")
        
        # Logistics insights
        if 'logistics' in self.data:
            on_time_rate = len(self.data['logistics'][self.data['logistics']['delay_days'] <= 0]) / len(self.data['logistics'])
            insights.append(f"🚚 On-time delivery rate: {on_time_rate:.1%}")
            
            avg_satisfaction = self.data['logistics']['customer_satisfaction'].mean()
            insights.append(f"😊 Average customer satisfaction: {avg_satisfaction:.1f}/5")
        
        print("\n🎯 KEY INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
        
        return insights
    
    def run_complete_analysis(self):
        """Run complete supply chain analysis pipeline"""
        print("🚀 Starting Comprehensive Supply Chain Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.exploratory_data_analysis()
        self.demand_forecasting_analysis()
        self.supplier_optimization_analysis()
        self.inventory_optimization_analysis()
        self.logistics_efficiency_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate insights
        self.generate_insights_report()
        
        print("\n✅ Analysis complete! Check 'supply_chain_analysis.png' for visualizations.")
        return self.results

if __name__ == "__main__":
    # Initialize and run analysis
    analyzer = SupplyChainAnalyzer()
    results = analyzer.run_complete_analysis()
