#!/usr/bin/env python3
"""
Quick script to run supply chain analysis
"""

import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def run_analysis():
    """Run the supply chain analysis"""
    print("🚀 Running Supply Chain Analysis...")
    try:
        # Import and run the analysis
        from supply_chain_analysis import SupplyChainAnalyzer
        
        analyzer = SupplyChainAnalyzer()
        results = analyzer.run_complete_analysis()
        
        print("\n✅ Analysis completed successfully!")
        print("📊 Check 'supply_chain_analysis.png' for visualizations")
        return True
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Supply Chain Analysis Setup")
    print("=" * 40)
    
    # Install requirements
    if install_requirements():
        # Run analysis
        run_analysis()
    else:
        print("❌ Failed to install requirements. Please check your Python environment.")
