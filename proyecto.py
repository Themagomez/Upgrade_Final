# %%
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# Set the start and end date for 3-year historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Introduction", "S&P 500", "Sector Analysis", "Stock Analysis", "Portfolio Comparison", "3-Year Prediction"]
selection = st.sidebar.radio("Go to", pages)

# Data loading function with retry mechanism
@st.cache_data
def load_data(ticker, start, end, max_retries=3, retry_delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            return data
        except Exception as e:
            st.warning(f"Failed to load data for {ticker}: {e}. Retrying in {retry_delay} seconds...")
            attempt += 1
            time.sleep(retry_delay)
    st.error(f"Failed to load data for {ticker} after {max_retries} attempts.")
    return None
    
# Introduction Page
if selection == "Introduction":
    st.title("Introduction to Investing")
    
    st.markdown(
        """
        **“If you don't find a way to make money while you sleep, you will work until you die.”**  
        *Warren Buffet*
        
        ### What is investing?
        Investing is allocating capital (money) into an asset, project, or activity that will, over time, provide a financial return through income or appreciation.
        
        ### Why should I invest?
        Although there can be several reasons for investment, the most common is to maintain the purchasing power of your savings and protect yourself from inflation.
        
        ### How can I invest?
        There are many ways to invest, from starting your own business to luxury watches, from rare comic books to higher education, from livestock to financial assets. You need to find the most suited for yourself.
        
        ### What investment is right for me?
        Many factors should be considered before answering this question. The two most important are: time horizon and risk tolerance. Another factor is whether you seek an active (work for it) or a passive (provide capital) investment.
        
        For the purpose of this introduction, we will focus on financial assets, which are considered passive income. Financial assets include the two most popular assets: **Stocks** and **Bonds**.
        
        **A stock** represents a share in the ownership of a company, including a claim on the company's earnings.
        
        **A bond** represents a loan to the issuer, and they agree to pay you back the initial amount of the loan on a specific date, and to pay you periodic interest payments along the way.
        
        We will focus on **Stocks**, specifically those included in the **Standard & Poor’s 500 Index**.
        
        ### What is the Standard & Poor’s 500 Index (S&P 500)?
        The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the stock performance of 500 of the largest companies listed on stock exchanges in the United States.
        
        ### What is an Index?
        An index measures the price performance of a basket of securities using a standardized metric and methodology.
        """
    )

# Home Page - S&P 500 Overview
if selection == "S&P 500":
    st.title("S&P 500 Index Overview")
    st.write("This page shows the overall performance of the S&P 500 index over the last 3 years.")
    
    # Load S&P 500 data
    sp500 = load_data('^GSPC', start_date, end_date)
    
    if not sp500.empty:
        # Calculating the required statistics
        opening_value = sp500['Open'].iloc[0]
        closing_value = sp500['Close'].iloc[-1]
        total_return = ((closing_value - opening_value) / opening_value) * 100
        minimum_value = sp500['Low'].min()
        maximum_value = sp500['High'].max()
        volatility = sp500['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        
        # Calculating CAGR
        num_years = (end_date - start_date).days / 365.25  # Accounts for leap years
        cagr = ((closing_value / opening_value) ** (1 / num_years) - 1) * 100
        
        # Creating a DataFrame to display the statistics
        summary_data = {
            "Statistic": ["Opening Value", "Closing Value", "Total Return (%)", "CAGR (%)", "Minimum Value", "Maximum Value", "Volatility"],
            "Value": [
                f"${opening_value:,.2f}", 
                f"${closing_value:,.2f}", 
                f"{total_return:.2f}%", 
                f"{cagr:.2f}%", 
                f"${minimum_value:,.2f}", 
                f"${maximum_value:,.2f}", 
                f"{volatility:.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Displaying the line chart of S&P 500 closing prices
        st.line_chart(sp500['Close'])
        
        # Displaying the summary statistics in a table
        st.table(summary_df)
        
        # footnote explaining volatility
        st.markdown(
            """
            **Footnote:**
            
            Volatility is calculated as the annualized standard deviation of daily returns over the selected period. 
            This measure gives an indication of the degree of variation in the S&P 500 index price, with higher volatility 
            indicating more significant price fluctuations. Annualizing the standard deviation helps compare volatility 
            across different time periods on a consistent basis.
            """
        )

# Sector Analysis Page
if selection == "Sector Analysis":
    st.title("Sector Analysis")
    st.write("This page compares different sectors within the S&P 500 over the last 3 years.")
    
    # Defining the sectors and their corresponding ETFs
    sectors = {
        'Communication Services': 'XLC',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Industrials': 'XLI',
        'Information Technology': 'XLK',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU'
    }
    
    sector_data = {}
    
    # Loading data for each sector
    for sector, ticker in sectors.items():
        data = load_data(ticker, start_date, end_date)
        sector_data[sector] = data

    # Plotting sector performance over time
    plt.figure(figsize=(10, 6))
    for sector, data in sector_data.items():
        plt.plot(data['Close'], label=sector)
    
    plt.title("Sector Performance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    st.pyplot(plt)

    # Creating a summary table for the sectors
    summary_data = {
        "Sector": [],
        "Opening Value": [],
        "Closing Value": [],
        "Total Return (%)": [],
        "CAGR (%)": []
    }
    
    for sector, data in sector_data.items():
        opening_value = data['Open'].iloc[0]
        closing_value = data['Close'].iloc[-1]
        total_return = ((closing_value - opening_value) / opening_value) * 100
        num_years = (end_date - start_date).days / 365.25
        cagr = ((closing_value / opening_value) ** (1 / num_years) - 1) * 100
        
        summary_data["Sector"].append(sector)
        summary_data["Opening Value"].append(f"${opening_value:,.2f}")
        summary_data["Closing Value"].append(f"${closing_value:,.2f}")
        summary_data["Total Return (%)"].append(f"{total_return:.2f}%")
        summary_data["CAGR (%)"].append(f"{cagr:.2f}%")
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Sector definitions and examples
    sector_definitions = {
        'Communication Services': ("Companies providing communication services (e.g., telecom, media).", ["AT&T", "Verizon", "Netflix"]),
        'Consumer Discretionary': ("Companies producing goods/services not essential (e.g., retail, media).", ["Amazon", "Tesla", "Nike"]),
        'Consumer Staples': ("Companies producing essential goods (e.g., food, beverages).", ["Procter & Gamble", "Coca-Cola", "Walmart"]),
        'Energy': ("Companies involved in the production and supply of energy.", ["ExxonMobil", "Chevron", "ConocoPhillips"]),
        'Financials': ("Companies providing financial services (e.g., banks, insurance).", ["JPMorgan Chase", "Bank of America", "Goldman Sachs"]),
        'Healthcare': ("Companies providing healthcare products/services.", ["Johnson & Johnson", "Pfizer", "UnitedHealth Group"]),
        'Industrials': ("Companies producing goods used in construction and manufacturing.", ["General Electric", "Boeing", "3M"]),
        'Information Technology': ("Companies producing technology-related products/services.", ["Apple", "Microsoft", "NVIDIA"]),
        'Materials': ("Companies providing raw materials (e.g., chemicals, metals).", ["DuPont", "Newmont", "Dow"]),
        'Real Estate': ("Companies involved in real estate operations.", ["American Tower", "Prologis", "Equinix"]),
        'Utilities': ("Companies providing essential utility services (e.g., electricity, water).", ["Duke Energy", "NextEra Energy", "Southern Company"])
    }

    sector_info = {
        "Sector": [],
        "Definition": [],
        "Example Companies": []
    }
    
    for sector, (definition, examples) in sector_definitions.items():
        sector_info["Sector"].append(sector)
        sector_info["Definition"].append(definition)
        sector_info["Example Companies"].append(", ".join(examples))
    
    sector_info_df = pd.DataFrame(sector_info)
    st.table(sector_info_df)
    
sp500_companies = {
    'AAPL': ['Apple Inc.', 'Information Technology'],
    'MSFT': ['Microsoft Corporation', 'Information Technology'],
    'GOOGL': ['Alphabet Inc. Class A', 'Communication Services'],
    'GOOG': ['Alphabet Inc. Class C', 'Communication Services'],
    'AMZN': ['Amazon.com Inc.', 'Consumer Discretionary'],
    'TSLA': ['Tesla Inc.', 'Consumer Discretionary'],
    'NVDA': ['NVIDIA Corporation', 'Information Technology'],
    'META': ['Meta Platforms Inc. Class A', 'Communication Services'],
    'UNH': ['UnitedHealth Group Incorporated', 'Healthcare'],
    'JNJ': ['Johnson & Johnson', 'Healthcare'],
    'V': ['Visa Inc. Class A', 'Financials'],
    'PG': ['Procter & Gamble Company', 'Consumer Staples'],
    'XOM': ['Exxon Mobil Corporation', 'Energy'],
    'JPM': ['JPMorgan Chase & Co.', 'Financials'],
    'MA': ['Mastercard Incorporated Class A', 'Financials'],
    'HD': ['Home Depot Inc.', 'Consumer Discretionary'],
    'LLY': ['Eli Lilly and Company', 'Healthcare'],
    'PFE': ['Pfizer Inc.', 'Healthcare'],
    'MRK': ['Merck & Co. Inc.', 'Healthcare'],
    'PEP': ['PepsiCo Inc.', 'Consumer Staples'],
    'KO': ['Coca-Cola Company', 'Consumer Staples'],
    'ABBV': ['AbbVie Inc.', 'Healthcare'],
    'AVGO': ['Broadcom Inc.', 'Information Technology'],
    'COST': ['Costco Wholesale Corporation', 'Consumer Staples'],
    'WMT': ['Walmart Inc.', 'Consumer Staples'],
    'DIS': ['Walt Disney Company', 'Communication Services'],
    'CSCO': ['Cisco Systems Inc.', 'Information Technology'],
    'ADBE': ['Adobe Inc.', 'Information Technology'],
    'NFLX': ['Netflix Inc.', 'Communication Services'],
    'VZ': ['Verizon Communications Inc.', 'Communication Services'],
    'INTC': ['Intel Corporation', 'Information Technology'],
    'TMO': ['Thermo Fisher Scientific Inc.', 'Healthcare'],
    'CMCSA': ['Comcast Corporation Class A', 'Communication Services'],
    'TMUS': ['T-Mobile US Inc.', 'Communication Services'],
    'NKE': ['Nike Inc. Class B', 'Consumer Discretionary'],
    'ORCL': ['Oracle Corporation', 'Information Technology'],
    'ABT': ['Abbott Laboratories', 'Healthcare'],
    'MCD': ['McDonald\'s Corporation', 'Consumer Discretionary'],
    'MDT': ['Medtronic plc', 'Healthcare'],
    'AMD': ['Advanced Micro Devices Inc.', 'Information Technology'],
    'CRM': ['Salesforce Inc.', 'Information Technology'],
    'HON': ['Honeywell International Inc.', 'Industrials'],
    'DHR': ['Danaher Corporation', 'Healthcare'],
    'BMY': ['Bristol-Myers Squibb Company', 'Healthcare'],
    'SBUX': ['Starbucks Corporation', 'Consumer Discretionary'],
    'AMGN': ['Amgen Inc.', 'Healthcare'],
    'BA': ['Boeing Company', 'Industrials'],
    'LIN': ['Linde plc', 'Materials'],
    'LOW': ['Lowe\'s Companies Inc.', 'Consumer Discretionary'],
    'SPGI': ['S&P Global Inc.', 'Financials'],
    'CAT': ['Caterpillar Inc.', 'Industrials'],
    'NEE': ['NextEra Energy Inc.', 'Utilities'],
    'RTX': ['Raytheon Technologies Corporation', 'Industrials'],
    'TXN': ['Texas Instruments Incorporated', 'Information Technology'],
    'GS': ['Goldman Sachs Group Inc.', 'Financials'],
    'BLK': ['BlackRock Inc.', 'Financials'],
    'AXP': ['American Express Company', 'Financials'],
    'UNP': ['Union Pacific Corporation', 'Industrials'],
    'QCOM': ['QUALCOMM Incorporated', 'Information Technology'],
    'T': ['AT&T Inc.', 'Communication Services'],
    'AMT': ['American Tower Corporation', 'Real Estate'],
    'SYK': ['Stryker Corporation', 'Healthcare'],
    'MS': ['Morgan Stanley', 'Financials'],
    'MMM': ['3M Company', 'Industrials'],
    'CVX': ['Chevron Corporation', 'Energy'],
    'EL': ['Estee Lauder Companies Inc. Class A', 'Consumer Staples'],
    'PLD': ['Prologis Inc.', 'Real Estate'],
    'C': ['Citigroup Inc.', 'Financials'],
    'ISRG': ['Intuitive Surgical Inc.', 'Healthcare'],
    'BKNG': ['Booking Holdings Inc.', 'Consumer Discretionary'],
    'USB': ['U.S. Bancorp', 'Financials'],
    'COP': ['ConocoPhillips', 'Energy'],
    'NOW': ['ServiceNow Inc.', 'Information Technology'],
    'DE': ['Deere & Company', 'Industrials'],
    'ADP': ['Automatic Data Processing Inc.', 'Information Technology'],
    'VRTX': ['Vertex Pharmaceuticals Incorporated', 'Healthcare'],
    'LMT': ['Lockheed Martin Corporation', 'Industrials'],
    'GILD': ['Gilead Sciences Inc.', 'Healthcare'],
    'ZTS': ['Zoetis Inc. Class A', 'Healthcare'],
    'SCHW': ['Charles Schwab Corporation', 'Financials'],
    'REGN': ['Regeneron Pharmaceuticals Inc.', 'Healthcare'],
    'NOC': ['Northrop Grumman Corporation', 'Industrials'],
    'FISV': ['Fiserv Inc.', 'Information Technology'],
    'MO': ['Altria Group Inc.', 'Consumer Staples'],
    'CCI': ['Crown Castle International Corp', 'Real Estate'],
    'MCO': ['Moody\'s Corporation', 'Financials'],
    'GE': ['General Electric Company', 'Industrials'],
    'LRCX': ['Lam Research Corporation', 'Information Technology'],
    'CL': ['Colgate-Palmolive Company', 'Consumer Staples'],
    'SO': ['Southern Company', 'Utilities'],
    'KMB': ['Kimberly-Clark Corporation', 'Consumer Staples'],
    'TJX': ['TJX Companies Inc.', 'Consumer Discretionary'],
    'MDLZ': ['Mondelez International Inc. Class A', 'Consumer Staples'],
    'HUM': ['Humana Inc.', 'Healthcare'],
    'EOG': ['EOG Resources Inc.', 'Energy'],
    'D': ['Dominion Energy Inc.', 'Utilities'],
    'PGR': ['Progressive Corporation', 'Financials'],
    'DUK': ['Duke Energy Corporation', 'Utilities'],
    'AON': ['Aon plc Class A', 'Financials'],
    'BDX': ['Becton Dickinson and Company', 'Healthcare'],
    'CSX': ['CSX Corporation', 'Industrials'],
    'PSA': ['Public Storage', 'Real Estate'],
    'MMC': ['Marsh & McLennan Companies Inc.', 'Financials'],
    'CME': ['CME Group Inc. Class A', 'Financials'],
    'ADSK': ['Autodesk Inc.', 'Information Technology'],
    'GM': ['General Motors Company', 'Consumer Discretionary'],
    'ROP': ['Roper Technologies Inc.', 'Industrials'],
    'NSC': ['Norfolk Southern Corporation', 'Industrials'],
    'LHX': ['L3Harris Technologies Inc.', 'Industrials'],
    'WM': ['Waste Management Inc.', 'Industrials'],
    'ITW': ['Illinois Tool Works Inc.', 'Industrials'],
    'MAR': ['Marriott International Inc. Class A', 'Consumer Discretionary'],
    'CNC': ['Centene Corporation', 'Healthcare'],
    'MET': ['MetLife Inc.', 'Financials'],
    'COF': ['Capital One Financial Corporation', 'Financials'],
    'CTSH': ['Cognizant Technology Solutions Corporation Class A', 'Information Technology'],
    'FDX': ['FedEx Corporation', 'Industrials'],
    'KLAC': ['KLA Corporation', 'Information Technology'],
    'BAX': ['Baxter International Inc.', 'Healthcare'],
    'STZ': ['Constellation Brands Inc. Class A', 'Consumer Staples'],
    'PAYX': ['Paychex Inc.', 'Information Technology'],
    'WBA': ['Walgreens Boots Alliance Inc.', 'Consumer Staples'],
    'AIG': ['American International Group Inc.', 'Financials'],
    'DXCM': ['DexCom Inc.', 'Healthcare'],
    'DOW': ['Dow Inc.', 'Materials'],
    'XEL': ['Xcel Energy Inc.', 'Utilities'],
    'HPQ': ['HP Inc.', 'Information Technology'],
    'DD': ['DuPont de Nemours Inc.', 'Materials'],
    'ROST': ['Ross Stores Inc.', 'Consumer Discretionary'],
    'YUM': ['Yum! Brands Inc.', 'Consumer Discretionary'],
    'ROK': ['Rockwell Automation Inc.', 'Industrials'],
    'APH': ['Amphenol Corporation Class A', 'Information Technology'],
    'IDXX': ['IDEXX Laboratories Inc.', 'Healthcare'],
    'ETN': ['Eaton Corp. plc', 'Industrials'],
    'ORLY': ['O\'Reilly Automotive Inc.', 'Consumer Discretionary'],
    'ECL': ['Ecolab Inc.', 'Materials'],
    'MTD': ['Mettler-Toledo International Inc.', 'Healthcare'],
    'SYY': ['Sysco Corporation', 'Consumer Staples'],
    'AME': ['AMETEK Inc.', 'Industrials'],
    'MCHP': ['Microchip Technology Incorporated', 'Information Technology'],
    'GPN': ['Global Payments Inc.', 'Information Technology'],
    'AVB': ['AvalonBay Communities Inc.', 'Real Estate'],
    'CDNS': ['Cadence Design Systems Inc.', 'Information Technology'],
    'VLO': ['Valero Energy Corporation', 'Energy'],
    'A': ['Agilent Technologies Inc.', 'Healthcare'],
    'DLR': ['Digital Realty Trust Inc.', 'Real Estate'],
    'IQV': ['IQVIA Holdings Inc.', 'Healthcare'],
    'AWK': ['American Water Works Company Inc.', 'Utilities'],
    'F': ['Ford Motor Company', 'Consumer Discretionary'],
    'SBAC': ['SBA Communications Corp. Class A', 'Real Estate'],
    'TRV': ['Travelers Companies Inc.', 'Financials'],
    'WMB': ['Williams Companies Inc.', 'Energy'],
    'APD': ['Air Products and Chemicals Inc.', 'Materials'],
    'SWK': ['Stanley Black & Decker Inc.', 'Industrials'],
    'RMD': ['ResMed Inc.', 'Healthcare'],
    'HLT': ['Hilton Worldwide Holdings Inc.', 'Consumer Discretionary'],
    'VTR': ['Ventas Inc.', 'Real Estate'],
    'NDAQ': ['Nasdaq Inc.', 'Financials'],
    'TT': ['Trane Technologies plc', 'Industrials'],
    'ALB': ['Albemarle Corporation', 'Materials'],
    'EXR': ['Extra Space Storage Inc.', 'Real Estate'],
    'EFX': ['Equifax Inc.', 'Industrials'],
    'FAST': ['Fastenal Company', 'Industrials'],
    'RSG': ['Republic Services Inc.', 'Industrials'],
    'PPL': ['PPL Corporation', 'Utilities'],
    'DTE': ['DTE Energy Company', 'Utilities'],
    'AEE': ['Ameren Corporation', 'Utilities'],
    'TDG': ['TransDigm Group Incorporated', 'Industrials'],
    'DRI': ['Darden Restaurants Inc.', 'Consumer Discretionary'],
    'CBRE': ['CBRE Group Inc. Class A', 'Real Estate'],
    'AES': ['AES Corporation', 'Utilities'],
    'ZBH': ['Zimmer Biomet Holdings Inc.', 'Healthcare'],
    'PPG': ['PPG Industries Inc.', 'Materials'],
    'MPC': ['Marathon Petroleum Corporation', 'Energy'],
    'STE': ['STERIS Plc', 'Healthcare'],
    'CFG': ['Citizens Financial Group Inc.', 'Financials'],
    'OMC': ['Omnicom Group Inc.', 'Communication Services'],
    'ADM': ['Archer-Daniels-Midland Company', 'Consumer Staples'],
    'BKR': ['Baker Hughes Company Class A', 'Energy'],
    'CAG': ['Conagra Brands Inc.', 'Consumer Staples'],
    'O': ['Realty Income Corporation', 'Real Estate'],
    'OKE': ['ONEOK Inc.', 'Energy'],
    'ETR': ['Entergy Corporation', 'Utilities'],
    'TDY': ['Teledyne Technologies Incorporated', 'Information Technology'],
    'DHI': ['D.R. Horton Inc.', 'Consumer Discretionary'],
    'XYL': ['Xylem Inc.', 'Industrials'],
    'WEC': ['WEC Energy Group Inc.', 'Utilities'],
    'GLW': ['Corning Inc.', 'Information Technology'],
    'FITB': ['Fifth Third Bancorp', 'Financials'],
    'VFC': ['VF Corporation', 'Consumer Discretionary'],
    'PCAR': ['PACCAR Inc.', 'Industrials'],
    'HIG': ['Hartford Financial Services Group Inc.', 'Financials'],
    'GWW': ['W.W. Grainger Inc.', 'Industrials'],
    'AKAM': ['Akamai Technologies Inc.', 'Information Technology'],
    'WYNN': ['Wynn Resorts Ltd.', 'Consumer Discretionary'],
    'IP': ['International Paper Company', 'Materials'],
    'GRMN': ['Garmin Ltd.', 'Consumer Discretionary'],
    'LUV': ['Southwest Airlines Co.', 'Industrials'],
    'NRG': ['NRG Energy Inc.', 'Utilities'],
    'HSY': ['Hershey Company', 'Consumer Staples'],
    'URI': ['United Rentals Inc.', 'Industrials'],
    'DOV': ['Dover Corporation', 'Industrials'],
    'KEYS': ['Keysight Technologies Inc.', 'Information Technology'],
    'ETSY': ['Etsy Inc.', 'Consumer Discretionary'],
    'SIVB': ['SVB Financial Group', 'Financials'],
    'CE': ['Celanese Corporation', 'Materials'],
    'BWA': ['BorgWarner Inc.', 'Consumer Discretionary'],
    'MOS': ['Mosaic Company', 'Materials'],
    'HBI': ['Hanesbrands Inc.', 'Consumer Discretionary'],
    'HOG': ['Harley-Davidson Inc.', 'Consumer Discretionary'],
    'CLX': ['Clorox Company', 'Consumer Staples'],
    'ULTA': ['Ulta Beauty Inc.', 'Consumer Discretionary'],
    'HOLX': ['Hologic Inc.', 'Healthcare'],
    'SWKS': ['Skyworks Solutions Inc.', 'Information Technology'],
    'HRL': ['Hormel Foods Corporation', 'Consumer Staples'],
    'SEE': ['Sealed Air Corporation', 'Materials'],
    'HES': ['Hess Corporation', 'Energy'],
    'RJF': ['Raymond James Financial Inc.', 'Financials'],
    'IRM': ['Iron Mountain Incorporated', 'Real Estate'],
    'CPRT': ['Copart Inc.', 'Industrials'],
    'TSCO': ['Tractor Supply Company', 'Consumer Discretionary'],
    'WAT': ['Waters Corporation', 'Healthcare'],
    'TER': ['Teradyne Inc.', 'Information Technology'],
    'EXPD': ['Expeditors International of Washington Inc.', 'Industrials'],
    'PKG': ['Packaging Corporation of America', 'Materials'],
    'NVR': ['NVR Inc.', 'Consumer Discretionary'],
    'LYV': ['Live Nation Entertainment Inc.', 'Communication Services'],
    'VMC': ['Vulcan Materials Company', 'Materials'],
    'REXR': ['Rexford Industrial Realty Inc.', 'Real Estate'],
    'TDOC': ['Teladoc Health Inc.', 'Healthcare'],
    'LVS': ['Las Vegas Sands Corp.', 'Consumer Discretionary'],
    'MRO': ['Marathon Oil Corporation', 'Energy'],
    'VNO': ['Vornado Realty Trust', 'Real Estate'],
    'ALLE': ['Allegion Plc', 'Industrials'],
    'BRO': ['Brown & Brown Inc.', 'Financials'],
    'EVRG': ['Evergy Inc.', 'Utilities'],
    'SLB': ['Schlumberger NV', 'Energy'],
    'IPGP': ['IPG Photonics Corporation', 'Information Technology'],
    'ZION': ['Zions Bancorp N.A.', 'Financials'],
    'BEN': ['Franklin Resources Inc.', 'Financials'],
    'WHR': ['Whirlpool Corporation', 'Consumer Discretionary'],
    'MHK': ['Mohawk Industries Inc.', 'Consumer Discretionary'],
    'PVH': ['PVH Corp.', 'Consumer Discretionary'],
    'MTB': ['M&T Bank Corporation', 'Financials'],
    'RCL': ['Royal Caribbean Cruises Ltd.', 'Consumer Discretionary'],
    'TXT': ['Textron Inc.', 'Industrials'],
    'AVY': ['Avery Dennison Corporation', 'Materials'],
    'AAP': ['Advance Auto Parts Inc.', 'Consumer Discretionary'],
    'CHRW': ['C.H. Robinson Worldwide Inc.', 'Industrials'],
    'IVZ': ['Invesco Ltd.', 'Financials'],
    'BXP': ['Boston Properties Inc.', 'Real Estate'],
    'XRAY': ['DENTSPLY SIRONA Inc.', 'Healthcare'],
    'HSIC': ['Henry Schein Inc.', 'Healthcare'],
    'NWSA': ['News Corporation Class A', 'Communication Services'],
    'CTLT': ['Catalent Inc.', 'Healthcare'],
    'UHS': ['Universal Health Services Inc. Class B', 'Healthcare'],
    'HII': ['Huntington Ingalls Industries Inc.', 'Industrials'],
    'MLM': ['Martin Marietta Materials Inc.', 'Materials'],
    'AKR': ['Acadia Realty Trust', 'Real Estate'],
    'RMD': ['ResMed Inc.', 'Healthcare'],
    'CBOE': ['Cboe Global Markets Inc.', 'Financials'],
    'SWK': ['Stanley Black & Decker Inc.', 'Industrials'],
    'SJM': ['J.M. Smucker Company', 'Consumer Staples'],
    'FE': ['FirstEnergy Corp.', 'Utilities'],
    'LEG': ['Leggett & Platt Incorporated', 'Consumer Discretionary'],
    'NWS': ['News Corporation Class B', 'Communication Services'],
    'UDR': ['UDR Inc.', 'Real Estate'],
    'QRVO': ['Qorvo Inc.', 'Information Technology'],
    'FMC': ['FMC Corporation', 'Materials'],
    'WY': ['Weyerhaeuser Company', 'Real Estate'],
    'AFL': ['Aflac Incorporated', 'Financials'],
    'EFX': ['Equifax Inc.', 'Industrials'],
    'REG': ['Regency Centers Corporation', 'Real Estate'],
    'LKQ': ['LKQ Corporation', 'Consumer Discretionary'],
    'LKQ': ['LKQ Corporation', 'Consumer Discretionary'],
    'ESS': ['Essex Property Trust Inc.', 'Real Estate'],
    'SEE': ['Sealed Air Corporation', 'Materials'],
    'LNT': ['Alliant Energy Corporation', 'Utilities'],
    'TFX': ['Teleflex Incorporated', 'Healthcare'],
    'YUMC': ['Yum China Holdings Inc.', 'Consumer Discretionary'],
    'TAP': ['Molson Coors Beverage Company Class B', 'Consumer Staples'],
    'ATO': ['Atmos Energy Corporation', 'Utilities'],
    'CNP': ['CenterPoint Energy Inc.', 'Utilities'],
    'HST': ['Host Hotels & Resorts Inc.', 'Real Estate'],
    'PNW': ['Pinnacle West Capital Corporation', 'Utilities'],
    'MAS': ['Masco Corporation', 'Industrials'],
    'GPC': ['Genuine Parts Company', 'Consumer Discretionary'],
    'INCY': ['Incyte Corporation', 'Healthcare'],
    'AAP': ['Advance Auto Parts Inc.', 'Consumer Discretionary'],
    'EVRG': ['Evergy Inc.', 'Utilities'],
    'BIO': ['Bio-Rad Laboratories Inc. Class A', 'Healthcare'],
    'RCL': ['Royal Caribbean Cruises Ltd.', 'Consumer Discretionary'],
    'GWW': ['W.W. Grainger Inc.', 'Industrials'],
    'HPE': ['Hewlett Packard Enterprise Co.', 'Information Technology'],
    'ETR': ['Entergy Corporation', 'Utilities'],
    'OMC': ['Omnicom Group Inc.', 'Communication Services'],
    'CHD': ['Church & Dwight Co. Inc.', 'Consumer Staples'],
    'MCHP': ['Microchip Technology Incorporated', 'Information Technology'],
    'BXP': ['Boston Properties Inc.', 'Real Estate'],
    'MHK': ['Mohawk Industries Inc.', 'Consumer Discretionary'],
    'DG': ['Dollar General Corporation', 'Consumer Discretionary'],
    'WRK': ['WestRock Company', 'Materials'],
    'PWR': ['Quanta Services Inc.', 'Industrials'],
    'HRB': ['H&R Block Inc.', 'Consumer Discretionary'],
    'ATO': ['Atmos Energy Corporation', 'Utilities'],
    'AMCR': ['Amcor plc', 'Materials'],
    'AIZ': ['Assurant Inc.', 'Financials'],
    'UAL': ['United Airlines Holdings Inc.', 'Industrials'],
    'VTRS': ['Viatris Inc.', 'Healthcare'],
    'ANET': ['Arista Networks Inc.', 'Information Technology'],
    'VTR': ['Ventas Inc.', 'Real Estate'],
    'NUE': ['Nucor Corporation', 'Materials'],
    'O': ['Realty Income Corporation', 'Real Estate'],
    'AJG': ['Arthur J. Gallagher & Co.', 'Financials'],
    'MPC': ['Marathon Petroleum Corporation', 'Energy'],
    'CSX': ['CSX Corporation', 'Industrials'],
    'FDS': ['FactSet Research Systems Inc.', 'Financials'],
    'BKR': ['Baker Hughes Company Class A', 'Energy'],
    'ALB': ['Albemarle Corporation', 'Materials'],
    'STLD': ['Steel Dynamics Inc.', 'Materials'],
    'JKHY': ['Jack Henry & Associates Inc.', 'Information Technology'],
    'ROL': ['Rollins Inc.', 'Industrials'],
    'CTLT': ['Catalent Inc.', 'Healthcare'],
    'HII': ['Huntington Ingalls Industries Inc.', 'Industrials'],
    'EXPD': ['Expeditors International of Washington Inc.', 'Industrials'],
    'IPG': ['Interpublic Group of Companies Inc.', 'Communication Services'],
    'NEM': ['Newmont Corporation', 'Materials'],
    'RMD': ['ResMed Inc.', 'Healthcare'],
    'FIS': ['Fidelity National Information Services Inc.', 'Information Technology'],
    'MPWR': ['Monolithic Power Systems Inc.', 'Information Technology'],
    'ALLE': ['Allegion Plc', 'Industrials'],
    'AES': ['AES Corporation', 'Utilities'],
    'DRI': ['Darden Restaurants Inc.', 'Consumer Discretionary'],
    'CBOE': ['Cboe Global Markets Inc.', 'Financials'],
    'AMT': ['American Tower Corporation', 'Real Estate'],
    'CMS': ['CMS Energy Corporation', 'Utilities'],
    'MOS': ['Mosaic Company', 'Materials'],
    'CHRW': ['C.H. Robinson Worldwide Inc.', 'Industrials'],
    'CPT': ['Camden Property Trust', 'Real Estate'],
    'HSY': ['Hershey Company', 'Consumer Staples'],
    'TYL': ['Tyler Technologies Inc.', 'Information Technology'],
    'MAS': ['Masco Corporation', 'Industrials'],
    'HRL': ['Hormel Foods Corporation', 'Consumer Staples'],
    'ANSS': ['ANSYS Inc.', 'Information Technology'],
    'CFG': ['Citizens Financial Group Inc.', 'Financials'],
    'TROW': ['T. Rowe Price Group', 'Financials'],
    'ATO': ['Atmos Energy Corporation', 'Utilities'],
    'RMD': ['ResMed Inc.', 'Healthcare'],
    'BIO': ['Bio-Rad Laboratories Inc. Class A', 'Healthcare'],
    'HPE': ['Hewlett Packard Enterprise Co.', 'Information Technology'],
    'AJG': ['Arthur J. Gallagher & Co.', 'Financials'],
    'CARR': ['Carrier Global Corporation', 'Industrials'],
    'WAB': ['Westinghouse Air Brake Technologies Corporation', 'Industrials'],
    'XYL': ['Xylem Inc.', 'Industrials'],
    'RMD': ['ResMed Inc.', 'Healthcare'],
    'OKE': ['ONEOK Inc.', 'Energy'],
    'BRO': ['Brown & Brown Inc.', 'Financials'],
    'IQV': ['IQVIA Holdings Inc.', 'Healthcare'],
    'AME': ['AMETEK Inc.', 'Industrials'],
    'LVS': ['Las Vegas Sands Corp.', 'Consumer Discretionary'],
    'AAP': ['Advance Auto Parts Inc.', 'Consumer Discretionary'],
    'WRK': ['WestRock Company', 'Materials'],
    'CTSH': ['Cognizant Technology Solutions Corporation Class A', 'Information Technology'],
    'VNO': ['Vornado Realty Trust', 'Real Estate'],
    'PNW': ['Pinnacle West Capital Corporation', 'Utilities'],
    'HWM': ['Howmet Aerospace Inc.', 'Industrials'],
    'FMC': ['FMC Corporation', 'Materials'],
    'VFC': ['VF Corporation', 'Consumer Discretionary'],
    'PWR': ['Quanta Services Inc.', 'Industrials'],
    'TTWO': ['Take-Two Interactive Software Inc.', 'Communication Services'],
    'VMC': ['Vulcan Materials Company', 'Materials'],
    'KMX': ['CarMax Inc.', 'Consumer Discretionary'],
    'DLR': ['Digital Realty Trust Inc.', 'Real Estate'],
    'WYNN': ['Wynn Resorts Ltd.', 'Consumer Discretionary'],
    'KIM': ['Kimco Realty Corporation', 'Real Estate'],
    'VNO': ['Vornado Realty Trust', 'Real Estate'],
    'AAP': ['Advance Auto Parts Inc.', 'Consumer Discretionary'],
    'NVR': ['NVR Inc.', 'Consumer Discretionary'],
    'STX': ['Seagate Technology Holdings PLC', 'Information Technology'],
    'AVB': ['AvalonBay Communities Inc.', 'Real Estate'],
    'OMC': ['Omnicom Group Inc.', 'Communication Services'],
    'TROW': ['T. Rowe Price Group', 'Financials'],
    }

# Stock Analysis Page
if selection == "Stock Analysis":
    st.title("Stock Analysis")
    st.write("This page allows you to analyze individual stocks within the S&P 500 over the last 3 years.")
    
    # Dropdown menu to select the stock ticker
    tickers = list(sp500_companies.keys())
    ticker_choice = st.selectbox("Select a Stock Ticker:", tickers)
    
    company_name = sp500_companies[ticker_choice][0]
    sector_name = sp500_companies[ticker_choice][1]
    
    # Loading data for the selected stock
    stock_data = load_data(ticker_choice, start_date, end_date)
    
    if not stock_data.empty:
        # Displaying the interactive performance graph
        st.line_chart(stock_data['Close'])
        
        # Fetching additional stock information
        stock_info = yf.Ticker(ticker_choice).info
        shares_outstanding = stock_info.get('sharesOutstanding', 'N/A')
        market_cap = stock_info.get('marketCap', 'N/A')
        
        # Estimating total market cap of S&P 500 (placeholder value)
        total_market_cap_sp500 = 40e12  # Assuming the total market cap of S&P 500 is $40 trillion
        
        # Calculating weight in S&P 500
        weight_in_sp500 = (market_cap / total_market_cap_sp500) * 100 if market_cap != 'N/A' else 'N/A'
        
        # Displaying stock information
        st.subheader("Stock Information")
        st.write(f"**Ticker**: {ticker_choice}")
        st.write(f"**Company Name**: {company_name}")
        st.write(f"**Sector**: {sector_name}")
        st.write(f"**Shares Outstanding**: {shares_outstanding:,} shares")
        st.write(f"**Market Capitalization**: ${market_cap:,.2f}")
        st.write(f"**Weight in S&P 500 (2024)**: {weight_in_sp500:.4f}%" if weight_in_sp500 != 'N/A' else "**Weight in S&P 500 (2024)**: N/A")
        
        # Calculating the required statistics for the table
        opening_value = stock_data['Open'].iloc[0]
        closing_value = stock_data['Close'].iloc[-1]
        total_return = ((closing_value - opening_value) / opening_value) * 100
        minimum_value = stock_data['Low'].min()
        maximum_value = stock_data['High'].max()
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        
        # Calculating CAGR
        num_years = (end_date - start_date).days / 365.25
        cagr = ((closing_value / opening_value) ** (1 / num_years) - 1) * 100
        
        # Creating a DataFrame to display the statistics
        summary_data = {
            "Statistic": ["Opening Value", "Closing Value", "Total Return (%)", "CAGR (%)", "Minimum Value", "Maximum Value", "Volatility"],
            "Value": [
                f"${opening_value:,.2f}", 
                f"${closing_value:,.2f}", 
                f"{total_return:.2f}%", 
                f"{cagr:.2f}%", 
                f"${minimum_value:,.2f}", 
                f"${maximum_value:,.2f}", 
                f"{volatility:.2f}",
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Displaying the summary statistics in a table
        st.subheader("Stock Performance Summary")
        st.table(summary_df)
# Data loading function with error handling
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        return None

# Portfolio Comparison Page
if selection == "Portfolio Comparison":
    st.title("Portfolio Comparison")
    st.write("This page compares the volatility of different portfolios.")

    # Step 1: Calculate the volatility of each stock with error handling
    volatilities = {}
    for ticker in sp500_companies:
        data = load_data(ticker, start_date, end_date)
        if data is not None:
            volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            volatilities[ticker] = volatility
    
    if not volatilities:
        st.error("No valid data available for any tickers.")
    else:
        # Step 2: Sort stocks by volatility
        sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        
        # Step 3: Define portfolios
        aggressive_portfolio = [ticker for ticker, vol in sorted_volatilities[:5]]  # Top 5 most volatile
        
        low_risk_portfolio = [ticker for ticker, vol in sorted_volatilities[-100:]]  # Bottom 100 least volatile
        
        diversified_portfolio = []
        sectors = list(set([sp500_companies[ticker][1] for ticker in sp500_companies]))  # Unique sectors
        for sector in sectors:
            sector_stocks = [(ticker, vol) for ticker, vol in sorted_volatilities if sp500_companies[ticker][1] == sector]
            diversified_portfolio += [ticker for ticker, vol in sector_stocks[:2]]  # Top 2 in volatility
            diversified_portfolio += [ticker for ticker, vol in sector_stocks[-2:]]  # Bottom 2 in volatility
        
        # Step 4: Loading data for the portfolios, calculating returns, CAGR, and volatilities
        portfolio_data = {
            "Aggressive": aggressive_portfolio,
            "Low-Risk": low_risk_portfolio,
            "Diversified": diversified_portfolio
        }
        
        for portfolio_name, tickers in portfolio_data.items():
            portfolio_returns = []
            valid_tickers = []
            for ticker in tickers:
                data = load_data(ticker, start_date, end_date)
                if data is not None:
                    returns = data['Close'].pct_change()
                    portfolio_returns.append(returns)
                    valid_tickers.append(ticker)
            
            if portfolio_returns:
                combined_returns = pd.concat(portfolio_returns, axis=1).mean(axis=1)  # Average returns for the portfolio
                cumulative_returns = (1 + combined_returns).cumprod() - 1
                
                # Calculating portfolio statistics
                total_return = cumulative_returns[-1]
                cagr = (1 + total_return) ** (1 / 3) - 1  # Assuming 3-year period
                portfolio_volatility = combined_returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Plotting portfolio performance
                plt.figure(figsize=(10, 6))
                plt.plot(cumulative_returns.index, cumulative_returns, label=f"{portfolio_name} Portfolio")
                plt.title(f"{portfolio_name} Portfolio Performance")
                plt.xlabel("Date")
                plt.ylabel("Cumulative Returns")
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)
                
                # Displaying portfolio statistics in a table
                summary_data = {
                    "Tickers": [', '.join(valid_tickers)],
                    "Total Return (%)": [f"{total_return * 100:.2f}%"],
                    "CAGR (%)": [f"{cagr * 100:.2f}%"],
                    "Volatility": [f"{portfolio_volatility:.4f}"]
                }
                
                summary_df = pd.DataFrame(summary_data)
                
                st.subheader(f"{portfolio_name} Portfolio Summary")
                st.table(summary_df)
                
# Portfolio Comparison Portfolios (reused here)
def get_portfolios():
    volatilities = {}
    for ticker in sp500_companies:
        data = load_data(ticker, start_date, end_date)
        if data is not None:
            volatility = data['Close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
            volatilities[ticker] = volatility
    
    sorted_volatilities = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
    
    aggressive_portfolio = [ticker for ticker, vol in sorted_volatilities[:5]]  # Top 5 most volatile
    low_risk_portfolio = [ticker for ticker, vol in sorted_volatilities[-100:]]  # Bottom 100 least volatile
    
    diversified_portfolio = []
    sectors = list(set([sp500_companies[ticker][1] for ticker in sp500_companies]))  # Unique sectors
    for sector in sectors:
        sector_stocks = [(ticker, vol) for ticker, vol in sorted_volatilities if sp500_companies[ticker][1] == sector]
        diversified_portfolio += [ticker for ticker, vol in sector_stocks[:2]]  # Top 2 in volatility
        diversified_portfolio += [ticker for ticker, vol in sector_stocks[-2:]]  # Bottom 2 in volatility
    
    return {
        "Aggressive": aggressive_portfolio,
        "Low-Risk": low_risk_portfolio,
        "Diversified": diversified_portfolio
    }

# 3-Year Prediction Page
if selection == "3-Year Prediction":
    st.title("3-Year Prediction")
    st.write("This page predicts the expected return for each portfolio in 3 years from today.")
    
    # Loading S&P 500 data
    sp500 = load_data('^GSPC', start_date, end_date)
    
    if sp500 is None or sp500.empty:
        st.error("Failed to load S&P 500 data. Please check your connection or try again later.")
    else:
        # Get portfolios from the Portfolio Comparison logic
        portfolios = get_portfolios()
        
        future_date = end_date + timedelta(days=3*365)
        predictions = {}
        
        for portfolio_name, tickers in portfolios.items():
            combined_values = []
            for ticker in tickers:
                data = load_data(ticker, start_date, end_date)
                if data is not None:
                    combined_values.append(data['Close'])
            
            if combined_values:
                portfolio_df = pd.concat(combined_values, axis=1).mean(axis=1)  # Average closing price for the portfolio
                
                # Preparing the linear regression model for prediction
                days = np.array(range(len(portfolio_df))).reshape(-1, 1)
                model = LinearRegression()
                model.fit(days, portfolio_df.values)
                
                # Predicting the portfolio value 3 years into the future
                future_days = np.array(range(len(portfolio_df), len(portfolio_df) + 3*365)).reshape(-1, 1)
                prediction = model.predict(future_days)
                
                # Calculating expected return in %
                initial_value = portfolio_df.values[-1]
                final_value = prediction[-1]
                expected_return = (final_value - initial_value) / initial_value * 100
                predictions[portfolio_name] = expected_return
                
                # Calculating R-squared
                r_squared = model.score(days, portfolio_df.values)
                
                st.write(f"Expected Return for {portfolio_name} Portfolio in 3 years: {predictions[portfolio_name]:.2f}%")
                st.write(f"R-squared for {portfolio_name} Portfolio: {r_squared:.4f}")
                
                # Plotting the linear estimation graph
                plt.figure(figsize=(10, 6))
                plt.plot(days, portfolio_df.values, label=f"{portfolio_name} Historical Prices")
                plt.plot(future_days, prediction, label=f"{portfolio_name} Predicted Prices", linestyle="--")
                plt.title(f"{portfolio_name} Portfolio - 3-Year Linear Estimation")
                plt.xlabel("Days")
                plt.ylabel("Portfolio Value")
                plt.legend()
                plt.tight_layout()
                st.pyplot(plt)
        
        
        # footnote explaining the model used
        st.markdown(
            """
            **Footnote:**
            
            The predicted returns are estimated using a simple Linear Regression model. This model assumes that the historical 
            trend in portfolio prices will continue in the future. Linear Regression is a statistical method that models 
            the relationship between a dependent variable (in this case, portfolio value) and one or more independent variables 
            (in this case, time). The model fits a straight line to the historical data points and then projects this line 
            into the future to predict future values. It's important to note that this model does not account for market volatility, 
            macroeconomic factors, or other external influences that could impact future returns.
            
            The R-squared value indicates how well the linear model fits the historical data. A higher R-squared value 
            (closer to 1) indicates a better fit, meaning the model explains a larger portion of the variance in the portfolio value.
            """
        )


