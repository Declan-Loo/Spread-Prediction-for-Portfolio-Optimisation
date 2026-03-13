TICKERS = [
    # Technology (Semiconductors)
    "NVDA.O",   # Nvidia
    "AMD.O",    # AMD
    "TSM.N",    # TSMC

    # Consumer Staples (Beverages)
    "KO.N",     # Coca-Cola
    "PEP.O",    # PepsiCo

    # Financials (Banking)
    "JPM.N",    # JPMorgan Chase
    "BAC.N",    # Bank of America
    "GS.N",     # Goldman Sachs
    "MS.N",     # Morgan Stanley

    # Energy
    "XOM.N",    # ExxonMobil
    "CVX.N",    # Chevron

    # E-Commerce / Cloud
    "AMZN.O",   # Amazon
    "MSFT.O",   # Microsoft
    "META.O",   # Meta
    "GOOGL.O",  # Alphabet

    # Healthcare (Pharma)
    "JNJ.N",    # Johnson & Johnson
    "PFE.N",    # Pfizer
]

# Candidate pairs by sector (for cointegration testing)
CANDIDATE_PAIRS = [
    ("NVDA.O", "AMD.O"),    # Semiconductors
    ("NVDA.O", "TSM.N"),    # Semiconductors
    ("KO.N",   "PEP.O"),    # Beverages
    ("JPM.N",  "BAC.N"),    # Banking
    ("GS.N",   "MS.N"),     # Banking
    ("XOM.N",  "CVX.N"),    # Energy
    ("AMZN.O", "MSFT.O"),   # Cloud
    ("META.O", "GOOGL.O"),  # Digital Advertising
    ("JNJ.N",  "PFE.N"),    # Healthcare Pharma
]


TICKER_NAMES = {
    "NVDA.O":  "Nvidia",
    "AMD.O":   "AMD",
    "TSM.N":   "TSMC",
    "KO.N":    "Coca-Cola",
    "PEP.O":   "PepsiCo",
    "JPM.N":   "JPMorgan",
    "BAC.N":   "BofA",
    "GS.N":    "Goldman",
    "MS.N":    "Morgan Stanley",
    "XOM.N":   "ExxonMobil",
    "CVX.N":   "Chevron",
    "AMZN.O":  "Amazon",
    "MSFT.O":  "Microsoft",
    "META.O":  "Meta",
    "GOOGL.O": "Alphabet",
    "JNJ.N":   "Johnson & Johnson",
    "PFE.N":   "Pfizer",
}

# Date ranges
TRAIN_START = "2019-01-01"
TRAIN_END   = "2023-12-31"   # in-sample
TEST_START  = "2024-01-01"   # out-of-sample
TEST_END    = "2025-12-31"

INTERVAL = "1d"

RISK_FREE_RATE = 0.02