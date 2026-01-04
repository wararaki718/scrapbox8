from .models import FinancialData

def calculate_dividend_safety_score(data: FinancialData) -> tuple[float, str]:
    """
    Calculates the dividend safety score based on financial data.
    - Free CF = Operating CF + Investing CF
    - Score = Free CF / Dividend Payout
    """
    if data.dividend_payout == 0:
        return float('inf'), "配当支払なし"

    try:
        free_cash_flow = data.operating_cf + data.investing_cf
        score = free_cash_flow / data.dividend_payout
        judgment = "安全" if score >= 1.2 else "減配警戒"
        return score, judgment
    except ZeroDivisionError:
        return 0.0, "配当支払額が0のため計算不可"
