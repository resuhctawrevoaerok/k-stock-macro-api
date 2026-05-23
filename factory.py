import FinanceDataReader as fdr
import requests
import json
import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse
from fastapi import FastAPI, Query, HTTPException
import uvicorn

# 1. FastAPI 앱 객체 생성
app = FastAPI(
    title="K-Stock Macro Intelligence API",
    description="Python 파이썬 기반 한국 기술주 매크로 분석 및 AI 리포트 자동 생성 API",
    version="1.0.0"
)

def get_english_news(search_query: str):
    query = urllib.parse.quote(search_query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        news_titles = []
        for item in root.findall('.//item')[:3]:
            news_titles.append(f"- {item.find('title').text}")
        return "\n".join(news_titles) if news_titles else "- No critical news headlines found."
    except Exception:
        return "- Failed to retrieve live news."

def calculate_rsi(df, period=14):
    if len(df) < period:
        return 50.0
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# ===================================================================
# [API 엔드포인트 설계] 사용자가 주소창에 변수를 실어 보내면 호출되는 함수
# 예: http://localhost:8000/report?ticker=000660&company_name=SK%20Hynix
# ===================================================================
@app.get("/report")
async def get_stock_report(
    ticker: str = Query(..., description="주식 종목 코드 (e.g., 005930)"),
    company_name: str = Query(..., description="구글 뉴스 검색용 영문 사명 (e.g., Samsung Electronics)"),
    rsi_period: int = Query(14, description="RSI 산출 기간 (기본값 14)"),
    openai_key: str = Query(..., description="사용자의 OpenAI API Key")
):
    try:
        # 주식 및 매크로 데이터 수집
        df_stock = fdr.DataReader(ticker)
        if df_stock.empty:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' 데이터를 찾을 수 없습니다.")
            
        latest_data = df_stock.iloc[-1]
        current_price = int(latest_data.get('Close', 0))
        volume = int(latest_data.get('Volume', 0))
        
        change_rate = 0.0
        if 'Chg' in latest_data:
            val = latest_data['Chg']
            change_rate = val * 100 if abs(val) < 1 and val != 0 else val
        change_rate_str = f"{change_rate:+.2f}%"
        
        real_rsi = calculate_rsi(df_stock, period=rsi_period)
        rsi_status = "Overbought" if real_rsi >= 70 else "Oversold" if real_rsi <= 30 else "Neutral"

        df_usd = fdr.DataReader("USD/KRW")
        current_usd_krw = df_usd.iloc[-1]['Close'] if not df_usd.empty else 0.0
        fx_interpretation = "Weakening KRW (High FX Rate)" if current_usd_krw >= 1350 else "Strengthening KRW (Normal/Low FX Rate)"
        
        df_sox = fdr.DataReader("^SOX")
        sox_change = 0.0
        if not df_sox.empty and len(df_sox) > 1:
            sox_change = ((df_sox.iloc[-1]['Close'] - df_sox.iloc[-2]['Close']) / df_sox.iloc[-2]['Close']) * 100
        sox_status = "Bullish global chip momentum" if sox_change > 0 else "Bearish global chip momentum"

        live_news = get_english_news(company_name)

        # AI 프롬프트 구성
        packaged_context = (
            f"[HARD MARKET METRICS]\n"
            f"- Ticker: {ticker} ({company_name})\n"
            f"- Samsung Price: {current_price:,} KRW ({change_rate_str})\n"
            f"- Technical RSI ({rsi_period}): {real_rsi:.2f} ({rsi_status})\n"
            f"- USD/KRW Rate: {current_usd_krw:,.2f} KRW -> Condition: {fx_interpretation}\n"
            f"- Global Chip Index (^SOX) Shift: {sox_change:+.2f}% -> Condition: {sox_status}\n\n"
            f"[CRITICAL LIVE NEWS HEADLINES]\n"
            f"{live_news}\n"
        )

        system_instruction = "You are a strict financial report generator. Reference the headlines and conditions directly. Do not invent facts."
        user_prompt = f"Write exactly 3 distinct paragraphs (Macro, Bull Case, Bear Case) based on this data:\n{packaged_context}"

        # OpenAI API 호출
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="OpenAI API 호출에 실패했습니다. 키를 확인해 주세요.")
            
        result = response.json()
        ai_analysis_text = result['choices'][0]['message']['content'].strip()
        
        # 표준화된 JSON 응답 반환
        return {
            "status": "success",
            "data": {
                "ticker": ticker,
                "company_name": company_name,
                "metrics": {
                    "price_krw": current_price,
                    "change": change_rate_str,
                    "rsi": round(real_rsi, 2),
                    "rsi_status": rsi_status,
                    "usd_krw": round(current_usd_krw, 2),
                    "fx_condition": fx_interpretation,
                    "sox_shift": round(sox_change, 2),
                    "sox_condition": sox_status,
                    "volume": volume
                },
                "news": live_news.split("\n"),
                "report": ai_analysis_text
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 로컬에서 서버 가동 테스트용
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)