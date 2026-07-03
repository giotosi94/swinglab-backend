"""
SwingLab Telegram Routes
Endpoint per test connessione + trigger manuale briefing/report.
"""

from fastapi import APIRouter
from datetime import datetime
from app.config import settings as app_settings
from app.services.telegram_bot import (
    send_telegram,
    send_daily_briefing,
    send_evening_report,
)

router = APIRouter()


# ============================================
# TEST — Verifica connessione bot
# ============================================

@router.post("/test")
async def telegram_test():
    """
    🧪 Test connessione Telegram bot.
    Invia un messaggio di test al chat configurato.
    """
    # Check config
    if not app_settings.TELEGRAM_BOT_TOKEN:
        return {
            "success": False,
            "error": "TELEGRAM_BOT_TOKEN not configured",
            "hint": "Add TELEGRAM_BOT_TOKEN to Render env vars",
        }
    if not app_settings.TELEGRAM_CHAT_ID:
        return {
            "success": False,
            "error": "TELEGRAM_CHAT_ID not configured",
            "hint": "Add TELEGRAM_CHAT_ID to Render env vars",
        }
    
    # Send test message
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    msg = (
        "🧪 <b>SwingLab Test Message</b>\n\n"
        f"✅ Connection OK!\n"
        f"⏰ Time: {now}\n"
        f"🤖 Bot is working!\n\n"
        "You will receive notifications for:\n"
        "  • 🟢 Buy/Sell executed\n"
        "  • 🚨 Software SL/TP triggered\n"
        "  • 📈 Trailing stops\n"
        "  • 🌅 Daily briefings\n"
        "  • 🌙 Evening reports"
    )
    
    result = await send_telegram(msg)
    
    if result:
        return {
            "success": True,
            "message": "Test message sent successfully!",
            "chat_id": app_settings.TELEGRAM_CHAT_ID,
            "sent_at": now,
        }
    else:
        return {
            "success": False,
            "error": "Failed to send message",
            "hints": [
                "Check TELEGRAM_BOT_TOKEN is correct",
                "Check TELEGRAM_CHAT_ID is correct",
                "Make sure you pressed START on the bot",
                "Check Render logs for details",
            ],
        }


# ============================================
# BRIEFING — Morning report manuale
# ============================================

@router.post("/briefing")
async def telegram_briefing():
    """
    🌅 Trigger manuale del Morning Briefing.
    Invia: market regime, top sector, portfolio, top picks.
    """
    try:
        await send_daily_briefing()
        return {
            "success": True,
            "message": "Morning briefing sent!",
            "sent_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ============================================
# REPORT — Evening report manuale
# ============================================

@router.post("/report")
async def telegram_report():
    """
    🌙 Trigger manuale dell'Evening Report.
    Invia: equity, trade del giorno, P&L, nuove posizioni.
    """
    try:
        await send_evening_report()
        return {
            "success": True,
            "message": "Evening report sent!",
            "sent_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ============================================
# STATUS — Info configurazione
# ============================================

@router.get("/status")
async def telegram_status():
    """
    ℹ️ Stato configurazione Telegram.
    Ritorna info se il bot è configurato (senza rivelare token).
    """
    return {
        "configured": bool(app_settings.TELEGRAM_BOT_TOKEN and app_settings.TELEGRAM_CHAT_ID),
        "has_token": bool(app_settings.TELEGRAM_BOT_TOKEN),
        "has_chat_id": bool(app_settings.TELEGRAM_CHAT_ID),
        "token_preview": (
            app_settings.TELEGRAM_BOT_TOKEN[:10] + "..." + app_settings.TELEGRAM_BOT_TOKEN[-4:]
            if app_settings.TELEGRAM_BOT_TOKEN else None
        ),
        "chat_id": app_settings.TELEGRAM_CHAT_ID if app_settings.TELEGRAM_CHAT_ID else None,
    }
