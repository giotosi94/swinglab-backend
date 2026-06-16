from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DB_NAME]
    print(f"✅ Connected to MongoDB: {settings.DB_NAME}")

    # ==========================================
    # FASE 20: MongoDB Indexes
    # ==========================================
    try:
        # trade_history — query più frequenti
        await db.trade_history.create_index("order_id", unique=True, sparse=True)
        await db.trade_history.create_index([("ticker", 1), ("side", 1), ("date", -1)])
        await db.trade_history.create_index([("side", 1), ("date", -1)])
        await db.trade_history.create_index([("ticker", 1), ("side", 1), ("sell_linked", 1)])
        await db.trade_history.create_index("date")

        # assets — lookup per ticker
        await db.assets.create_index("ticker", unique=True)
        await db.assets.create_index("sector_code")
        await db.assets.create_index("setup_score")

        # stock_bars — lookup per ticker
        await db.stock_bars.create_index("ticker", unique=True)

        # sectors
        await db.sectors.create_index("code", unique=True)

        # agent decisions — query per tipo e data
        for agent in ["macro_analyst", "alpha_strategist", "risk_manager", "executor"]:
            col = db[f"agent_decisions_{agent}"]
            await col.create_index([("created_at", -1)])
            await col.create_index([("type", 1), ("created_at", -1)])

        # trailing_stops
        await db.trailing_stops.create_index("ticker", unique=True)

        # watchlist (futuro)
        await db.watchlist.create_index("ticker", unique=True)

        print("✅ MongoDB indexes created")
    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")


async def close_db():
    global client
    if client:
        client.close()
        print("❌ MongoDB connection closed")


def get_db():
    return db
