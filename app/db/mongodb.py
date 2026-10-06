from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from app.config import settings

client: AsyncIOMotorClient = None
db = None


async def _safe_index(collection, keys, **kwargs):
    try:
        await collection.create_index(keys, **kwargs)
        return True
    except Exception as e:
        print(f"⚠️ Index warning {collection.name} {kwargs.get('name', keys)}: {e}")
        return False


async def _create_max_strategy_indexes():
    signals = db.max_strategy_signals
    plans = db.max_strategy_plans
    results = [
        await _safe_index(
            signals,
            [("setup_key", ASCENDING)],
            name="uniq_setup_key",
            unique=True,
            partialFilterExpression={"setup_key": {"$type": "string"}},
        ),
        await _safe_index(signals, [("ticker", ASCENDING), ("signal_date", DESCENDING)], name="ticker_signal_date"),
        await _safe_index(signals, [("plan_key", ASCENDING)], name="plan_key"),
        await _safe_index(signals, [("plan_id", ASCENDING)], name="plan_id", sparse=True),
        await _safe_index(signals, [("validation_cohort", ASCENDING), ("signal_date", DESCENDING)], name="cohort_signal_date"),
        await _safe_index(
            plans,
            [("plan_key", ASCENDING)],
            name="uniq_plan_key",
            unique=True,
            partialFilterExpression={"plan_key": {"$type": "string"}},
        ),
        await _safe_index(plans, [("ticker", ASCENDING), ("lifecycle_state", ASCENDING)], name="ticker_lifecycle_state"),
        await _safe_index(plans, [("is_active", ASCENDING), ("updated_at", DESCENDING)], name="active_updated_at"),
        await _safe_index(db.stock_bars_4h, [("ticker", ASCENDING)], name="ticker"),
    ]
    print(f"✅ Max Strategy indexes: {sum(results)}/{len(results)} ok")


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DB_NAME]
    print(f"✅ Connected to MongoDB: {settings.DB_NAME}")

    try:
        await db.trade_history.create_index("order_id", unique=True, sparse=True)
        await db.trade_history.create_index([("ticker", 1), ("side", 1), ("date", -1)])
        await db.trade_history.create_index([("side", 1), ("date", -1)])
        await db.trade_history.create_index([("ticker", 1), ("side", 1), ("sell_linked", 1)])
        await db.trade_history.create_index("date")

        await db.assets.create_index("ticker", unique=True)
        await db.assets.create_index("sector_code")
        await db.assets.create_index("setup_score")

        await db.stock_bars.create_index("ticker", unique=True)

        await db.sectors.create_index("code", unique=True)

        for agent in ["macro_analyst", "alpha_strategist", "risk_manager", "executor"]:
            col = db[f"agent_decisions_{agent}"]
            await col.create_index([("created_at", -1)])
            await col.create_index([("type", 1), ("created_at", -1)])

        await db.trailing_stops.create_index("ticker", unique=True)

        await db.watchlist.create_index("ticker", unique=True)

        print("✅ MongoDB indexes created")
    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")

    await _create_max_strategy_indexes()


async def close_db():
    global client
    if client:
        client.close()
        print("❌ MongoDB connection closed")


def get_db():
    return db
