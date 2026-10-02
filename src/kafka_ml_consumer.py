from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
import pandas as pd

MODEL_PATH = "models/final_solar_model.pkl"

# ============================================================
# LOAD FINAL MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("========== SOLAR ML KAFKA CONSUMER ==========")
print("Final model loaded successfully!")
print("Waiting for solar data...")
print()


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(
    "solar-data",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="solar-ml-consumer"
)


# ============================================================
# KAFKA PRODUCER
# ============================================================

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


# ============================================================
# MODEL FEATURES
# ============================================================

features = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR",
    "MINUTE",
    "DAY_OF_YEAR",
    "AC_POWER_LAG_1",
    "AC_POWER_LAG_2",
    "AC_POWER_LAG_3"
]


# ============================================================
# PROCESS SOLAR DATA
# ============================================================

for message in consumer:

    try:

        data = message.value

        print("========== NEW SOLAR DATA ==========")
        print(data)

        # Create model input
        X = pd.DataFrame([data])[features]

        # Predict AC power
        prediction = model.predict(X)[0]

        print()
        print("========== SOLAR POWER FORECAST ==========")
        print("Forecast Time       :", data["DATE_TIME"])
        print("Predicted AC Power  :", round(prediction, 2))

        # ====================================================
        # PREPARE FORECAST OUTPUT
        # ====================================================

        forecast = {
            "DATE_TIME": data["DATE_TIME"],
            "PREDICTED_AC_POWER": float(prediction)
        }

        # Send forecast to Kafka
        producer.send(
            "solar-forecast",
            value=forecast
        )

        producer.flush()

        print()
        print("Forecast sent to Kafka topic: solar-forecast")
        print()

    except Exception as e:

        print()
        print("ERROR processing solar data:")
        print(e)
        print()