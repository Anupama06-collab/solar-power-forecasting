from kafka import KafkaConsumer
import json

print("========== SOLAR FORECAST OUTPUT CONSUMER ==========")
print("Waiting for forecasts...\n")

consumer = KafkaConsumer(
    "solar-forecast",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

for message in consumer:
    forecast = message.value

    print("========== NEW FORECAST ==========")
    print(f"Forecast Time       : {forecast['DATE_TIME']}")
    print(f"Predicted AC Power  : {forecast['PREDICTED_AC_POWER']:.2f}")
    print()