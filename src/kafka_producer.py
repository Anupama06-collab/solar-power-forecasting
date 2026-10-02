from kafka import KafkaProducer
import json

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Test solar data
solar_data = {
    "DATE_TIME": "2020-06-11 09:30:00",
    "IRRADIATION": 0.45,
    "AMBIENT_TEMPERATURE": 27.5,
    "MODULE_TEMPERATURE": 35.2,
    "HOUR": 9,
    "MINUTE": 30,
    "DAY_OF_YEAR": 163,
    "AC_POWER_LAG_1": 0.0,
    "AC_POWER_LAG_2": 0.0,
    "AC_POWER_LAG_3": 0.0
}

# Send data
producer.send("solar-data", solar_data)
producer.flush()

print("Solar data sent successfully!")
print(solar_data)

producer.close()