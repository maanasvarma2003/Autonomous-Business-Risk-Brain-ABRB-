import json
import time
import random
from typing import Callable, Any

class KafkaSimulator:
    """
    Simulates Kafka message streaming for development and testing.
    """
    def __init__(self, topic: str):
        self.topic = topic
        self.running = False

    def produce_mock_events(self, callback: Callable[[Any], None], interval: float = 1.0):
        """
        Produces mock risk events.
        """
        self.running = True
        print(f"Starting Kafka Simulator for topic: {self.topic}")
        while self.running:
            event = self._generate_event()
            callback(event)
            time.sleep(interval)

    def _generate_event(self) -> dict:
        event_types = ["transaction", "login", "audit", "customer_update"]
        event_type = random.choice(event_types)
        
        return {
            "timestamp": time.time(),
            "event_type": event_type,
            "entity_id": f"USER_{random.randint(1000, 9999)}",
            "data": self._generate_event_data(event_type)
        }

    def _generate_event_data(self, event_type: str) -> dict:
        if event_type == "transaction":
            return {"amount": random.uniform(10, 5000), "currency": "USD"}
        elif event_type == "login":
            return {"ip": f"192.168.1.{random.randint(1, 255)}", "status": "success"}
        elif event_type == "audit":
            return {"action": "access_control", "policy_id": "GDPR_ART_5"}
        else:
            return {"tenure": random.randint(1, 72), "contract": "Month-to-month"}

    def stop(self):
        self.running = False

def process_event(event: dict):
    """
    Example event processor.
    """
    print(f"Processing Event: {event['event_type']} for {event['entity_id']}")

if __name__ == "__main__":
    simulator = KafkaSimulator(topic="risk-events")
    try:
        simulator.produce_mock_events(process_event, interval=2.0)
    except KeyboardInterrupt:
        simulator.stop()
        print("Simulator stopped.")
